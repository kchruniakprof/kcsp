"""
DocFilter: Protocol + adapters for document-level pre-filtering.

Replaces inline Sparte/Tarif filter in retriever.py (done in D5).
CompositeDocFilter returns union; empty union = no-filter signal.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Protocol, List

import pandas as pd

from src.constants import SPARTES

_HAUSRAT_KEYWORDS_RE = re.compile(r"Hausrat|Haushalt|Wohnung", re.IGNORECASE)


def _detect_tarif(normalized_query: str, tarif_names: list[str]) -> Optional[str]:
    """Deterministic word-boundary tarif match. Longest-first; rider aliases from split('+')."""
    candidates = sorted(tarif_names, key=len, reverse=True)
    for t in candidates:
        tokens = [t] + t.split("+")[1:]  # full compound + rider tokens only (not base)
        for tok in tokens:
            if re.search(rf"\b{re.escape(tok)}\b", normalized_query, re.IGNORECASE):
                return t
    return None


def resolve_doc_set(
    sparte_hints: list[str],
    tarif: Optional[str],
    documents_df: pd.DataFrame,
    normalized_query: str = "",
) -> Optional[frozenset[str]]:
    """Map sparte_hints + tarif → frozenset[doc_id], or None = no filter."""
    if not sparte_hints:
        return None

    hints = [s for s in sparte_hints if s in SPARTES]
    if not hints:
        return None

    if len(hints) == 1:
        sparte = hints[0]
        mask = documents_df["sparte"] == sparte
        if tarif is not None:
            mask &= documents_df["tarif"] == tarif
        result = frozenset(documents_df.loc[mask, "doc_id"].tolist())
    else:
        # multi-sparte: union, ignore tarif
        mask = documents_df["sparte"].isin(hints)
        result = frozenset(documents_df.loc[mask, "doc_id"].tolist())

    # related_sparte safety-net: Glas/Schmuck + Hausrat keyword in query → add Hausrat
    glas_schmuck = {"Glas", "Schmuck"}
    if (
        glas_schmuck & set(hints)
        and "Hausrat" not in hints
        and normalized_query
        and _HAUSRAT_KEYWORDS_RE.search(normalized_query)
    ):
        hausrat_ids = frozenset(
            documents_df.loc[documents_df["sparte"] == "Hausrat", "doc_id"].tolist()
        )
        result = result | hausrat_ids

    return result


GENERIC_BLOCKLIST: frozenset[str] = frozenset({
    "Versicherung",
    "Schaden",
    "Vertrag",
    "Versicherer",
    "Versicherungsnehmer",
    "Prämie",
    "Leistung",
})


class DocFilter(Protocol):
    def filter(self, query: Any) -> Optional[frozenset[str]]:
        """Return frozenset[doc_id] to restrict, or None = no-filter (search all)."""
        ...


class ProductDetectorAdapter:
    """Maps sparte_hints + tarif → frozenset[doc_id] via resolve_doc_set.

    tarif is supplied by caller (detected deterministically by RAGAssistant).
    sparte_hints are read from query object.
    """

    def __init__(
        self,
        documents_df: pd.DataFrame,
        tarif: Optional[str] = None,
        # legacy compat: sparte= accepted but ignored in favour of query.sparte_hints
        sparte: Optional[str] = None,
    ) -> None:
        self._docs = documents_df
        self._tarif = tarif
        self._legacy_sparte = sparte  # kept for old tests that pass sparte= directly

    def filter(self, query: Any) -> Optional[frozenset[str]]:
        sparte_hints: list[str] = list(getattr(query, "sparte_hints", None) or [])
        normalized_query: str = getattr(query, "normalized_query", "") or ""

        # legacy fallback: if query has no sparte_hints but old sparte= was provided
        if not sparte_hints and self._legacy_sparte:
            sparte_hints = [self._legacy_sparte]

        return resolve_doc_set(sparte_hints, self._tarif, self._docs, normalized_query)


class RareTagMatcherAdapter:
    """Maps domain_terms → frozenset[doc_id] via topic_tags in sections.

    Generic blocklist terms are ignored.
    Empty domain_terms → frozenset().
    """

    def __init__(
        self,
        sections_df: pd.DataFrame,
        subsections_df: pd.DataFrame,
    ) -> None:
        self._combined = pd.concat(
            [
                sections_df[["doc_id", "topic_tags"]],
                subsections_df[["doc_id", "topic_tags"]],
            ],
            ignore_index=True,
        )

    def filter(self, query: Any) -> Optional[frozenset[str]]:
        terms = getattr(query, "domain_terms", []) or []
        rare = [t for t in terms if t not in GENERIC_BLOCKLIST]
        if not rare:
            return None  # no rare terms → no-filter fallback

        matched_ids: set[str] = set()
        for _, row in self._combined.iterrows():
            tags = row["topic_tags"]
            if tags is None or len(tags) == 0:
                continue
            tag_set = set(tags) if not isinstance(tags, set) else tags
            if tag_set & set(rare):
                matched_ids.add(str(row["doc_id"]))

        if not matched_ids:
            return None  # rare terms present but no tag match → no-filter fallback
        return frozenset(matched_ids)


class CompositeDocFilter:
    """Gate + optional rare narrow.

    adapters[0] = gate (ProductDetectorAdapter): None → no-filter; frozenset → restrict
    adapters[1] = rare (RareTagMatcherAdapter, optional): narrows within gate

    Semantics:
      gate=None → None (search all)
      gate set, rare=None → gate
      gate set, rare set → gate ∩ rare if non-empty, else gate (cross-sparte rare → ignore)
    """

    def __init__(self, adapters: list[DocFilter]) -> None:
        self._adapters = adapters

    def filter(self, query: Any) -> Optional[frozenset[str]]:
        if not self._adapters:
            return None
        gate_result = self._adapters[0].filter(query)
        if gate_result is None:
            return None  # gate says no-filter → pass-through
        rare_result = self._adapters[1].filter(query) if len(self._adapters) > 1 else None
        if rare_result is None:
            return gate_result
        narrowed = gate_result & rare_result
        return narrowed if narrowed else gate_result  # cross-sparte rare → ignore, keep gate
