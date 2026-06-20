"""
DocFilter: Protocol + adapters for document-level pre-filtering.

Replaces inline Sparte/Tarif filter in retriever.py (done in D5).
CompositeDocFilter returns union; empty union = no-filter signal.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, List

import pandas as pd


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
    """Maps sparte/tarif → frozenset[doc_id] via documents.parquet lookup.

    sparte/tarif in constructor take priority; falls back to query.sparte_hint.
    tarif=None → return all doc_ids for the sparte.
    Unknown tarif → frozenset() (no match, no exception).
    """

    def __init__(
        self,
        documents_df: pd.DataFrame,
        sparte: Optional[str] = None,
        tarif: Optional[str] = None,
    ) -> None:
        self._docs = documents_df
        self._sparte = sparte
        self._tarif = tarif

    def filter(self, query: Any) -> Optional[frozenset[str]]:
        sparte = self._sparte
        if sparte is None:
            sparte = getattr(query, "sparte_hint", None)
        if not sparte:
            return None  # no sparte identified → no-filter signal

        mask = self._docs["sparte"] == sparte
        if self._tarif is not None:
            mask &= self._docs["tarif"] == self._tarif

        matched = self._docs.loc[mask, "doc_id"]
        return frozenset(matched.tolist())


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
    """Union of adapter results.
    None = no-filter (all adapters returned None — search everywhere).
    frozenset() = active filter with no matches → empty result.
    """

    def __init__(self, adapters: list[DocFilter]) -> None:
        self._adapters = adapters

    def filter(self, query: Any) -> Optional[frozenset[str]]:
        non_none: List[frozenset[str]] = []
        for adapter in self._adapters:
            r = adapter.filter(query)
            if r is not None:
                non_none.append(r)
        if not non_none:
            return None  # all adapters returned None → no-filter
        result: frozenset[str] = frozenset()
        for fs in non_none:
            result = result | fs
        return result
