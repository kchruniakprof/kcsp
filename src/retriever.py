"""
Retriever: BGE-M3 embeddings + vectorized dot-product scoring.
D5 changes:
  - filters to is_retrieval_unit=True in __init__
  - DocFilter replaces inline sparte/tarif filter
  - ContextPruner produces dual-view: markdown (verbatim) + pruned_markdown
  - RetrievalResult gains pruned_markdown field
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.context_pruner import ContextPruner
from src.observability import get_logger

_log = get_logger("retriever")

_MIN_SECTION_CHUNKS = 3


@dataclass
class RetrievalResult:
    section_id: int
    doc_id: str
    sparte: str
    tarif: Optional[str]
    heading: str
    markdown: str
    pruned_markdown: str
    breadcrumb: str
    score: float
    section_types: list[str]
    topic_tags: list[str]


class Retriever:
    def __init__(
        self,
        sections: list[dict[str, Any]],
        embedder: Any,
        sec_embs: Optional[np.ndarray] = None,
        pruner: Optional[ContextPruner] = None,
    ) -> None:
        # Only retrieval units enter the index
        self._sections = [s for s in sections if s.get("is_retrieval_unit", True)]
        self._embedder = embedder
        self._pruner = pruner or ContextPruner()

        if sec_embs is not None:
            self._sec_embs = np.array(sec_embs, dtype=np.float32)
        else:
            self._sec_embs = self._build_index(self._sections, embedder)

    @staticmethod
    def _build_index(sections: list[dict[str, Any]], embedder: Any) -> np.ndarray:
        if not sections:
            return np.empty((0, 1), dtype=np.float32)
        texts = [s["heading"] + " " + s["markdown"][:512] for s in sections]
        vecs = embedder.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)

    def _build_result(self, score: float, idx: int) -> RetrievalResult:
        s = self._sections[idx]
        md = s["markdown"]
        pruned = self._pruner.prune(md).pruned_text
        return RetrievalResult(
            section_id=s["section_id"],
            doc_id=s["doc_id"],
            sparte=s["sparte"],
            tarif=s.get("tarif"),
            heading=s["heading"],
            markdown=md,
            pruned_markdown=pruned,
            breadcrumb=s["breadcrumb"],
            score=score,
            section_types=list(s.get("section_types", [])),
            topic_tags=list(s.get("topic_tags", [])),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        section_types: Optional[list[str]] = None,
        doc_filter: Optional[Any] = None,
        query_obj: Optional[Any] = None,
    ) -> list[RetrievalResult]:
        return self.retrieve_multi(
            queries=[query],
            top_k=top_k,
            section_types=section_types,
            doc_filter=doc_filter,
            query_obj=query_obj,
        )

    def retrieve_multi(
        self,
        queries: list[str],
        top_k: int = 5,
        section_types: Optional[list[str]] = None,
        doc_filter: Optional[Any] = None,
        query_obj: Optional[Any] = None,
    ) -> list[RetrievalResult]:
        """Retrieve using multiple queries (normalized_query + paraphrases).

        Scoring:
          1. DocFilter → candidate positions (pre-filter by doc_id)
          2. section_type filter with fallback guard (< _MIN_SECTION_CHUNKS → drop)
          3. Matrix dot product: candidate_embs @ q_vecs.T → max across queries
          4. Sort desc, return top_k
        """
        _log.info("step_start", step="retriever", queries_count=len(queries),
                  query_primary=queries[0] if queries else "", top_k=top_k)

        if not self._sections:
            _log.info("step_done", step="retriever", results_count=0, reason="empty_index")
            return []

        # ── Step 1: DocFilter ─────────────────────────────────────────────────
        if doc_filter is not None:
            # Use provided query_obj (ExpandedQuery) so DocFilter reads real sparte_hint/domain_terms.
            # Fall back to empty proxy only for legacy callers that omit query_obj.
            _q = query_obj if query_obj is not None else type("_Q", (), {"sparte_hint": None, "domain_terms": []})()
            allowed_ids = doc_filter.filter(_q)
            if allowed_ids is None:
                # None = no-filter → search all sections (F2 will pass real query_obj)
                positions = list(range(len(self._sections)))
            elif not allowed_ids:
                # frozenset() = active filter with no matches → empty result
                _log.info("step_done", step="retriever", results_count=0,
                          reason="doc_filter_empty")
                return []
            else:
                positions = [
                    i for i, s in enumerate(self._sections)
                    if s["doc_id"] in allowed_ids
                ]
                if not positions:
                    _log.info("step_done", step="retriever", results_count=0,
                              reason="no_candidates_after_doc_filter")
                    return []
        else:
            positions = list(range(len(self._sections)))

        # ── Step 2: section_type filter with fallback guard ───────────────────
        typed: list[int] = []
        if section_types:
            typed = [
                i for i in positions
                if any(t in self._sections[i].get("section_types", []) for t in section_types)
            ]
            if len(typed) >= _MIN_SECTION_CHUNKS:
                positions = typed

        # ── Step 3: vectorized scoring ────────────────────────────────────────
        cand_embs = self._sec_embs[positions]

        q_vecs = self._embedder.encode(queries, normalize_embeddings=True)
        q_vecs = np.array(q_vecs, dtype=np.float32)
        if q_vecs.ndim == 1:
            q_vecs = q_vecs[np.newaxis, :]

        scores_matrix = cand_embs @ q_vecs.T
        best_scores = scores_matrix.max(axis=1)

        # ── Step 4: top_k ─────────────────────────────────────────────────────
        if len(best_scores) <= top_k:
            top_idx = np.argsort(best_scores)[::-1]
        else:
            top_idx = np.argpartition(best_scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(best_scores[top_idx])[::-1]]

        results = [
            self._build_result(float(best_scores[ci]), positions[ci])
            for ci in top_idx
        ]

        _log.info("step_done", step="retriever", results_count=len(results),
                  top_scores=[round(r.score, 4) for r in results],
                  top_sections=[r.section_id for r in results])
        return results
