"""
Retriever: BGE-M3 embeddings + vectorized dot-product scoring.
D5 changes:
  - filters to is_retrieval_unit=True in __init__
  - DocFilter replaces inline sparte/tarif filter
  - ContextPruner produces dual-view: markdown (verbatim) + pruned_markdown
  - RetrievalResult gains pruned_markdown field
G5 changes:
  - CrossEncoderReranker with lazy model loading
  - Retriever accepts optional reranker + pool_k parameter
D3 changes:
  - sparse_embs: list[dict[int, float]] for BGE-M3 lexical weights
  - _compute_rrf_scores: pure RRF fusion (k=60)
  - retrieve_multi: RRF when sparse available + embedder.encode_sparse; dense-only fallback
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.context_pruner import ContextPruner
from src.doc_filter import GENERIC_BLOCKLIST
from src.model_registry import REGISTRY
from src.observability import get_logger

_log = get_logger("retriever")

_SECTION_TYPE_BOOST = np.float32(0.04)
_RRF_K = 60


def _compute_rrf_scores(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    k: int = _RRF_K,
) -> np.ndarray:
    """Reciprocal Rank Fusion: 1/(k+dense_rank) + 1/(k+sparse_rank)."""
    n = len(dense_scores)
    dense_rank = np.empty(n, dtype=np.int32)
    sparse_rank = np.empty(n, dtype=np.int32)
    dense_rank[np.argsort(dense_scores)[::-1]] = np.arange(n)
    sparse_rank[np.argsort(sparse_scores)[::-1]] = np.arange(n)
    return (1.0 / (k + dense_rank) + 1.0 / (k + sparse_rank)).astype(np.float32)


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
    shared_tarifs: list[str] = field(default_factory=list)


class CrossEncoderReranker:
    """Cross-encoder reranker with lazy model loading.

    The model is NOT loaded at __init__ or at import time.
    It is loaded on the first call to rerank().
    """

    def __init__(self, model_name: str = REGISTRY["reranker"]) -> None:
        self._model_name = model_name
        self._model = None  # lazy

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        pairs = [(query, r.heading + " " + r.markdown[:512]) for r in results]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [r for _, r in ranked]


class Retriever:
    def __init__(
        self,
        sections: list[dict[str, Any]],
        embedder: Any,
        sec_embs: Optional[np.ndarray] = None,
        sparse_embs: Optional[list] = None,
        pruner: Optional[ContextPruner] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        # Only retrieval units enter the index
        self._sections = [s for s in sections if s.get("is_retrieval_unit", True)]
        self._embedder = embedder
        self._pruner = pruner or ContextPruner()
        self._reranker = reranker
        self._sparse_embs = sparse_embs  # list[dict[int, float]] or None

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
        pool_k: int = 20,
    ) -> list[RetrievalResult]:
        return self.retrieve_multi(
            queries=[query],
            top_k=top_k,
            section_types=section_types,
            doc_filter=doc_filter,
            query_obj=query_obj,
            pool_k=pool_k,
        )

    def retrieve_multi(
        self,
        queries: list[str],
        top_k: int = 5,
        section_types: Optional[list[str]] = None,
        doc_filter: Optional[Any] = None,
        query_obj: Optional[Any] = None,
        pool_k: int = 20,
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

        # ── D1: exact-term force include (within DocFilter gate) ─────────────
        _forced_positions: set[int] = set()
        if query_obj is not None:
            _raw = list(getattr(query_obj, "domain_terms", None) or [])
            _raw += list(getattr(query_obj, "topic_tags", None) or [])
            _qterms = [t.lower().strip() for t in _raw
                       if t.strip() and t not in GENERIC_BLOCKLIST]
            if _qterms:
                for _pos in positions:
                    _md = self._sections[_pos].get("markdown", "").lower()
                    if any(_t in _md for _t in _qterms):
                        _forced_positions.add(_pos)

        # ── Step 2: section_type soft boost (no hard-drop) ───────────────────
        _section_types_set = set(section_types) if section_types else set()

        # ── Step 3: vectorized scoring ────────────────────────────────────────
        cand_embs = self._sec_embs[positions]

        q_vecs = self._embedder.encode(queries, normalize_embeddings=True)
        q_vecs = np.array(q_vecs, dtype=np.float32)
        if q_vecs.ndim == 1:
            q_vecs = q_vecs[np.newaxis, :]

        scores_matrix = cand_embs @ q_vecs.T
        best_scores = scores_matrix.max(axis=1)

        # ── D3: RRF fusion when sparse embeddings + encoder available ─────────
        _use_rrf = (
            self._sparse_embs is not None
            and hasattr(self._embedder, "encode_sparse")
        )
        if _use_rrf:
            q_sparse = self._embedder.encode_sparse(queries[0])
            if isinstance(q_sparse, list):
                q_sparse = q_sparse[0]
            sparse_scores = np.zeros(len(positions), dtype=np.float32)
            for ci, pos in enumerate(positions):
                s_emb = self._sparse_embs[pos]
                if s_emb:
                    sparse_scores[ci] = sum(
                        s_emb.get(tid, 0.0) * w for tid, w in q_sparse.items()
                    )
            best_scores = _compute_rrf_scores(best_scores, sparse_scores)

        # A1: additive +0.04 boost for chunks matching ≥1 requested type (once, not stacked)
        if _section_types_set:
            for ci, pos in enumerate(positions):
                if set(self._sections[pos].get("section_types", [])) & _section_types_set:
                    best_scores[ci] += _SECTION_TYPE_BOOST

        # ── Step 4: A2 dynamic pool_k policy ─────────────────────────────────
        # pool ≤50 → give reranker the full filtered set; pool >50 → cap at 30
        n_candidates = len(positions)
        pool_k_effective = n_candidates if n_candidates <= 50 else 30
        fetch_k = pool_k_effective if (self._reranker is not None and pool_k_effective > top_k) else top_k

        if len(best_scores) <= fetch_k:
            top_idx = np.argsort(best_scores)[::-1]
        else:
            top_idx = np.argpartition(best_scores, -fetch_k)[-fetch_k:]
            top_idx = top_idx[np.argsort(best_scores[top_idx])[::-1]]

        candidates = [
            self._build_result(float(best_scores[ci]), positions[ci])
            for ci in top_idx
        ]
        pool_embs: list[np.ndarray] = [cand_embs[ci] for ci in top_idx]

        # D1: append forced candidates not already in dense pool
        if _forced_positions:
            _in_pool = {positions[ci] for ci in top_idx}
            for _ci, _pos in enumerate(positions):
                if _pos in _forced_positions and _pos not in _in_pool:
                    candidates.append(self._build_result(float(best_scores[_ci]), _pos))
                    pool_embs.append(cand_embs[_ci])

        # ── D2: near-duplicate dedup (before reranker) ───────────────────────
        _dedup_threshold = float(REGISTRY.get("dedup_threshold", 0.98))
        if len(candidates) > 1:
            _e = np.array(pool_embs, dtype=np.float32)
            _norms = np.linalg.norm(_e, axis=1, keepdims=True)
            _norms = np.where(_norms == 0, 1.0, _norms)
            _ne = _e / _norms
            _sim = _ne @ _ne.T
            _merged: set[int] = set()
            _deduped: list[RetrievalResult] = []
            for _i in range(len(candidates)):
                if _i in _merged:
                    continue
                _cluster = [_i]
                for _j in range(_i + 1, len(candidates)):
                    if _j not in _merged and _sim[_i, _j] > _dedup_threshold:
                        _cluster.append(_j)
                        _merged.add(_j)
                _rep_i = max(_cluster, key=lambda _k: candidates[_k].score)
                _rep = candidates[_rep_i]
                if len(_cluster) > 1:
                    _rep.shared_tarifs = [
                        candidates[_k].tarif for _k in _cluster if candidates[_k].tarif
                    ]
                _deduped.append(_rep)
            candidates = _deduped

        # ── Step 5: optional cross-encoder reranking ──────────────────────────
        if self._reranker is not None and pool_k_effective > top_k:
            results = self._reranker.rerank(queries[0], candidates)[:top_k]
        else:
            results = candidates[:top_k]

        _log.info("step_done", step="retriever", results_count=len(results),
                  top_scores=[round(r.score, 4) for r in results],
                  top_sections=[r.section_id for r in results])
        return results
