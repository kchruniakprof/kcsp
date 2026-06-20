"""
EmbeddingPruner: BGE-M3 sentence-scoring pruner with dual-view output.

Same PrunedChunk contract as context_pruner:
  verbatim_text = original, always unchanged
  pruned_text   = top-scoring sentences (by cosine sim to chunk centroid)

Bypass: chunk < 2500 chars → pruned_text == verbatim_text (no model needed)
Empty-guard: if pruning empty → fallback to verbatim_text
Lazy-load: model loaded on first non-bypass call; not reloaded if already set.
"""
from __future__ import annotations

import re
from typing import Optional

from src.context_pruner import PrunedChunk

_BYPASS_THRESHOLD = 2500
_DEFAULT_MAX_CHARS = 2000
_BGE_MODEL_NAME = "BAAI/bge-m3"


class EmbeddingPruner:
    def __init__(self, max_chars: int = _DEFAULT_MAX_CHARS) -> None:
        self._max_chars = max_chars
        self._model: Optional[object] = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(_BGE_MODEL_NAME)

    def prune(self, chunk: str) -> PrunedChunk:
        if len(chunk) < _BYPASS_THRESHOLD:
            return PrunedChunk(verbatim_text=chunk, pruned_text=chunk)

        self._load_model()
        pruned = self._embed_prune(chunk)
        if not pruned:
            pruned = chunk

        return PrunedChunk(verbatim_text=chunk, pruned_text=pruned)

    def _embed_prune(self, text: str) -> str:
        import numpy as np

        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            return ""

        embeddings = self._model.encode(  # type: ignore[union-attr]
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        centroid = embeddings.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        scores = embeddings @ centroid_norm

        max_chars = getattr(self, "_max_chars", _DEFAULT_MAX_CHARS)
        ranked = sorted(zip(scores, sentences), reverse=True)
        result: list[str] = []
        total = 0
        for _, sent in ranked:
            if total + len(sent) > max_chars:
                break
            result.append(sent)
            total += len(sent) + 1

        return " ".join(result).strip()
