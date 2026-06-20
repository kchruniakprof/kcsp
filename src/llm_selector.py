"""
ContextSelector: LLM-based top-k reranker with abstain-on-low-score.

Key KCSP difference from DKV: low confidence → Abstain (not BruteForce fallback).
In insurance domain, low confidence = hallucination risk → better to say "I don't know".

Uses pruned_text from PrunedChunk for LLM; passes verbatim_text unchanged to output.
EMBED_THRESHOLD = None → never abstain (safe default before E1 calibration).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.context_pruner import PrunedChunk
from src.llm_providers import groq_client
from src.model_registry import REGISTRY
from src.observability import get_logger

_log = get_logger("llm_selector")

_DEFAULT_MODEL = REGISTRY["llm_selector"]


@dataclass
class SelectedChunk:
    verbatim_text: str
    confidence: float


@dataclass
class Abstain:
    reason: str


class _SelectorResponse(BaseModel):
    selected_index: int = Field(..., ge=0, description="0-based index of the best candidate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    reason: str = Field(..., description="One-sentence rationale")


_SYSTEM_PROMPT = """\
You are a context selector for ERGO P&C insurance Q&A (B2B internal tool).
Given a user query and a list of candidate text chunks, select the SINGLE BEST chunk
that most directly and accurately answers the query.

Respond with:
- selected_index: 0-based index of the best chunk
- confidence: 0.0-1.0 (1.0 = perfectly answers the query, 0.0 = no relevant content)
- reason: one sentence explaining why this chunk is the best

If no chunk is relevant, pick index 0 with confidence 0.0.
"""


class ContextSelector:
    def __init__(
        self,
        client: Optional[Any] = None,
        threshold: Optional[float] = None,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._client = client if client is not None else groq_client()
        self._threshold = threshold
        self._model = model

    def select(
        self,
        candidates: list[PrunedChunk],
        query: str = "",
    ) -> SelectedChunk | Abstain:
        if not candidates:
            return Abstain(reason="no candidates provided")

        resp = self._call_llm(candidates, query)

        if self._threshold is not None and resp.confidence < self._threshold:
            _log.info(
                "abstain",
                confidence=resp.confidence,
                threshold=self._threshold,
                reason=resp.reason,
            )
            return Abstain(reason=f"confidence {resp.confidence:.3f} < threshold {self._threshold}")

        idx = max(0, min(resp.selected_index, len(candidates) - 1))
        verbatim = candidates[idx].verbatim_text
        _log.info("selected", index=idx, confidence=resp.confidence)
        return SelectedChunk(verbatim_text=verbatim, confidence=resp.confidence)

    def _call_llm(
        self,
        candidates: list[PrunedChunk],
        query: str,
    ) -> _SelectorResponse:
        chunks_text = "\n\n".join(
            f"[{i}] {c.pruned_text[:800]}"
            for i, c in enumerate(candidates)
        )
        user_msg = f"Query: {query}\n\nCandidates:\n{chunks_text}"

        return self._client.chat.completions.create(
            model=self._model,
            response_model=_SelectorResponse,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
