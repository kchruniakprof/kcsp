"""
ContextPruner: sentence-based text pruning with dual-view output.

Produces PrunedChunk where:
  verbatim_text = original, always unchanged (for generator / user)
  pruned_text   = shortened version (for LLM internals only)

Bypass: chunk < 2500 chars → pruned_text == verbatim_text
Empty-guard: if pruning yields empty → fallback to verbatim_text
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_BYPASS_THRESHOLD = 2500
_DEFAULT_MAX_CHARS = 2000


@dataclass
class PrunedChunk:
    verbatim_text: str
    pruned_text: str


class ContextPruner:
    def __init__(self, max_chars: int = _DEFAULT_MAX_CHARS) -> None:
        self._max_chars = max_chars

    def prune(self, chunk: str) -> PrunedChunk:
        if len(chunk) < _BYPASS_THRESHOLD:
            return PrunedChunk(verbatim_text=chunk, pruned_text=chunk)

        pruned = self._sentence_prune(chunk)
        if not pruned:
            pruned = chunk

        return PrunedChunk(verbatim_text=chunk, pruned_text=pruned)

    def _sentence_prune(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result: list[str] = []
        total = 0
        for sent in sentences:
            if total + len(sent) > self._max_chars:
                break
            result.append(sent)
            total += len(sent) + 1
        return " ".join(result).strip()
