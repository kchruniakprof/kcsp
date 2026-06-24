"""Thread-local trace collector for per-request LLM step tracking."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StepCost:
    name: str           # e.g. "QueryExpansion"
    model_id: str       # e.g. "meta-llama/llama-4-scout-17b-16e-instruct"
    provider: str       # e.g. "groq" | "openrouter"
    prompt_tokens: int
    completion_tokens: int
    cost_eur: float
    duration_ms: int


StepTrace = StepCost  # alias for compatibility


class TraceCollector:
    def __init__(self) -> None:
        self.steps: list[StepCost] = []
        self.stage_markers: dict[str, str] = {}
        self.query_expansion_detail: Optional[dict] = None

    @property
    def total_cost_eur(self) -> float:
        return sum(s.cost_eur for s in self.steps)

    def add_step(self, step: StepCost) -> None:
        self.steps.append(step)


_local = threading.local()


def get_active_collector() -> Optional[TraceCollector]:
    return getattr(_local, "collector", None)


def set_active_collector(collector: Optional[TraceCollector]) -> None:
    _local.collector = collector
