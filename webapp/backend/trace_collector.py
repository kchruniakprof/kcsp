"""Re-exports TraceCollector from src.tracing (created by webapp_runner agent)."""
try:
    from src.tracing import (  # noqa: F401
        TraceCollector, StepTrace, StepCost,
        get_active_collector, set_active_collector,
    )
except ImportError:
    # Fallback: define minimal TraceCollector inline for tests
    import threading
    from dataclasses import dataclass, field
    from typing import Optional, Any

    @dataclass
    class StepCost:
        name: str
        model_id: str
        provider: str
        prompt_tokens: int
        completion_tokens: int
        cost_eur: float
        duration_ms: int

    StepTrace = StepCost  # alias

    _local = threading.local()

    class TraceCollector:
        def __init__(self):
            self.steps: list[StepCost] = []
            self.stage_markers: dict = {}
            self.query_expansion_detail: Optional[dict] = None

        @property
        def total_cost_eur(self) -> float:
            return sum(s.cost_eur for s in self.steps)

        def add_step(self, step: StepCost) -> None:
            self.steps.append(step)

    def get_active_collector() -> Optional[TraceCollector]:
        return getattr(_local, "collector", None)

    def set_active_collector(c: Optional[TraceCollector]) -> None:
        _local.collector = c
