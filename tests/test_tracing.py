"""Unit tests for TraceCollector (src/tracing.py)."""
import threading
from src.tracing import (
    StepCost,
    StepTrace,
    TraceCollector,
    get_active_collector,
    set_active_collector,
)


def _make_step(**kwargs) -> StepCost:
    defaults = dict(
        name="TestStep",
        model_id="llama-3.1-8b-instant",
        provider="groq",
        prompt_tokens=100,
        completion_tokens=50,
        cost_eur=0.007,
        duration_ms=120,
    )
    defaults.update(kwargs)
    return StepCost(**defaults)


def test_step_cost_alias():
    """StepTrace is an alias for StepCost."""
    assert StepTrace is StepCost


def test_add_step_appends():
    col = TraceCollector()
    col.add_step(_make_step())
    assert len(col.steps) == 1


def test_total_cost_eur_sums_steps():
    col = TraceCollector()
    col.add_step(_make_step(cost_eur=0.01))
    col.add_step(_make_step(cost_eur=0.02))
    assert abs(col.total_cost_eur - 0.03) < 1e-9


def test_total_cost_empty():
    col = TraceCollector()
    assert col.total_cost_eur == 0.0


def test_set_and_get_active_collector():
    col = TraceCollector()
    set_active_collector(col)
    assert get_active_collector() is col
    set_active_collector(None)
    assert get_active_collector() is None


def test_thread_local_isolation():
    """Each thread gets its own collector."""
    results: dict[str, object] = {}

    def worker(name: str, col: TraceCollector) -> None:
        set_active_collector(col)
        results[name] = get_active_collector()

    col_a = TraceCollector()
    col_b = TraceCollector()
    t1 = threading.Thread(target=worker, args=("a", col_a))
    t2 = threading.Thread(target=worker, args=("b", col_b))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert results["a"] is col_a
    assert results["b"] is col_b
    # Main thread collector unchanged
    assert get_active_collector() is None


def test_stage_markers_dict():
    col = TraceCollector()
    col.stage_markers["retrieval"] = "done"
    assert col.stage_markers["retrieval"] == "done"


def test_query_expansion_detail_default_none():
    col = TraceCollector()
    assert col.query_expansion_detail is None
