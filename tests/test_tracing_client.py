"""Unit tests for TracingClient — uses mocked instructor client."""
from unittest.mock import MagicMock
from src.tracing import TraceCollector, set_active_collector, get_active_collector
from src.tracing_client import TracingClient, _calc_cost


def make_mock_client(prompt_tokens=100, completion_tokens=50, model_id="llama-3.1-8b-instant"):
    mock = MagicMock()
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    completion = MagicMock()
    completion.usage = usage
    mock.chat.completions.create_with_completion.return_value = ("parsed_result", completion)
    return TracingClient(mock, model_id=model_id, provider="groq", name="Test")


def test_cost_positive_when_tokens_nonzero():
    collector = TraceCollector()
    set_active_collector(collector)
    client = make_mock_client(prompt_tokens=1000, completion_tokens=500)
    result, _ = client.create_with_completion(response_model=None, messages=[])
    assert collector.total_cost_eur > 0
    assert len(collector.steps) == 1
    assert collector.steps[0].cost_eur > 0
    set_active_collector(None)


def test_zero_cost_when_zero_tokens():
    cost = _calc_cost("llama-3.1-8b-instant", 0, 0)
    assert cost == 0.0


def test_step_recorded_in_collector():
    collector = TraceCollector()
    set_active_collector(collector)
    client = make_mock_client()
    client.create_with_completion(response_model=None, messages=[])
    assert len(collector.steps) == 1
    step = collector.steps[0]
    assert step.name == "Test"
    assert step.prompt_tokens == 100
    assert step.completion_tokens == 50
    set_active_collector(None)


def test_no_collector_doesnt_crash():
    set_active_collector(None)
    client = make_mock_client()
    result, _ = client.create_with_completion(response_model=None, messages=[])
    assert result == "parsed_result"
