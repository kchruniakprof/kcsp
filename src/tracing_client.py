"""TracingClient: wraps instructor/raw client, records per-call cost into TraceCollector."""
from __future__ import annotations

import time
from typing import Any

from src.tracing import StepCost, get_active_collector

# EUR per 1M tokens (input, output)
# Groq USD rates × 0.92 EUR/USD
_PRICING: dict[str, tuple[float, float]] = {
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.18, 0.18),   # Groq ~$0.20
    "llama-3.1-8b-instant": (0.046, 0.046),                       # Groq ~$0.05
    "llama-3.3-70b-versatile": (0.59, 0.79),                      # Groq ~$0.72
    "qwen/qwen3-32b": (0.29, 0.29),                               # Groq
    "openai/gpt-4o-mini-2024-07-18": (0.138, 0.552),              # OpenRouter
}
_EUR_RATE = 0.92  # USD→EUR


def _calc_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    if prompt_tokens == 0 and completion_tokens == 0:
        return 0.0
    input_rate, output_rate = _PRICING.get(model_id, (0.046, 0.046))
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


def _record(name: str, model_id: str, provider: str, usage: Any, duration_ms: int) -> None:
    collector = get_active_collector()
    if collector is None:
        return
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    collector.add_step(StepCost(
        name=name,
        model_id=model_id,
        provider=provider,
        prompt_tokens=pt,
        completion_tokens=ct,
        cost_eur=_calc_cost(model_id, pt, ct),
        duration_ms=duration_ms,
    ))


class TracingClient:
    """
    Transparent proxy around an instructor or raw OpenAI/Groq client.
    Intercepts .chat.completions.create() and .create_with_completion() to record
    timing + token cost into the thread-local TraceCollector.

    Wrapping is safe on the lru-cached RAG singleton: the TracingClient is stateless
    per-call — it looks up the collector via thread-local on each invocation.
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        provider: str,
        name: str,
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._provider = provider
        self._name = name

    @property
    def chat(self) -> "TracingClient":
        return self

    @property
    def completions(self) -> "TracingClient":
        return self

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept .chat.completions.create(). Works for both instructor and raw clients."""
        # Use the model kwarg if provided (e.g. Generator switches models per call)
        call_model = kwargs.get("model") or self._model_id
        t0 = time.perf_counter()

        inner = self._client.chat.completions
        # Instructor clients expose create_with_completion — use it to get usage
        if hasattr(inner, "create_with_completion"):
            result, completion = inner.create_with_completion(*args, **kwargs)
            usage = getattr(completion, "usage", None)
        else:
            result = inner.create(*args, **kwargs)
            usage = getattr(result, "usage", None)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        _record(self._name, call_model, self._provider, usage, duration_ms)
        return result

    def create_with_completion(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Intercept .chat.completions.create_with_completion() (instructor pattern)."""
        call_model = kwargs.get("model") or self._model_id
        t0 = time.perf_counter()
        result, completion = self._client.chat.completions.create_with_completion(
            *args, **kwargs
        )
        duration_ms = int((time.perf_counter() - t0) * 1000)
        usage = getattr(completion, "usage", None)
        _record(self._name, call_model, self._provider, usage, duration_ms)
        return result, completion
