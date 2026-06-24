"""TracingClient: instructor client wrapper that records per-call cost into TraceCollector."""
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
    """Return cost in EUR for the given token counts."""
    if prompt_tokens == 0 and completion_tokens == 0:
        return 0.0
    rates = _PRICING.get(model_id)
    if rates is None:
        # Unknown model: use a conservative default (Groq cheapest tier)
        input_rate, output_rate = 0.046, 0.046
    else:
        input_rate, output_rate = rates
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


class TracingClient:
    """Drop-in replacement for instructor client. Intercepts create_with_completion."""

    def __init__(
        self,
        instructor_client: Any,
        model_id: str,
        provider: str,
        name: str,
    ) -> None:
        self._client = instructor_client
        self._model_id = model_id
        self._provider = provider
        self._name = name

    def chat(self) -> "TracingClient":
        return self  # proxy .chat.completions

    @property
    def completions(self) -> "TracingClient":
        return self

    def create_with_completion(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        t0 = time.perf_counter()
        result, completion = self._client.chat.completions.create_with_completion(
            *args, **kwargs
        )
        duration_ms = int((time.perf_counter() - t0) * 1000)

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost_eur = _calc_cost(self._model_id, prompt_tokens, completion_tokens)

        collector = get_active_collector()
        if collector is not None:
            collector.add_step(
                StepCost(
                    name=self._name,
                    model_id=self._model_id,
                    provider=self._provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_eur=cost_eur,
                    duration_ms=duration_ms,
                )
            )

        return result, completion
