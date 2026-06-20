"""
Per-step model registry.

Groq models: latency-sensitive steps (runtime path).
OpenRouter models: quality-sensitive batch steps.
"""

REGISTRY: dict[str, str] = {
    # Groq — runtime
    "query_expansion": "meta-llama/llama-4-scout-17b-16e-instruct",
    "generator_verbatim": "llama-3.1-8b-instant",
    "generator_compare": "llama-3.3-70b-versatile",
    "llm_selector": "llama-3.3-70b-versatile",
    "critic": "qwen/qwen3-32b",
    # OpenRouter — batch
    "enrichment": "openai/gpt-4o-mini-2024-07-18",
}
