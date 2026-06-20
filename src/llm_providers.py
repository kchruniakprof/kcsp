"""
LLM client factories.

Groq = runtime (low latency): query_expansion, generator, critic.
OpenRouter = batch (quality): enrichment.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import instructor
import openai

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def groq_client(api_key: Optional[str] = None) -> Any:
    """instructor-wrapped OpenAI client pointed at Groq."""
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing (.env, not committed)")
    raw = openai.OpenAI(api_key=key, base_url=_GROQ_BASE_URL)
    return instructor.from_openai(raw, mode=instructor.Mode.MD_JSON)


def openrouter_client(api_key: Optional[str] = None) -> Any:
    """instructor-wrapped OpenAI client pointed at OpenRouter."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing (.env, not committed)")
    raw = openai.OpenAI(api_key=key, base_url=_OPENROUTER_BASE_URL)
    return instructor.from_openai(raw)
