"""
Critic: evaluates generated answer against source sections.
Verdict: PASS | REGEN | ABSTAIN. Uses Groq llama-3.3-70b, temperature=0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from src.observability import get_logger

_log = get_logger("critic")


class CriticVerdict(str, Enum):
    PASS    = "PASS"
    REGEN   = "REGEN"
    ABSTAIN = "ABSTAIN"


@dataclass
class CriticResult:
    verdict: CriticVerdict
    reason: str
    confidence: float


class _CriticResponse(BaseModel):
    verdict: CriticVerdict
    reason: str
    confidence: float


_SYSTEM_PROMPT = """\
Du bist ein Qualitätsprüfer für ERGO Versicherungs-Antworten (B2B).
Prüfe, ob die Antwort korrekt und vollständig aus den gegebenen Abschnitten ableitbar ist.

Antworte NUR mit einem JSON-Objekt:
- verdict: "PASS" (korrekt und vollständig) | "REGEN" (regenerieren, z.B. zu kurz) | "ABSTAIN" (Halluzination oder keine Quellen)
- reason: kurze Begründung auf Deutsch
- confidence: 0.0–1.0

Kein Kommentar, nur JSON.
"""


class Critic:
    def __init__(self, client: Any, model: str = "qwen/qwen3-32b") -> None:
        self._client = client
        self._model = model

    def evaluate(
        self,
        query: str,
        answer: str,
        sections: list[dict[str, Any]],
    ) -> CriticResult:
        _log.info("step_start", step="critic", model=self._model,
                  answer_len=len(answer), sections_count=len(sections))

        context = "\n\n".join(
            s.get("markdown", "") for s in sections
        ) if sections else "(keine Quellen)"

        prompt = (
            f"Frage: {query}\n\n"
            f"Quell-Abschnitte:\n{context}\n\n"
            f"Generierte Antwort:\n{answer}"
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        parsed = _CriticResponse(**data)
        result = CriticResult(
            verdict=parsed.verdict,
            reason=parsed.reason,
            confidence=parsed.confidence,
        )
        _log.info("step_done", step="critic", verdict=result.verdict.value,
                  confidence=result.confidence, reason=result.reason)
        return result
