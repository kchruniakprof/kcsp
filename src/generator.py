"""
Generator: verbatim markdown (llama-3.1-8b-instant) + COMPARE diff-step (llama-3.3-70b-versatile).
Verbatim mode: stitch relevant sections as-is, minimal rewrite prompt.
Compare mode: side-by-side diff synthesis for tarif comparison queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.observability import get_logger

_log = get_logger("generator")


class AnswerMode(str, Enum):
    VERBATIM = "VERBATIM"
    COMPARE  = "COMPARE"


@dataclass
class GeneratedAnswer:
    answer: str
    sources: list[int]
    mode: AnswerMode
    breadcrumbs: list[str]


_VERBATIM_SYSTEM = """\
Du bist ein Assistent für ERGO Versicherung (internes B2B-Tool).
Antworte NUR auf Basis der gegebenen Bedingungsabschnitte.
Gib den relevanten Inhalt WÖRTLICH aus Markdown wieder — kein Umschreiben.
Markiere die Quelle mit dem Breadcrumb am Ende jedes Abschnitts.
Falls keine passende Information vorhanden ist, antworte mit leerem String.
"""

_COMPARE_SYSTEM = """\
Du bist ein Assistent für ERGO Versicherung (internes B2B-Tool).
Erstelle einen strukturierten Vergleich der gegebenen Tarifabschnitte.
Verwende eine Markdown-Tabelle oder Aufzählung. Nur Fakten aus dem Text — kein Hinzufügen.
"""


_MODEL_VERBATIM = "llama-3.1-8b-instant"
_MODEL_COMPARE  = "llama-3.3-70b-versatile"


class Generator:
    def __init__(self, client: Any, model: str = _MODEL_VERBATIM) -> None:
        self._client = client
        self._model = model

    def generate(
        self,
        query: str,
        sections: list[dict[str, Any]],
        mode: AnswerMode = AnswerMode.VERBATIM,
    ) -> GeneratedAnswer:
        _log.info("step_start", step="generator", mode=mode.value,
                  sections_count=len(sections), query=query)

        if not sections:
            _log.info("step_done", step="generator", mode=mode.value, answer_len=0, reason="no_sections")
            return GeneratedAnswer(answer="", sources=[], mode=mode, breadcrumbs=[])

        if mode == AnswerMode.VERBATIM:
            result = self._verbatim(query, sections)
        else:
            result = self._compare(query, sections)

        _log.info("step_done", step="generator", mode=mode.value,
                  answer_len=len(result.answer), sources=result.sources)
        return result

    # ------------------------------------------------------------------

    def _verbatim(self, query: str, sections: list[dict[str, Any]]) -> GeneratedAnswer:
        context = self._build_context(sections)
        prompt = f"Frage: {query}\n\nAbschnitte:\n{context}"
        answer = self._call(prompt, system=_VERBATIM_SYSTEM)
        return GeneratedAnswer(
            answer=answer,
            sources=[s["section_id"] for s in sections],
            mode=AnswerMode.VERBATIM,
            breadcrumbs=[s["breadcrumb"] for s in sections],
        )

    def _compare(self, query: str, sections: list[dict[str, Any]]) -> GeneratedAnswer:
        context = self._build_context(sections)
        prompt = f"Vergleichsfrage: {query}\n\nAbschnitte:\n{context}"
        answer = self._call(prompt, system=_COMPARE_SYSTEM, model=_MODEL_COMPARE)
        return GeneratedAnswer(
            answer=answer,
            sources=[s["section_id"] for s in sections],
            mode=AnswerMode.COMPARE,
            breadcrumbs=[s["breadcrumb"] for s in sections],
        )

    def _call(self, prompt: str, system: str, model: Optional[str] = None) -> str:
        resp = self._client.chat.completions.create(
            model=model or self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content

    @staticmethod
    def _build_context(sections: list[dict[str, Any]]) -> str:
        parts = []
        for s in sections:
            bc = s.get("breadcrumb", "")
            md = s.get("markdown", "")
            parts.append(f"<!-- {bc} -->\n{md}")
        return "\n\n".join(parts)
