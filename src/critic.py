"""
Critic: evaluates generated answer against source sections.
Verdict: PASS | REGEN | ABSTAIN.
F3: structured output (instructor + Pydantic CriticOutput), anti-over-abstain prompt.
F4: run_critic() — REGEN loop + graceful PASS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

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
    answer: Optional[str] = None
    retried: bool = False
    used_ensemble: bool = False


class CriticOutput(BaseModel):
    chain_of_thought: List[str] = Field(
        default_factory=list,
        description="Key facts checked — max 5 bullets, ≤15 words each",
    )
    reasoning: List[str] = Field(
        default_factory=list,
        description="Decision reasoning — exactly 1 bullet for PASS, max 2 for REGEN/ABSTAIN",
    )
    verdict: Literal["PASS", "REGEN", "ABSTAIN"]
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the verdict (0.0 low, 1.0 high)",
    )

    @field_validator("chain_of_thought", "reasoning", mode="before")
    @classmethod
    def _coerce_str_list(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            v = [v]
        out: list[str] = []
        for item in v:
            if isinstance(item, dict):
                out.append("; ".join(f"{k}: {val}" for k, val in item.items()))
            else:
                out.append(str(item))
        return out


_SYSTEM_PROMPT = """\
Du bist ein strenger Fakten-Prüfer für ERGO Versicherungsantworten (B2B-Agent).
Prüfe, ob die generierte Antwort durch die gegebenen Quell-Abschnitte gestützt wird.

BEWERTUNGSREGELN:
- PASS: Jede Sachaussage ist durch den Kontext belegt (Paraphrasen und Synthese erlaubt).
- REGEN: Antwort enthält einen spezifischen Fehler (falsche Zahl, falsches Datum, falsche Bedingung),
  der aber im Kontext korrigierbar ist.
- ABSTAIN: Antwort erfindet konkrete Beträge, Daten oder genannte Bedingungen,
  die vollständig im Kontext fehlen — ODER Kontext ist völlig irrelevant zur Frage.

WICHTIG — DEFAULT TO PASS:
- Hedging-Phrasen sind KEINE Halluzination → PASS. Beispiele für Hedging:
  "nicht explizit erwähnt", "Quellen geben nicht an", "nicht separat aufgeführt",
  "nicht in den vorliegenden Abschnitten", "nicht direkt beschrieben",
  "kann nicht direkt verglichen werden", "laut den vorliegenden Bedingungen nicht aufgeführt".
- Unvollständige Antworten (nur Teil der Frage beantwortet) → PASS, nicht ABSTAIN.
- ABSTAIN NUR bei erfundenen konkreten Beträgen (€), Daten oder named conditions,
  die im Kontext explizit fehlen oder widerlegt werden.
- REGEN NUR bei konkretem, korrigierbarem Sachfehler — NICHT für Stil, Länge oder Hedging.

EXCLUSION_QUERY Sonderregel:
- Wenn der Kontext Abschnitte enthält wie "Nicht versichert", "Ausschlüsse",
  "wogegen ist ... nicht versichert" — UND die Antwort diese Ausschlüsse korrekt zitiert → PASS.
- Allgemeine Ausschlüsse (z.B. Kriegsereignisse, Kernenergie, Vorsatz des VN) gelten für alle
  Tarife einer Sparte. Deren Nennung in der Antwort auf Basis des Kontexts ist KEIN Halluzinieren → PASS.
- Tarif-spezifische Frage + allgemeine Ausschlüsse im Kontext → PASS (nicht ABSTAIN).

Antworte ausschließlich im geforderten strukturierten Format.
"""


class Critic:
    def __init__(
        self,
        client: Any,
        model: str = "qwen/qwen3-32b",
        _wrap_instructor: bool = True,
    ) -> None:
        if _wrap_instructor:
            import instructor
            self._client = instructor.from_groq(client, mode=instructor.Mode.JSON)
        else:
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

        output: CriticOutput = self._client.chat.completions.create(
            model=self._model,
            response_model=CriticOutput,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        reason = " ".join(output.reasoning) if output.reasoning else ""
        verdict = CriticVerdict(output.verdict)
        result = CriticResult(
            verdict=verdict,
            reason=reason,
            confidence=output.confidence_score or 0.0,
            answer=answer if verdict != CriticVerdict.ABSTAIN else None,
        )
        _log.info("step_done", step="critic", verdict=result.verdict.value,
                  confidence=result.confidence, reason=result.reason)
        return result


def run_critic(
    query: str,
    answer: str,
    sections: list[dict[str, Any]],
    critic: "Critic",
    generate_fn: Callable[[], str],
    ensemble_critic: Optional["Critic"] = None,
    enable_ensemble: bool = False,
) -> CriticResult:
    """Orchestrate primary evaluate + REGEN loop + optional ensemble + graceful PASS.

    Verdicts never leak REGEN to the caller — only PASS or ABSTAIN returned.
    """
    try:
        primary = critic.evaluate(query, answer, sections)
    except Exception:
        _log.warning("critic_primary_failed", query=query[:80])
        return CriticResult(
            verdict=CriticVerdict.PASS,
            reason="primary failed — graceful pass",
            confidence=0.0,
            answer=answer,
            retried=False,
        )

    if primary.verdict == CriticVerdict.ABSTAIN:
        return CriticResult(
            verdict=CriticVerdict.ABSTAIN,
            reason=primary.reason,
            confidence=primary.confidence,
            answer=None,
            retried=False,
        )

    if primary.verdict == CriticVerdict.REGEN:
        new_answer = generate_fn()
        try:
            recheck = critic.evaluate(query, new_answer, sections)
        except Exception:
            _log.warning("critic_recheck_failed", query=query[:80])
            return CriticResult(
                verdict=CriticVerdict.PASS,
                reason="recheck failed — graceful pass",
                confidence=0.0,
                answer=new_answer,
                retried=True,
            )
        if recheck.verdict == CriticVerdict.ABSTAIN:
            return CriticResult(
                verdict=CriticVerdict.ABSTAIN,
                reason=recheck.reason,
                confidence=recheck.confidence,
                answer=None,
                retried=True,
            )
        # recheck=PASS or recheck=REGEN → accept regenerated answer, then ensemble
        return _maybe_ensemble(query, new_answer, sections, recheck, new_answer, retried=True,
                               ensemble_critic=ensemble_critic, enable_ensemble=enable_ensemble)

    # primary.verdict == PASS — run optional ensemble
    return _maybe_ensemble(query, answer, sections, primary, answer, retried=False,
                           ensemble_critic=ensemble_critic, enable_ensemble=enable_ensemble)


def _maybe_ensemble(
    query: str,
    current_answer: str,
    sections: list[dict[str, Any]],
    primary_result: CriticResult,
    original_answer: str,
    retried: bool,
    ensemble_critic: Optional["Critic"],
    enable_ensemble: bool,
) -> CriticResult:
    if not enable_ensemble or ensemble_critic is None:
        return CriticResult(
            verdict=CriticVerdict.PASS,
            reason=primary_result.reason,
            confidence=primary_result.confidence,
            answer=current_answer,
            retried=retried,
            used_ensemble=False,
        )
    try:
        ens = ensemble_critic.evaluate(query, current_answer, sections)
    except Exception:
        _log.warning("critic_ensemble_failed", query=query[:80])
        return CriticResult(
            verdict=CriticVerdict.PASS,
            reason=primary_result.reason,
            confidence=primary_result.confidence,
            answer=current_answer,
            retried=retried,
            used_ensemble=False,
        )
    if ens.verdict == CriticVerdict.ABSTAIN:
        return CriticResult(
            verdict=CriticVerdict.ABSTAIN,
            reason=ens.reason,
            confidence=ens.confidence,
            answer=None,
            retried=retried,
            used_ensemble=True,
        )
    return CriticResult(
        verdict=CriticVerdict.PASS,
        reason=ens.reason,
        confidence=ens.confidence,
        answer=current_answer,
        retried=retried,
        used_ensemble=True,
    )
