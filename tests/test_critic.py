"""Tests for critic — F3: structured output + anti-over-abstain prompt."""
import pytest
from unittest.mock import MagicMock

from src.critic import Critic, CriticVerdict, CriticResult, CriticOutput


# ---------------------------------------------------------------------------
# CriticVerdict enum
# ---------------------------------------------------------------------------

def test_verdict_has_pass():
    assert CriticVerdict.PASS in CriticVerdict


def test_verdict_has_regen():
    assert CriticVerdict.REGEN in CriticVerdict


def test_verdict_has_abstain():
    assert CriticVerdict.ABSTAIN in CriticVerdict


def test_verdict_exactly_three():
    assert len(CriticVerdict) == 3


# ---------------------------------------------------------------------------
# CriticOutput — Pydantic model + _coerce_str_list validator
# ---------------------------------------------------------------------------

def test_critic_output_basic():
    out = CriticOutput(
        chain_of_thought=["fact 1", "fact 2"],
        reasoning=["All facts supported"],
        verdict="PASS",
        confidence_score=0.95,
    )
    assert out.verdict == "PASS"
    assert out.confidence_score == 0.95


def test_coerce_str_list_flattens_dict():
    """_coerce_str_list: list-of-dict → list-of-str (anti-crash for instructor retries)."""
    out = CriticOutput(
        chain_of_thought=[{"claim": "supported", "source": "§1"}],
        reasoning=["ok"],
        verdict="PASS",
        confidence_score=0.9,
    )
    assert all(isinstance(s, str) for s in out.chain_of_thought)


def test_coerce_str_list_handles_none():
    out = CriticOutput(
        chain_of_thought=None,
        reasoning=None,
        verdict="PASS",
        confidence_score=0.9,
    )
    assert out.chain_of_thought == []
    assert out.reasoning == []


def test_coerce_str_list_handles_single_str():
    out = CriticOutput(
        chain_of_thought="single string",
        reasoning="single",
        verdict="PASS",
        confidence_score=0.9,
    )
    assert isinstance(out.chain_of_thought, list)


# ---------------------------------------------------------------------------
# CriticResult — fields including new `answer` field
# ---------------------------------------------------------------------------

def test_critic_result_fields():
    r = CriticResult(
        verdict=CriticVerdict.PASS,
        reason="Antwort korrekt und vollständig.",
        confidence=0.95,
    )
    assert r.verdict == CriticVerdict.PASS
    assert r.confidence == 0.95


def test_critic_result_has_answer_field():
    r = CriticResult(verdict=CriticVerdict.PASS, reason="ok", confidence=0.9)
    assert hasattr(r, "answer")
    assert r.answer is None  # default


def test_critic_result_reason_str():
    r = CriticResult(verdict=CriticVerdict.ABSTAIN, reason="Keine Grundlage.", confidence=0.1)
    assert isinstance(r.reason, str)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_QUERY = "Was ist im Hausrat Smart versichert?"
_SECTIONS = [
    {"section_id": 1, "heading": "1. Was ist versichert",
     "markdown": "## 1. Was ist versichert\nVersichert ist der Hausrat."},
]
_ANSWER_GOOD = "Versichert ist der Hausrat."
_ANSWER_HALLUCINATED = "Der Hausrat ist bis 10 Millionen Euro versichert."


def _make_critic(verdict: str = "PASS", reason: str = "ok", confidence: float = 0.9) -> Critic:
    """Build Critic with mock instructor-style client (returns CriticOutput directly)."""
    client = MagicMock()
    client.chat.completions.create.return_value = CriticOutput(
        chain_of_thought=["checked"],
        reasoning=[reason],
        verdict=verdict,
        confidence_score=confidence,
    )
    return Critic(client=client, _wrap_instructor=False)


# ---------------------------------------------------------------------------
# evaluate() — verdict classification
# ---------------------------------------------------------------------------

def test_evaluate_returns_critic_result():
    c = _make_critic("PASS")
    result = c.evaluate(_QUERY, _ANSWER_GOOD, _SECTIONS)
    assert isinstance(result, CriticResult)


def test_evaluate_pass_verdict():
    c = _make_critic("PASS")
    result = c.evaluate(_QUERY, _ANSWER_GOOD, _SECTIONS)
    assert result.verdict == CriticVerdict.PASS


def test_evaluate_regen_verdict():
    c = _make_critic("REGEN", "Antwort zu kurz.")
    result = c.evaluate(_QUERY, "kurz", _SECTIONS)
    assert result.verdict == CriticVerdict.REGEN


def test_evaluate_abstain_verdict():
    c = _make_critic("ABSTAIN", "Keine Grundlage.", 0.1)
    result = c.evaluate(_QUERY, _ANSWER_HALLUCINATED, _SECTIONS)
    assert result.verdict == CriticVerdict.ABSTAIN


def test_evaluate_confidence_in_range():
    c = _make_critic("PASS", "ok", 0.88)
    result = c.evaluate(_QUERY, _ANSWER_GOOD, _SECTIONS)
    assert 0.0 <= result.confidence <= 1.0


def test_evaluate_reason_nonempty():
    c = _make_critic("PASS", "Antwort korrekt.")
    result = c.evaluate(_QUERY, _ANSWER_GOOD, _SECTIONS)
    assert result.reason.strip() != ""


# ---------------------------------------------------------------------------
# temperature=0
# ---------------------------------------------------------------------------

def test_critic_uses_temperature_zero():
    calls = []
    client = MagicMock()

    def fake_create(**kwargs):
        calls.append(kwargs)
        return CriticOutput(
            chain_of_thought=["ok"],
            reasoning=["ok"],
            verdict="PASS",
            confidence_score=0.9,
        )

    client.chat.completions.create.side_effect = fake_create
    c = Critic(client=client, _wrap_instructor=False)
    c.evaluate(_QUERY, _ANSWER_GOOD, _SECTIONS)

    assert len(calls) > 0
    assert calls[0].get("temperature", 999) == 0


# ---------------------------------------------------------------------------
# Edge: empty sections
# ---------------------------------------------------------------------------

def test_empty_sections_abstain():
    c = _make_critic("ABSTAIN", "Keine Quellen.", 0.0)
    result = c.evaluate(_QUERY, "", [])
    assert result.verdict == CriticVerdict.ABSTAIN


# ---------------------------------------------------------------------------
# F4: run_critic — REGEN loop + graceful PASS
# ---------------------------------------------------------------------------

from src.critic import run_critic


def _make_critic_for_run(verdicts: list[str]) -> Critic:
    """Critic whose evaluate() cycles through verdicts in order."""
    calls = iter(verdicts)
    client = MagicMock()
    def _create(**kwargs):
        v = next(calls)
        return CriticOutput(chain_of_thought=["ok"], reasoning=["ok"], verdict=v, confidence_score=0.9)
    client.chat.completions.create.side_effect = _create
    return Critic(client=client, _wrap_instructor=False)


def test_run_critic_pass_returns_pass():
    c = _make_critic_for_run(["PASS"])
    result = run_critic(_QUERY, _ANSWER_GOOD, _SECTIONS, c, generate_fn=lambda: "regen")
    assert result.verdict == CriticVerdict.PASS
    assert result.retried is False
    assert result.answer == _ANSWER_GOOD


def test_run_critic_abstain_returns_abstain():
    c = _make_critic_for_run(["ABSTAIN"])
    result = run_critic(_QUERY, _ANSWER_HALLUCINATED, _SECTIONS, c, generate_fn=lambda: "regen")
    assert result.verdict == CriticVerdict.ABSTAIN
    assert result.answer is None
    assert result.retried is False


def test_run_critic_regen_then_pass():
    """REGEN → regenerate → recheck=PASS → CriticResult(PASS, answer=new_answer, retried=True)."""
    c = _make_critic_for_run(["REGEN", "PASS"])
    new_answer = "Regenerierte Antwort"
    result = run_critic(_QUERY, "kurz", _SECTIONS, c, generate_fn=lambda: new_answer)
    assert result.verdict == CriticVerdict.PASS
    assert result.answer == new_answer
    assert result.retried is True


def test_run_critic_regen_then_regen_accepts():
    """REGEN → recheck=REGEN → still PASS (don't double-block)."""
    c = _make_critic_for_run(["REGEN", "REGEN"])
    new_answer = "Zweite Antwort"
    result = run_critic(_QUERY, "kurz", _SECTIONS, c, generate_fn=lambda: new_answer)
    assert result.verdict == CriticVerdict.PASS
    assert result.answer == new_answer
    assert result.retried is True


def test_run_critic_regen_then_abstain():
    """REGEN → recheck=ABSTAIN → CriticResult(ABSTAIN)."""
    c = _make_critic_for_run(["REGEN", "ABSTAIN"])
    result = run_critic(_QUERY, "kurz", _SECTIONS, c, generate_fn=lambda: "regen")
    assert result.verdict == CriticVerdict.ABSTAIN
    assert result.answer is None
    assert result.retried is True


def test_run_critic_generate_fn_called_on_regen():
    """generate_fn() called exactly once when primary=REGEN."""
    c = _make_critic_for_run(["REGEN", "PASS"])
    calls = []
    def gen():
        calls.append(1)
        return "new"
    run_critic(_QUERY, "kurz", _SECTIONS, c, generate_fn=gen)
    assert len(calls) == 1


def test_run_critic_generate_fn_not_called_on_pass():
    c = _make_critic_for_run(["PASS"])
    calls = []
    run_critic(_QUERY, _ANSWER_GOOD, _SECTIONS, c, generate_fn=lambda: calls.append(1) or "x")
    assert calls == []


def test_run_critic_graceful_pass_on_primary_exception():
    """Primary raises → log warning → CriticResult(PASS, original answer)."""
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("API down")
    c = Critic(client=client, _wrap_instructor=False)
    result = run_critic(_QUERY, _ANSWER_GOOD, _SECTIONS, c, generate_fn=lambda: "x")
    assert result.verdict == CriticVerdict.PASS
    assert result.answer == _ANSWER_GOOD
    assert result.retried is False


# ---------------------------------------------------------------------------
# F5: ensemble
# ---------------------------------------------------------------------------

def test_run_critic_ensemble_disabled_by_default():
    """enable_ensemble=False → ensemble_critic never called."""
    primary = _make_critic_for_run(["PASS"])
    ensemble_calls = []
    ensemble_client = MagicMock()
    ensemble_client.chat.completions.create.side_effect = lambda **k: ensemble_calls.append(1)
    ensemble = Critic(client=ensemble_client, _wrap_instructor=False)

    run_critic(_QUERY, _ANSWER_GOOD, _SECTIONS, primary, generate_fn=lambda: "x")
    assert ensemble_calls == []


def test_run_critic_ensemble_abstain_blocks():
    """enable_ensemble=True, primary=PASS, ensemble=ABSTAIN → CriticResult(ABSTAIN, used_ensemble=True)."""
    primary = _make_critic_for_run(["PASS"])
    ensemble = _make_critic_for_run(["ABSTAIN"])
    result = run_critic(
        _QUERY, _ANSWER_GOOD, _SECTIONS, primary,
        generate_fn=lambda: "x",
        ensemble_critic=ensemble,
        enable_ensemble=True,
    )
    assert result.verdict == CriticVerdict.ABSTAIN
    assert result.used_ensemble is True


def test_run_critic_ensemble_pass_allows():
    """enable_ensemble=True, primary=PASS, ensemble=PASS → CriticResult(PASS, used_ensemble=True)."""
    primary = _make_critic_for_run(["PASS"])
    ensemble = _make_critic_for_run(["PASS"])
    result = run_critic(
        _QUERY, _ANSWER_GOOD, _SECTIONS, primary,
        generate_fn=lambda: "x",
        ensemble_critic=ensemble,
        enable_ensemble=True,
    )
    assert result.verdict == CriticVerdict.PASS
    assert result.used_ensemble is True


def test_run_critic_ensemble_exception_graceful_pass():
    """ensemble raises → graceful PASS, used_ensemble=False."""
    primary = _make_critic_for_run(["PASS"])
    ensemble_client = MagicMock()
    ensemble_client.chat.completions.create.side_effect = RuntimeError("timeout")
    ensemble = Critic(client=ensemble_client, _wrap_instructor=False)
    result = run_critic(
        _QUERY, _ANSWER_GOOD, _SECTIONS, primary,
        generate_fn=lambda: "x",
        ensemble_critic=ensemble,
        enable_ensemble=True,
    )
    assert result.verdict == CriticVerdict.PASS
    assert result.used_ensemble is False
