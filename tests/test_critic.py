"""Tests for critic — TDD cycle."""
import pytest
from unittest.mock import MagicMock

from src.critic import Critic, CriticVerdict, CriticResult


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
# CriticResult
# ---------------------------------------------------------------------------

def test_critic_result_fields():
    r = CriticResult(
        verdict=CriticVerdict.PASS,
        reason="Antwort korrekt und vollständig.",
        confidence=0.95,
    )
    assert r.verdict == CriticVerdict.PASS
    assert r.confidence == 0.95


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
_ANSWER_GOOD = "## 1. Was ist versichert\nVersichert ist der Hausrat."
_ANSWER_HALLUCINATED = "Der Hausrat ist bis 10 Millionen Euro versichert."


def _make_critic(json_response: str) -> Critic:
    client = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = json_response
    client.chat.completions.create.return_value = resp
    return Critic(client=client)


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------

def test_evaluate_returns_critic_result():
    c = _make_critic('{"verdict":"PASS","reason":"ok","confidence":0.95}')
    result = c.evaluate(_QUERY, _ANSWER_GOOD, _SECTIONS)
    assert isinstance(result, CriticResult)


def test_evaluate_pass_verdict():
    c = _make_critic('{"verdict":"PASS","reason":"ok","confidence":0.9}')
    result = c.evaluate(_QUERY, _ANSWER_GOOD, _SECTIONS)
    assert result.verdict == CriticVerdict.PASS


def test_evaluate_regen_verdict():
    c = _make_critic('{"verdict":"REGEN","reason":"Antwort zu kurz.","confidence":0.6}')
    result = c.evaluate(_QUERY, "kurz", _SECTIONS)
    assert result.verdict == CriticVerdict.REGEN


def test_evaluate_abstain_verdict():
    c = _make_critic('{"verdict":"ABSTAIN","reason":"Keine Grundlage.","confidence":0.1}')
    result = c.evaluate(_QUERY, _ANSWER_HALLUCINATED, _SECTIONS)
    assert result.verdict == CriticVerdict.ABSTAIN


def test_evaluate_confidence_in_range():
    c = _make_critic('{"verdict":"PASS","reason":"ok","confidence":0.88}')
    result = c.evaluate(_QUERY, _ANSWER_GOOD, _SECTIONS)
    assert 0.0 <= result.confidence <= 1.0


def test_evaluate_reason_nonempty():
    c = _make_critic('{"verdict":"PASS","reason":"Antwort korrekt.","confidence":0.9}')
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
        resp = MagicMock()
        resp.choices[0].message.content = (
            '{"verdict":"PASS","reason":"ok","confidence":0.9}'
        )
        return resp

    client.chat.completions.create.side_effect = fake_create
    c = Critic(client=client)
    c.evaluate(_QUERY, _ANSWER_GOOD, _SECTIONS)

    assert len(calls) > 0
    assert calls[0].get("temperature", 999) == 0


# ---------------------------------------------------------------------------
# Edge: empty sections → ABSTAIN
# ---------------------------------------------------------------------------

def test_empty_sections_abstain():
    c = _make_critic('{"verdict":"ABSTAIN","reason":"Keine Quellen.","confidence":0.0}')
    result = c.evaluate(_QUERY, "", [])
    assert result.verdict == CriticVerdict.ABSTAIN
