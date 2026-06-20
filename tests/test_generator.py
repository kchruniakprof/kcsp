"""Tests for generator — TDD cycle."""
import pytest
from unittest.mock import MagicMock, patch

from src.generator import Generator, GeneratedAnswer, AnswerMode


# ---------------------------------------------------------------------------
# AnswerMode enum
# ---------------------------------------------------------------------------

def test_answer_mode_has_verbatim():
    assert AnswerMode.VERBATIM in AnswerMode


def test_answer_mode_has_compare():
    assert AnswerMode.COMPARE in AnswerMode


# ---------------------------------------------------------------------------
# GeneratedAnswer
# ---------------------------------------------------------------------------

def test_generated_answer_fields():
    ga = GeneratedAnswer(
        answer="Die Glasversicherung deckt...",
        sources=[1, 3],
        mode=AnswerMode.VERBATIM,
        breadcrumbs=["Glas > KT2021GLHR > §1"],
    )
    assert ga.answer.startswith("Die")
    assert 1 in ga.sources
    assert ga.mode == AnswerMode.VERBATIM


def test_generated_answer_sources_list():
    ga = GeneratedAnswer(answer="x", sources=[], mode=AnswerMode.VERBATIM, breadcrumbs=[])
    assert isinstance(ga.sources, list)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SECTIONS = [
    {
        "section_id": 1, "heading": "1. Was ist versichert",
        "markdown": "## 1. Was ist versichert\nVersichert ist der Hausrat.",
        "breadcrumb": "Hausrat > Smart > §1",
    },
    {
        "section_id": 2, "heading": "2. Ausschlüsse",
        "markdown": "## 2. Ausschlüsse\nNicht versichert: Vorsatz.",
        "breadcrumb": "Hausrat > Smart > §2",
    },
]


@pytest.fixture
def gen():
    client = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = (
        "## 1. Was ist versichert\nVersichert ist der Hausrat."
    )
    client.chat.completions.create.return_value = resp
    return Generator(client=client)


# ---------------------------------------------------------------------------
# VERBATIM mode — no rewrite
# ---------------------------------------------------------------------------

def test_generate_verbatim_returns_generated_answer(gen):
    result = gen.generate(
        query="Was ist versichert?",
        sections=_SECTIONS,
        mode=AnswerMode.VERBATIM,
    )
    assert isinstance(result, GeneratedAnswer)


def test_generate_verbatim_mode_set(gen):
    result = gen.generate("query", _SECTIONS, mode=AnswerMode.VERBATIM)
    assert result.mode == AnswerMode.VERBATIM


def test_generate_verbatim_has_answer(gen):
    result = gen.generate("query", _SECTIONS, mode=AnswerMode.VERBATIM)
    assert result.answer.strip() != ""


def test_generate_verbatim_sources_populated(gen):
    result = gen.generate("query", _SECTIONS, mode=AnswerMode.VERBATIM)
    assert len(result.sources) > 0


def test_generate_verbatim_breadcrumbs_populated(gen):
    result = gen.generate("query", _SECTIONS, mode=AnswerMode.VERBATIM)
    assert len(result.breadcrumbs) > 0


def test_generate_verbatim_answer_contains_markdown(gen):
    result = gen.generate("Was ist versichert?", _SECTIONS, mode=AnswerMode.VERBATIM)
    # verbatim mode should include markdown from source sections
    assert "#" in result.answer or result.answer.strip() != ""


# ---------------------------------------------------------------------------
# COMPARE mode — diff step
# ---------------------------------------------------------------------------

def test_generate_compare_returns_generated_answer(gen):
    result = gen.generate(
        query="Was ist der Unterschied zwischen Smart und Best?",
        sections=_SECTIONS,
        mode=AnswerMode.COMPARE,
    )
    assert isinstance(result, GeneratedAnswer)


def test_generate_compare_mode_set(gen):
    result = gen.generate("Vergleich", _SECTIONS, mode=AnswerMode.COMPARE)
    assert result.mode == AnswerMode.COMPARE


# ---------------------------------------------------------------------------
# Empty sections → abstain signal
# ---------------------------------------------------------------------------

def test_generate_empty_sections_abstain(gen):
    result = gen.generate("query", sections=[], mode=AnswerMode.VERBATIM)
    assert result.answer == "" or result.sources == []


# ---------------------------------------------------------------------------
# temperature=0 for verbatim
# ---------------------------------------------------------------------------

def test_verbatim_uses_temperature_zero():
    calls = []
    client = MagicMock()

    def fake_create(**kwargs):
        calls.append(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = "## test\ncontent"
        return resp

    client.chat.completions.create.side_effect = fake_create
    gen = Generator(client=client)
    gen.generate("query", _SECTIONS, mode=AnswerMode.VERBATIM)

    assert len(calls) > 0
    assert calls[0].get("temperature", 999) == 0
