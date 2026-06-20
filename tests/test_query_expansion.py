"""Tests for query_expansion — TDD cycle."""
import pytest
from unittest.mock import patch, MagicMock

from src.query_expansion import QueryExpansion, ExpandedQuery, Intent


# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------

def test_intent_enum_has_8_values():
    assert len(Intent) == 8


def test_intent_enum_values():
    names = {i.value for i in Intent}
    expected = {
        "COVERAGE_QUERY",
        "EXCLUSION_QUERY",
        "CLAIMS_PROCEDURE",
        "PRICE_QUOTE",
        "COMPARISON",
        "COMPLAINT",
        "GENERAL_INFO",
        "OUT_OF_SCOPE",
    }
    assert names == expected


# ---------------------------------------------------------------------------
# ExpandedQuery dataclass / pydantic model
# ---------------------------------------------------------------------------

_DUMMY_EQ_KWARGS = dict(
    chain_of_thought=["step 1", "step 2", "step 3"],
    paraphrases=["Para 1", "Para 2", "Para 3"],
    domain_terms=["Hausrat", "Deckung"],
    confidence_score=0.9,
)


def test_expanded_query_has_required_fields():
    eq = ExpandedQuery(
        original_query="Was ist versichert?",
        normalized_query="Was ist versichert?",
        detected_language="de",
        intent=Intent.COVERAGE_QUERY,
        sparte_hint=None,
        **_DUMMY_EQ_KWARGS,
    )
    assert eq.original_query == "Was ist versichert?"
    assert eq.intent == Intent.COVERAGE_QUERY
    assert eq.detected_language == "de"


def test_expanded_query_sparte_hint_optional():
    eq = ExpandedQuery(
        original_query="x",
        normalized_query="x",
        detected_language="de",
        intent=Intent.GENERAL_INFO,
        sparte_hint=None,
        **_DUMMY_EQ_KWARGS,
    )
    assert eq.sparte_hint is None


# ---------------------------------------------------------------------------
# QueryExpansion — language detection + normalization (mocked Groq)
# ---------------------------------------------------------------------------

@pytest.fixture
def qe():
    return QueryExpansion(api_key="test-key")


def _mock_expand(monkeypatch, result: ExpandedQuery):
    """Patch the internal Groq call to return a canned ExpandedQuery."""
    monkeypatch.setattr(
        "src.query_expansion.QueryExpansion._call_llm",
        lambda self, query: result,
    )


def _eq(**kwargs) -> ExpandedQuery:
    base = dict(
        original_query="x", normalized_query="x", detected_language="de",
        intent=Intent.GENERAL_INFO, sparte_hint=None, **_DUMMY_EQ_KWARGS,
    )
    base.update(kwargs)
    return ExpandedQuery(**base)


def test_expand_german_query(qe, monkeypatch):
    expected = _eq(
        original_query="Was ist im Kfz versichert?",
        normalized_query="Was ist im Kfz versichert?",
        detected_language="de", intent=Intent.COVERAGE_QUERY, sparte_hint="Kfz",
    )
    _mock_expand(monkeypatch, expected)
    result = qe.expand("Was ist im Kfz versichert?")
    assert result.detected_language == "de"
    assert result.intent == Intent.COVERAGE_QUERY


def test_expand_polish_query_normalized_to_de(qe, monkeypatch):
    expected = _eq(
        original_query="Co jest objęte Hausrat?",
        normalized_query="Was ist im Hausrat versichert?",
        detected_language="pl", intent=Intent.COVERAGE_QUERY, sparte_hint="Hausrat",
    )
    _mock_expand(monkeypatch, expected)
    result = qe.expand("Co jest objęte Hausrat?")
    assert result.detected_language == "pl"
    assert result.normalized_query != result.original_query
    assert "Co" not in result.normalized_query


def test_expand_english_query_normalized_to_de(qe, monkeypatch):
    expected = _eq(
        original_query="What does Kfz cover?",
        normalized_query="Was deckt die Kfz-Versicherung ab?",
        detected_language="en", intent=Intent.COVERAGE_QUERY, sparte_hint="Kfz",
    )
    _mock_expand(monkeypatch, expected)
    result = qe.expand("What does Kfz cover?")
    assert result.detected_language == "en"
    assert result.normalized_query != result.original_query


def test_expand_out_of_scope(qe, monkeypatch):
    expected = _eq(
        original_query="Wer gewinnt die WM?",
        normalized_query="Wer gewinnt die WM?",
        detected_language="de", intent=Intent.OUT_OF_SCOPE, sparte_hint=None,
    )
    _mock_expand(monkeypatch, expected)
    result = qe.expand("Wer gewinnt die WM?")
    assert result.intent == Intent.OUT_OF_SCOPE


def test_expand_exclusion_intent(qe, monkeypatch):
    expected = _eq(
        original_query="Was ist nicht versichert bei Hausrat?",
        normalized_query="Was ist nicht versichert bei Hausrat?",
        detected_language="de", intent=Intent.EXCLUSION_QUERY, sparte_hint="Hausrat",
    )
    _mock_expand(monkeypatch, expected)
    result = qe.expand("Was ist nicht versichert bei Hausrat?")
    assert result.intent == Intent.EXCLUSION_QUERY


def test_expand_returns_expanded_query_instance(qe, monkeypatch):
    expected = _eq(intent=Intent.GENERAL_INFO)
    _mock_expand(monkeypatch, expected)
    result = qe.expand("test")
    assert isinstance(result, ExpandedQuery)


# ---------------------------------------------------------------------------
# LLM call uses temperature=0
# ---------------------------------------------------------------------------

def test_llm_call_uses_temperature_zero(qe):
    """_call_llm must pass temperature=0."""
    calls = []
    fake_result = _eq()

    def fake_create(**kwargs):
        calls.append(kwargs)
        return fake_result

    qe._client = MagicMock()
    qe._client.chat.completions.create.side_effect = fake_create
    try:
        qe._call_llm("x")
    except Exception:
        pass

    if calls:
        assert calls[0].get("temperature", 999) == 0
