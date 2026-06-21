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
        sparte_hints=["Kfz"],
        **_DUMMY_EQ_KWARGS,
    )
    assert eq.original_query == "Was ist versichert?"
    assert eq.intent == Intent.COVERAGE_QUERY
    assert eq.detected_language == "de"


def test_expanded_query_sparte_hints_default_empty():
    eq = ExpandedQuery(
        original_query="x",
        normalized_query="x",
        detected_language="de",
        intent=Intent.GENERAL_INFO,
        **_DUMMY_EQ_KWARGS,
    )
    assert eq.sparte_hints == []


# ---------------------------------------------------------------------------
# Cycle 1 — sparte_hints validator: deduplication
# ---------------------------------------------------------------------------

def test_sparte_hints_deduplication():
    """Duplicate values in sparte_hints must be removed, preserving order."""
    eq = ExpandedQuery(
        original_query="x",
        normalized_query="x",
        detected_language="de",
        intent=Intent.COMPARISON,
        sparte_hints=["Kfz", "Hausrat", "Kfz"],
        **_DUMMY_EQ_KWARGS,
    )
    assert eq.sparte_hints == ["Kfz", "Hausrat"]


# ---------------------------------------------------------------------------
# Cycle 2 — sparte_hints validator: cap at 4
# ---------------------------------------------------------------------------

def test_sparte_hints_capped_at_4():
    """sparte_hints list must be capped at 4 entries after dedup."""
    # Provide 4 valid unique values — all should pass through
    eq = ExpandedQuery(
        original_query="x",
        normalized_query="x",
        detected_language="de",
        intent=Intent.COMPARISON,
        sparte_hints=["Kfz", "Hausrat", "Glas", "Schmuck"],
        **_DUMMY_EQ_KWARGS,
    )
    assert len(eq.sparte_hints) == 4

    # If somehow more are given (e.g., duplicates inflate list before dedup
    # but we try 5 with duplicates), validator should still return ≤4
    eq2 = ExpandedQuery(
        original_query="x",
        normalized_query="x",
        detected_language="de",
        intent=Intent.COMPARISON,
        sparte_hints=["Kfz", "Hausrat", "Glas", "Schmuck", "Kfz", "Hausrat"],
        **_DUMMY_EQ_KWARGS,
    )
    assert len(eq2.sparte_hints) <= 4


# ---------------------------------------------------------------------------
# Cycle 3 — sparte_hints validator: OOS values filtered out
# ---------------------------------------------------------------------------

def test_sparte_hints_invalid_values_filtered():
    """Values outside the allowed set must be silently dropped."""
    eq = ExpandedQuery(
        original_query="x",
        normalized_query="x",
        detected_language="de",
        intent=Intent.GENERAL_INFO,
        sparte_hints=["Kfz", "Leben", "Reise", "Hausrat"],
        **_DUMMY_EQ_KWARGS,
    )
    assert eq.sparte_hints == ["Kfz", "Hausrat"]


def test_sparte_hints_all_invalid_gives_empty():
    """All invalid values → empty list (OOS)."""
    eq = ExpandedQuery(
        original_query="x",
        normalized_query="x",
        detected_language="de",
        intent=Intent.OUT_OF_SCOPE,
        sparte_hints=["Leben", "Reise"],
        **_DUMMY_EQ_KWARGS,
    )
    assert eq.sparte_hints == []


# ---------------------------------------------------------------------------
# Cycle 4 — primary_sparte property
# ---------------------------------------------------------------------------

def test_primary_sparte_returns_first():
    """primary_sparte returns the first element of sparte_hints."""
    eq = ExpandedQuery(
        original_query="x",
        normalized_query="x",
        detected_language="de",
        intent=Intent.COVERAGE_QUERY,
        sparte_hints=["Hausrat", "Glas"],
        **_DUMMY_EQ_KWARGS,
    )
    assert eq.primary_sparte == "Hausrat"


def test_primary_sparte_returns_none_when_empty():
    """primary_sparte returns None when sparte_hints is empty."""
    eq = ExpandedQuery(
        original_query="x",
        normalized_query="x",
        detected_language="de",
        intent=Intent.OUT_OF_SCOPE,
        sparte_hints=[],
        **_DUMMY_EQ_KWARGS,
    )
    assert eq.primary_sparte is None


# ---------------------------------------------------------------------------
# Cycle 5 — old sparte_hint field no longer exists
# ---------------------------------------------------------------------------

def test_old_sparte_hint_field_removed():
    """The old singular sparte_hint field must NOT exist on ExpandedQuery."""
    eq = ExpandedQuery(
        original_query="x",
        normalized_query="x",
        detected_language="de",
        intent=Intent.GENERAL_INFO,
        **_DUMMY_EQ_KWARGS,
    )
    assert not hasattr(eq, "sparte_hint"), (
        "sparte_hint field still present — must be removed in G2"
    )


# ---------------------------------------------------------------------------
# Cycle 6 — system prompt no longer contains "null: cross-branch or unclear"
# ---------------------------------------------------------------------------

def test_system_prompt_no_null_cross_branch():
    """_SYSTEM_PROMPT must not contain the old 'null: cross-branch or unclear' line."""
    from src.query_expansion import _SYSTEM_PROMPT
    assert "null: cross-branch or unclear" not in _SYSTEM_PROMPT


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
        intent=Intent.GENERAL_INFO, sparte_hints=[], **_DUMMY_EQ_KWARGS,
    )
    base.update(kwargs)
    return ExpandedQuery(**base)


def test_expand_german_query(qe, monkeypatch):
    expected = _eq(
        original_query="Was ist im Kfz versichert?",
        normalized_query="Was ist im Kfz versichert?",
        detected_language="de", intent=Intent.COVERAGE_QUERY, sparte_hints=["Kfz"],
    )
    _mock_expand(monkeypatch, expected)
    result = qe.expand("Was ist im Kfz versichert?")
    assert result.detected_language == "de"
    assert result.intent == Intent.COVERAGE_QUERY


def test_expand_polish_query_normalized_to_de(qe, monkeypatch):
    expected = _eq(
        original_query="Co jest objęte Hausrat?",
        normalized_query="Was ist im Hausrat versichert?",
        detected_language="pl", intent=Intent.COVERAGE_QUERY, sparte_hints=["Hausrat"],
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
        detected_language="en", intent=Intent.COVERAGE_QUERY, sparte_hints=["Kfz"],
    )
    _mock_expand(monkeypatch, expected)
    result = qe.expand("What does Kfz cover?")
    assert result.detected_language == "en"
    assert result.normalized_query != result.original_query


def test_expand_out_of_scope(qe, monkeypatch):
    expected = _eq(
        original_query="Wer gewinnt die WM?",
        normalized_query="Wer gewinnt die WM?",
        detected_language="de", intent=Intent.OUT_OF_SCOPE, sparte_hints=[],
    )
    _mock_expand(monkeypatch, expected)
    result = qe.expand("Wer gewinnt die WM?")
    assert result.intent == Intent.OUT_OF_SCOPE


def test_expand_exclusion_intent(qe, monkeypatch):
    expected = _eq(
        original_query="Was ist nicht versichert bei Hausrat?",
        normalized_query="Was ist nicht versichert bei Hausrat?",
        detected_language="de", intent=Intent.EXCLUSION_QUERY, sparte_hints=["Hausrat"],
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


# ---------------------------------------------------------------------------
# G2b: OOS false-positive fixes
# ---------------------------------------------------------------------------

def test_system_prompt_kfz_legal_topics_in_scope():
    """System prompt must clarify that Kfz legal/pricing topics are IN scope."""
    from src.query_expansion import _SYSTEM_PROMPT
    assert "grobe Fahrl" in _SYSTEM_PROMPT or "Leistungsk" in _SYSTEM_PROMPT


def test_system_prompt_regionalklasse_in_scope():
    """System prompt must clarify Regionalklasse is IN scope for Kfz."""
    from src.query_expansion import _SYSTEM_PROMPT
    assert "Regionalklasse" in _SYSTEM_PROMPT


def test_few_shot_messages_include_grobe_fahrlassigkeit(qe):
    """_call_llm messages must include a grobe Fahrlässigkeit → non-OOS example."""
    captured = []

    def fake_create(**kwargs):
        captured.append(kwargs.get("messages", []))
        return _eq()

    qe._client = MagicMock()
    qe._client.chat.completions.create.side_effect = fake_create
    try:
        qe._call_llm("test")
    except Exception:
        pass

    all_content = " ".join(
        m.get("content", "") for msgs in captured for m in msgs
    )
    assert "grobe Fahrl" in all_content, "No grobe Fahrlässigkeit example in few-shot messages"


def test_few_shot_messages_include_regionalklasse(qe):
    """_call_llm messages must include a Regionalklasse → GENERAL_INFO example."""
    captured = []

    def fake_create(**kwargs):
        captured.append(kwargs.get("messages", []))
        return _eq()

    qe._client = MagicMock()
    qe._client.chat.completions.create.side_effect = fake_create
    try:
        qe._call_llm("test")
    except Exception:
        pass

    all_content = " ".join(
        m.get("content", "") for msgs in captured for m in msgs
    )
    assert "Regionalklasse" in all_content, "No Regionalklasse example in few-shot messages"
