"""Tests for promptfoo_provider — TDD cycle."""
import pytest
from unittest.mock import MagicMock, patch

from src.ragassistant import FinalAnswer
from src.query_expansion import Intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_final(abstained=False, answer="## 1. Deckungsumfang\nVersichert ist Hausrat."):
    return FinalAnswer(
        answer=answer,
        sources=[1],
        breadcrumbs=["Hausrat > Smart > §1"],
        intent=Intent.COVERAGE_QUERY,
        abstained=abstained,
        cross_sell=None,
    )


@pytest.fixture
def mock_rag(monkeypatch):
    rag = MagicMock()
    rag.ask.return_value = _make_final()
    monkeypatch.setattr("src.promptfoo_provider._get_rag", lambda: rag)
    return rag


# ---------------------------------------------------------------------------
# call_api interface
# ---------------------------------------------------------------------------

def test_call_api_returns_dict(mock_rag):
    from src.promptfoo_provider import call_api
    result = call_api("Was ist versichert?", {}, {})
    assert isinstance(result, dict)


def test_call_api_output_key_present(mock_rag):
    from src.promptfoo_provider import call_api
    result = call_api("Was ist versichert?", {}, {})
    assert "output" in result


def test_call_api_output_nonempty_on_pass(mock_rag):
    from src.promptfoo_provider import call_api
    result = call_api("Was ist versichert?", {}, {})
    assert result["output"].strip() != ""


def test_call_api_passes_prompt_to_rag(mock_rag):
    from src.promptfoo_provider import call_api
    call_api("Was ist ausgeschlossen?", {}, {})
    mock_rag.ask.assert_called_once_with("Was ist ausgeschlossen?")


def test_call_api_abstained_output_marker(monkeypatch):
    """Abstained answer should be flagged in output (contains marker or empty)."""
    rag = MagicMock()
    rag.ask.return_value = _make_final(abstained=True, answer="Ich kann keine Antwort geben.")
    monkeypatch.setattr("src.promptfoo_provider._get_rag", lambda: rag)

    from src.promptfoo_provider import call_api
    result = call_api("Lebensversicherung?", {}, {})
    assert "output" in result
    # abstained answers propagate as-is; test they're not silently dropped
    assert result["output"] != ""


def test_call_api_metadata_includes_sources(mock_rag):
    from src.promptfoo_provider import call_api
    result = call_api("Was?", {}, {})
    # optional metadata field
    if "metadata" in result:
        assert "sources" in result["metadata"] or "abstained" in result["metadata"]


def test_call_api_metadata_abstained_flag(monkeypatch):
    rag = MagicMock()
    rag.ask.return_value = _make_final(abstained=True, answer="Keine Antwort.")
    monkeypatch.setattr("src.promptfoo_provider._get_rag", lambda: rag)

    from src.promptfoo_provider import call_api
    result = call_api("Out of scope", {}, {})
    if "metadata" in result:
        assert result["metadata"].get("abstained") is True


def test_call_api_error_handling(monkeypatch):
    """Exception in RAG pipeline → returns error key, no crash."""
    rag = MagicMock()
    rag.ask.side_effect = RuntimeError("Groq timeout")
    monkeypatch.setattr("src.promptfoo_provider._get_rag", lambda: rag)

    from src.promptfoo_provider import call_api
    result = call_api("Frage", {}, {})
    assert "error" in result or "output" in result  # graceful


# ---------------------------------------------------------------------------
# CLI entrypoint (stdout JSON)
# ---------------------------------------------------------------------------

def test_main_prints_json(monkeypatch, capsys):
    rag = MagicMock()
    rag.ask.return_value = _make_final()
    monkeypatch.setattr("src.promptfoo_provider._get_rag", lambda: rag)
    monkeypatch.setattr("sys.argv", ["provider.py", "Was ist versichert?"])

    import json
    from src.promptfoo_provider import main
    main()
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "output" in data
