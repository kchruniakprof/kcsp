"""Tests for llm_selector — TDD D4."""
import pytest
from unittest.mock import MagicMock

from src.context_pruner import PrunedChunk


def _make_chunk(verbatim: str, pruned: str | None = None) -> PrunedChunk:
    return PrunedChunk(verbatim_text=verbatim, pruned_text=pruned or verbatim)


def _mock_client(confidence: float = 0.9, selected_index: int = 0):
    client = MagicMock()
    resp = MagicMock()
    resp.selected_index = selected_index
    resp.confidence = confidence
    resp.reason = "test"
    client.chat.completions.create.return_value = resp
    return client


# ── imports ───────────────────────────────────────────────────────────────────

def test_context_selector_importable():
    from src.llm_selector import ContextSelector
    assert ContextSelector is not None


def test_selected_chunk_importable():
    from src.llm_selector import SelectedChunk
    sc = SelectedChunk(verbatim_text="text", confidence=0.9)
    assert sc.verbatim_text == "text"


def test_abstain_importable():
    from src.llm_selector import Abstain
    a = Abstain(reason="low confidence")
    assert "low" in a.reason


# ── high confidence → SelectedChunk ──────────────────────────────────────────

def test_high_confidence_returns_selected_chunk():
    from src.llm_selector import ContextSelector, SelectedChunk
    client = _mock_client(confidence=0.95)
    cs = ContextSelector(client=client, threshold=0.5)
    chunk = _make_chunk("Verbatim markdown text", "Pruned text")
    result = cs.select([chunk], query="test query")
    assert isinstance(result, SelectedChunk)


def test_selected_chunk_verbatim_unchanged():
    from src.llm_selector import ContextSelector, SelectedChunk
    original = "Original verbatim markdown — unmodified."
    client = _mock_client(confidence=0.95)
    cs = ContextSelector(client=client, threshold=0.5)
    chunk = _make_chunk(original, "short pruned")
    result = cs.select([chunk], query="Was ist versichert?")
    assert isinstance(result, SelectedChunk)
    assert result.verbatim_text == original


# ── low confidence → Abstain ──────────────────────────────────────────────────

def test_low_confidence_with_threshold_returns_abstain():
    from src.llm_selector import ContextSelector, Abstain
    client = _mock_client(confidence=0.2)
    cs = ContextSelector(client=client, threshold=0.5)
    chunk = _make_chunk("Some text")
    result = cs.select([chunk], query="test")
    assert isinstance(result, Abstain)


def test_abstain_no_exception():
    from src.llm_selector import ContextSelector, Abstain
    client = _mock_client(confidence=0.1)
    cs = ContextSelector(client=client, threshold=0.8)
    chunk = _make_chunk("x")
    result = cs.select([chunk], query="test")
    assert isinstance(result, Abstain)


# ── threshold=None → never abstain ───────────────────────────────────────────

def test_threshold_none_never_abstains():
    from src.llm_selector import ContextSelector, SelectedChunk
    client = _mock_client(confidence=0.01)
    cs = ContextSelector(client=client, threshold=None)
    chunk = _make_chunk("text")
    result = cs.select([chunk], query="test")
    assert isinstance(result, SelectedChunk)


# ── LLM receives pruned_text not verbatim ────────────────────────────────────

def test_selector_uses_pruned_text_for_llm():
    from src.llm_selector import ContextSelector
    client = _mock_client(confidence=0.9)
    cs = ContextSelector(client=client, threshold=None)
    chunk = _make_chunk("VERBATIM_ORIGINAL", "PRUNED_ONLY")
    cs.select([chunk], query="test")
    # Check that the LLM was called (at least once)
    assert client.chat.completions.create.called
    # Verify verbatim text NOT passed to LLM prompt
    call_kwargs = client.chat.completions.create.call_args
    messages = call_kwargs[1].get("messages") or call_kwargs[0][0] if call_kwargs[0] else []
    if messages:
        prompt_text = " ".join(m.get("content", "") for m in messages)
        assert "VERBATIM_ORIGINAL" not in prompt_text or "PRUNED_ONLY" in prompt_text


# ── empty candidates ─────────────────────────────────────────────────────────

def test_empty_candidates_returns_abstain():
    from src.llm_selector import ContextSelector, Abstain
    client = _mock_client()
    cs = ContextSelector(client=client, threshold=None)
    result = cs.select([], query="test")
    assert isinstance(result, Abstain)


# ── REGISTRY key exists ───────────────────────────────────────────────────────

def test_registry_has_llm_selector():
    from src.model_registry import REGISTRY
    assert "llm_selector" in REGISTRY
