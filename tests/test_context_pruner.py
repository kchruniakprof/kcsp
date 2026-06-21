"""Tests for context_pruner — TDD D3."""
import pytest

_SHORT = "Kurzer Text unter zweitausend fünfhundert Zeichen."
_LONG = ("Dies ist ein langer Satz über Kfz-Versicherung. " * 120).strip()  # ~6000 chars
_EMPTY_GUARD_CONTENT = "A" * 3000  # >2500 but no sentence breaks → pruned might be empty


# ── ContextPruner ─────────────────────────────────────────────────────────────

def test_pruned_chunk_importable():
    from src.context_pruner import PrunedChunk
    pc = PrunedChunk(verbatim_text="a", pruned_text="b")
    assert pc.verbatim_text == "a"
    assert pc.pruned_text == "b"


def test_context_pruner_importable():
    from src.context_pruner import ContextPruner
    assert ContextPruner is not None


def test_short_chunk_bypass():
    """Chunk < 2500 chars → pruned_text == verbatim_text."""
    from src.context_pruner import ContextPruner
    result = ContextPruner().prune(_SHORT)
    assert result.pruned_text == result.verbatim_text


def test_verbatim_never_modified_on_long_chunk():
    """verbatim_text must equal original markdown, always."""
    from src.context_pruner import ContextPruner
    result = ContextPruner().prune(_LONG)
    assert result.verbatim_text == _LONG


def test_long_chunk_pruned_shorter():
    """Pruner should shorten pruned_text for long input."""
    from src.context_pruner import ContextPruner
    result = ContextPruner().prune(_LONG)
    assert len(result.pruned_text) < len(_LONG)


def test_empty_guard_fallback():
    """If pruning yields empty → pruned_text == verbatim_text."""
    from src.context_pruner import ContextPruner
    result = ContextPruner(max_chars=0).prune(_LONG)
    assert result.pruned_text == result.verbatim_text


def test_verbatim_is_unchanged_after_empty_guard():
    from src.context_pruner import ContextPruner
    result = ContextPruner(max_chars=0).prune(_LONG)
    assert result.verbatim_text == _LONG


def test_context_pruner_returns_pruned_chunk():
    from src.context_pruner import ContextPruner, PrunedChunk
    result = ContextPruner().prune(_LONG)
    assert isinstance(result, PrunedChunk)

