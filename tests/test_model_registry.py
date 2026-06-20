"""Tests for model_registry — TDD D1."""
import pytest


_REQUIRED_KEYS = {
    "query_expansion",
    "generator_verbatim",
    "generator_compare",
    "critic",
    "enrichment",
}


def test_registry_has_required_keys():
    from src.model_registry import REGISTRY
    missing = _REQUIRED_KEYS - set(REGISTRY)
    assert not missing, f"REGISTRY missing keys: {missing}"


def test_registry_values_are_non_empty_strings():
    from src.model_registry import REGISTRY
    for key, val in REGISTRY.items():
        assert isinstance(val, str) and val, f"REGISTRY[{key!r}] must be non-empty string"


def test_query_expansion_default_model_from_registry(monkeypatch):
    """QueryExpansion default model must match REGISTRY['query_expansion']."""
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    from src.model_registry import REGISTRY
    from src.query_expansion import QueryExpansion
    qe = QueryExpansion()
    assert qe._model == REGISTRY["query_expansion"]


def test_enrich_section_default_model_from_registry():
    """enrich_section default model must match REGISTRY['enrichment']."""
    import inspect
    from src import enrichment
    from src.model_registry import REGISTRY
    sig = inspect.signature(enrichment.enrich_section)
    default = sig.parameters["model"].default
    assert default == REGISTRY["enrichment"], (
        f"enrich_section default model {default!r} != REGISTRY['enrichment'] {REGISTRY['enrichment']!r}"
    )
