"""Tests for src/constants — H2 centralization."""


def test_spartes_is_frozenset():
    from src.constants import SPARTES
    assert isinstance(SPARTES, frozenset)


def test_spartes_exact_values():
    from src.constants import SPARTES
    assert SPARTES == {"Kfz", "Hausrat", "Glas", "Schmuck"}


def test_section_types_no_duplicates():
    from src.constants import SECTION_TYPES
    assert len(SECTION_TYPES) == len(set(SECTION_TYPES))


def test_section_types_min_length():
    from src.constants import SECTION_TYPES
    assert len(SECTION_TYPES) >= 12


def test_section_types_full_is_superset():
    from src.constants import SECTION_TYPES, SECTION_TYPES_FULL
    assert set(SECTION_TYPES).issubset(set(SECTION_TYPES_FULL))
    assert len(SECTION_TYPES_FULL) == 16


def test_query_expansion_uses_spartes_from_constants():
    """query_expansion validator uses SPARTES — no local hardcoded set."""
    from src.query_expansion import ExpandedQuery
    from src.constants import SPARTES
    eq = ExpandedQuery(
        chain_of_thought=["test"],
        original_query="q",
        normalized_query="q",
        detected_language="de",
        intent="GENERAL_INFO",
        sparte_hints=["Kfz", "InvalidSparte"],
        paraphrases=["a", "b", "c"],
        domain_terms=["x", "y", "z"],
        confidence_score=0.9,
    )
    assert "InvalidSparte" not in eq.sparte_hints
    assert all(s in SPARTES for s in eq.sparte_hints)


def test_doc_filter_uses_spartes_from_constants(tmp_path):
    """resolve_doc_set handles only known spartes."""
    import pandas as pd
    from src.doc_filter import resolve_doc_set
    from src.constants import SPARTES
    docs_df = pd.DataFrame([
        {"doc_id": "d1", "sparte": "Kfz", "tarif": "Standard"},
        {"doc_id": "d2", "sparte": "Hausrat", "tarif": "Best"},
    ])
    result = resolve_doc_set(["Kfz"], None, docs_df)
    assert result == frozenset({"d1"})
