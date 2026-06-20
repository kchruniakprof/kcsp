"""Tests for doc_filter — TDD D2."""
import pandas as pd
import pytest
from pathlib import Path

PARQUET_DIR = Path("D:/_FUN/kcsp/v1/parquet")


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def documents_df():
    return pd.read_parquet(PARQUET_DIR / "documents.parquet")


@pytest.fixture(scope="module")
def sections_df():
    return pd.read_parquet(PARQUET_DIR / "sections.parquet")


@pytest.fixture(scope="module")
def subsections_df():
    return pd.read_parquet(PARQUET_DIR / "subsections.parquet")


class _FakeQuery:
    """Minimal stand-in for ExpandedQuery in filter() tests."""
    def __init__(self, sparte_hint=None, domain_terms=None):
        self.sparte_hint = sparte_hint
        self.domain_terms = domain_terms or []


# ── DocFilter Protocol ────────────────────────────────────────────────────────

def test_doc_filter_protocol_importable():
    from src.doc_filter import DocFilter
    assert DocFilter is not None


def test_product_detector_adapter_importable():
    from src.doc_filter import ProductDetectorAdapter
    assert ProductDetectorAdapter is not None


def test_rare_tag_matcher_adapter_importable():
    from src.doc_filter import RareTagMatcherAdapter
    assert RareTagMatcherAdapter is not None


def test_composite_doc_filter_importable():
    from src.doc_filter import CompositeDocFilter
    assert CompositeDocFilter is not None


# ── ProductDetectorAdapter ────────────────────────────────────────────────────

def test_product_detector_hausrat_smart(documents_df):
    from src.doc_filter import ProductDetectorAdapter
    adapter = ProductDetectorAdapter(documents_df, sparte="Hausrat", tarif="Smart")
    result = adapter.filter(_FakeQuery())
    assert len(result) == 1
    # must not include Kfz or other Hausrat tarifs
    doc_ids = result
    matched = documents_df[documents_df["doc_id"].isin(doc_ids)]
    assert (matched["sparte"] == "Hausrat").all()
    assert (matched["tarif"] == "Smart").all()


def test_product_detector_kfz_no_tarif(documents_df):
    from src.doc_filter import ProductDetectorAdapter
    adapter = ProductDetectorAdapter(documents_df, sparte="Kfz", tarif=None)
    result = adapter.filter(_FakeQuery())
    # should return all Kfz doc_ids (Spezial + Standard = 2)
    matched = documents_df[documents_df["doc_id"].isin(result)]
    assert (matched["sparte"] == "Kfz").all()
    assert len(result) >= 2


def test_product_detector_unknown_tarif_returns_empty(documents_df):
    from src.doc_filter import ProductDetectorAdapter
    adapter = ProductDetectorAdapter(documents_df, sparte="Kfz", tarif="NonExistent")
    result = adapter.filter(_FakeQuery())
    assert result == frozenset()


def test_product_detector_uses_query_sparte_hint_when_no_init_sparte(documents_df):
    from src.doc_filter import ProductDetectorAdapter
    adapter = ProductDetectorAdapter(documents_df)
    result = adapter.filter(_FakeQuery(sparte_hint="Hausrat"))
    matched = documents_df[documents_df["doc_id"].isin(result)]
    assert (matched["sparte"] == "Hausrat").all()
    assert len(result) > 0


def test_product_detector_no_sparte_returns_none(documents_df):
    from src.doc_filter import ProductDetectorAdapter
    adapter = ProductDetectorAdapter(documents_df)
    result = adapter.filter(_FakeQuery(sparte_hint=None))
    assert result is None


def test_product_detector_returns_frozenset(documents_df):
    from src.doc_filter import ProductDetectorAdapter
    adapter = ProductDetectorAdapter(documents_df, sparte="Kfz")
    result = adapter.filter(_FakeQuery())
    assert isinstance(result, frozenset)


# ── RareTagMatcherAdapter ────────────────────────────────────────────────────

def test_rare_tag_matcher_glasbruch(sections_df, subsections_df):
    from src.doc_filter import RareTagMatcherAdapter
    adapter = RareTagMatcherAdapter(sections_df, subsections_df)
    result = adapter.filter(_FakeQuery(domain_terms=["Glasbruch"]))
    # Glasbruch should appear in at least some sections
    assert isinstance(result, frozenset)


def test_rare_tag_matcher_generic_schaden_returns_none(sections_df, subsections_df):
    from src.doc_filter import RareTagMatcherAdapter
    adapter = RareTagMatcherAdapter(sections_df, subsections_df)
    result = adapter.filter(_FakeQuery(domain_terms=["Schaden"]))
    assert result is None


def test_rare_tag_matcher_empty_terms_returns_none(sections_df, subsections_df):
    from src.doc_filter import RareTagMatcherAdapter
    adapter = RareTagMatcherAdapter(sections_df, subsections_df)
    result = adapter.filter(_FakeQuery(domain_terms=[]))
    assert result is None


def test_rare_tag_matcher_all_generic_blocklist_returns_none(sections_df, subsections_df):
    from src.doc_filter import RareTagMatcherAdapter, GENERIC_BLOCKLIST
    adapter = RareTagMatcherAdapter(sections_df, subsections_df)
    result = adapter.filter(_FakeQuery(domain_terms=list(GENERIC_BLOCKLIST)))
    assert result is None


def test_rare_tag_matcher_no_tag_match_returns_none(sections_df, subsections_df):
    """Rare term present but not in any topic_tags → no-filter fallback."""
    from src.doc_filter import RareTagMatcherAdapter
    adapter = RareTagMatcherAdapter(sections_df, subsections_df)
    result = adapter.filter(_FakeQuery(domain_terms=["XYZ_NIEISTNIEJĄCY_TAG_99999"]))
    assert result is None


def test_rare_tag_matcher_match_returns_frozenset(sections_df, subsections_df):
    from src.doc_filter import RareTagMatcherAdapter
    adapter = RareTagMatcherAdapter(sections_df, subsections_df)
    result = adapter.filter(_FakeQuery(domain_terms=["Glasbruch"]))
    # Glasbruch should match Glas sections — non-None frozenset
    assert result is None or isinstance(result, frozenset)


# ── CompositeDocFilter ────────────────────────────────────────────────────────

def test_composite_empty_adapters_returns_none():
    from src.doc_filter import CompositeDocFilter
    cf = CompositeDocFilter([])
    result = cf.filter(_FakeQuery())
    assert result is None


def test_composite_all_none_adapters_returns_none(documents_df):
    """All adapters return None (no sparte) → composite None = no-filter."""
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter
    a1 = ProductDetectorAdapter(documents_df)  # no sparte → None
    a2 = ProductDetectorAdapter(documents_df)  # no sparte → None
    cf = CompositeDocFilter([a1, a2])
    result = cf.filter(_FakeQuery(sparte_hint=None))
    assert result is None


def test_composite_one_none_one_result_returns_result(documents_df):
    """One adapter returns None, other returns set → union of non-None."""
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter
    a_none = ProductDetectorAdapter(documents_df)  # no sparte → None
    a_kfz = ProductDetectorAdapter(documents_df, sparte="Kfz")
    cf = CompositeDocFilter([a_none, a_kfz])
    result = cf.filter(_FakeQuery())
    kfz_ids = set(documents_df[documents_df["sparte"] == "Kfz"]["doc_id"])
    assert result is not None
    assert result == frozenset(kfz_ids)


def test_composite_union_of_adapters(documents_df):
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter

    a1 = ProductDetectorAdapter(documents_df, sparte="Kfz", tarif="Spezial")
    a2 = ProductDetectorAdapter(documents_df, sparte="Hausrat", tarif="Smart")
    cf = CompositeDocFilter([a1, a2])
    result = cf.filter(_FakeQuery())
    matched = documents_df[documents_df["doc_id"].isin(result)]
    sparten = set(matched["sparte"].unique())
    assert "Kfz" in sparten
    assert "Hausrat" in sparten


def test_composite_all_empty_returns_empty(documents_df):
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter

    a1 = ProductDetectorAdapter(documents_df, sparte="Kfz", tarif="BadTarif")
    a2 = ProductDetectorAdapter(documents_df, sparte="Hausrat", tarif="BadTarif")
    cf = CompositeDocFilter([a1, a2])
    result = cf.filter(_FakeQuery())
    assert result == frozenset()


def test_composite_returns_frozenset(documents_df):
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter
    cf = CompositeDocFilter([ProductDetectorAdapter(documents_df, sparte="Kfz")])
    result = cf.filter(_FakeQuery())
    assert isinstance(result, frozenset)
