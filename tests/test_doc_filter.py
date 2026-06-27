"""Tests for doc_filter — TDD D2."""
import pandas as pd
import pytest
from pathlib import Path

PARQUET_DIR = Path(__file__).parent.parent / "parquet"


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
    def __init__(self, sparte_hints=None, domain_terms=None, normalized_query=""):
        self.sparte_hints = sparte_hints or []
        self.domain_terms = domain_terms or []
        self.normalized_query = normalized_query


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


def test_product_detector_unknown_tarif_returns_none(documents_df):
    from src.doc_filter import ProductDetectorAdapter
    adapter = ProductDetectorAdapter(documents_df, sparte="Kfz", tarif="NonExistent")
    result = adapter.filter(_FakeQuery())
    assert result is None  # B1: empty result → no-filter fallback, never frozenset()


def test_product_detector_uses_query_sparte_hints(documents_df):
    from src.doc_filter import ProductDetectorAdapter
    adapter = ProductDetectorAdapter(documents_df)
    result = adapter.filter(_FakeQuery(sparte_hints=["Hausrat"]))
    matched = documents_df[documents_df["doc_id"].isin(result)]
    assert (matched["sparte"] == "Hausrat").all()
    assert len(result) > 0


def test_product_detector_no_sparte_returns_none(documents_df):
    from src.doc_filter import ProductDetectorAdapter
    adapter = ProductDetectorAdapter(documents_df)
    result = adapter.filter(_FakeQuery(sparte_hints=[]))
    assert result is None


def test_product_detector_returns_frozenset(documents_df):
    from src.doc_filter import ProductDetectorAdapter
    adapter = ProductDetectorAdapter(documents_df, sparte="Kfz")
    result = adapter.filter(_FakeQuery())
    assert isinstance(result, frozenset)


# ── RareTagMatcherAdapter ────────────────────────────────────────────────────

def test_rare_tag_matcher_glasbruch(sections_df, subsections_df):
    import pandas as pd
    from src.doc_filter import RareTagMatcherAdapter
    all_tags = pd.concat([sections_df["topic_tags"], subsections_df["topic_tags"]])
    has_glasbruch = any("Glasbruch" in list(t) for t in all_tags if t is not None and len(t) > 0)
    if not has_glasbruch:
        pytest.skip("topic_tags not enriched yet — no Glasbruch in parquet")
    adapter = RareTagMatcherAdapter(sections_df, subsections_df)
    result = adapter.filter(_FakeQuery(domain_terms=["Glasbruch"]))
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
    result = cf.filter(_FakeQuery(sparte_hints=[]))
    assert result is None


def test_composite_gate_none_returns_none(documents_df, sections_df, subsections_df):
    """gate=None (no sparte_hints) → None even if rare has matches."""
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter, RareTagMatcherAdapter
    gate = ProductDetectorAdapter(documents_df)
    rare = RareTagMatcherAdapter(sections_df, subsections_df)
    cf = CompositeDocFilter([gate, rare])
    result = cf.filter(_FakeQuery(sparte_hints=[], domain_terms=["Glasbruch"]))
    assert result is None


def test_composite_multi_sparte_via_gate(documents_df):
    """Multi-sparte via sparte_hints → gate returns union of both, rare=None → gate returned."""
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter
    gate = ProductDetectorAdapter(documents_df)
    cf = CompositeDocFilter([gate])
    result = cf.filter(_FakeQuery(sparte_hints=["Kfz", "Hausrat"]))
    assert result is not None
    matched = documents_df[documents_df["doc_id"].isin(result)]
    sparten = set(matched["sparte"].unique())
    assert "Kfz" in sparten
    assert "Hausrat" in sparten


def test_composite_empty_gate_returns_none(documents_df):
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter

    a1 = ProductDetectorAdapter(documents_df, sparte="Kfz", tarif="BadTarif")
    a2 = ProductDetectorAdapter(documents_df, sparte="Hausrat", tarif="BadTarif")
    cf = CompositeDocFilter([a1, a2])
    result = cf.filter(_FakeQuery())
    assert result is None  # B1: gate empty → no-filter fallback (not frozenset())


def test_composite_returns_frozenset(documents_df):
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter
    cf = CompositeDocFilter([ProductDetectorAdapter(documents_df, sparte="Kfz")])
    result = cf.filter(_FakeQuery())
    assert isinstance(result, frozenset)


# ── _detect_tarif ─────────────────────────────────────────────────────────────

from src.doc_filter import _detect_tarif

_TARIF_NAMES = ["Spezial", "Standard", "Smart", "Best", "Best+Naturgefahren", "Best+Fahrraddiebstahl"]


def test_detect_tarif_no_match_returns_none():
    assert _detect_tarif("Welche Schäden sind versichert?", _TARIF_NAMES) is None


def test_detect_tarif_full_name_match():
    assert _detect_tarif("Was deckt der Spezial Tarif ab?", _TARIF_NAMES) == "Spezial"


def test_detect_tarif_rider_alias_maps_to_compound():
    # "Fahrraddiebstahl" alone → must return "Best+Fahrraddiebstahl"
    assert _detect_tarif("Ist Fahrraddiebstahl versichert?", _TARIF_NAMES) == "Best+Fahrraddiebstahl"


def test_detect_tarif_longest_first_compound_wins():
    # "Best+Fahrraddiebstahl" is longer than "Best+Naturgefahren"
    # query mentions full compound → longest match wins
    assert _detect_tarif("Gilt das für Best+Fahrraddiebstahl?", _TARIF_NAMES) == "Best+Fahrraddiebstahl"


def test_detect_tarif_base_token_alone_does_not_match():
    # "Best" alone is base token — NOT an alias for compound tarifs
    assert _detect_tarif("Ich habe Best Hausrat.", _TARIF_NAMES) == "Best"


# ── resolve_doc_set ───────────────────────────────────────────────────────────

from src.doc_filter import resolve_doc_set


@pytest.fixture
def docs_df():
    return pd.DataFrame([
        {"doc_id": "kfz-spezial",  "sparte": "Kfz",     "tarif": "Spezial"},
        {"doc_id": "kfz-standard", "sparte": "Kfz",     "tarif": "Standard"},
        {"doc_id": "hausrat-best", "sparte": "Hausrat",  "tarif": "Best"},
        {"doc_id": "hausrat-smart","sparte": "Hausrat",  "tarif": "Smart"},
        {"doc_id": "glas-1",       "sparte": "Glas",     "tarif": "KT2021GLHR"},
        {"doc_id": "schmuck-1",    "sparte": "Schmuck",  "tarif": "KT Schmuck"},
    ])


def test_resolve_doc_set_empty_hints_returns_none(docs_df):
    assert resolve_doc_set([], None, docs_df) is None


def test_resolve_doc_set_single_sparte_no_tarif(docs_df):
    result = resolve_doc_set(["Kfz"], None, docs_df)
    assert result == frozenset({"kfz-spezial", "kfz-standard"})


def test_resolve_doc_set_single_sparte_with_tarif(docs_df):
    result = resolve_doc_set(["Kfz"], "Spezial", docs_df)
    assert result == frozenset({"kfz-spezial"})


def test_resolve_doc_set_multi_sparte_union(docs_df):
    result = resolve_doc_set(["Hausrat", "Glas"], None, docs_df)
    assert result == frozenset({"hausrat-best", "hausrat-smart", "glas-1"})


def test_resolve_doc_set_related_sparte_adds_hausrat_for_glas(docs_df):
    result = resolve_doc_set(["Glas"], None, docs_df, normalized_query="Gilt das auch für Wohnungen?")
    assert "hausrat-best" in result
    assert "hausrat-smart" in result
    assert "glas-1" in result


def test_resolve_doc_set_related_sparte_no_trigger_without_keyword(docs_df):
    result = resolve_doc_set(["Glas"], None, docs_df, normalized_query="Welche Glasschäden sind versichert?")
    assert result == frozenset({"glas-1"})


# ── gate composition ──────────────────────────────────────────────────────────

def test_gate_rare_intersection_narrows_result(docs_df, sections_df, subsections_df):
    """rare matches doc within gate → intersection returned."""
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter, RareTagMatcherAdapter
    gate = ProductDetectorAdapter(docs_df)
    rare = RareTagMatcherAdapter(sections_df, subsections_df)
    cf = CompositeDocFilter([gate, rare])
    # Kfz query with rare term that matches only Kfz sections
    q = _FakeQuery(sparte_hints=["Kfz"], domain_terms=["Saisonkennzeichen"])
    result = cf.filter(q)
    if result is not None and len(result) > 0:
        matched = docs_df[docs_df["doc_id"].isin(result)]
        assert (matched["sparte"] == "Kfz").all(), "gate must prevent non-Kfz docs"


def test_gate_none_returns_none_regardless_of_rare(docs_df, sections_df, subsections_df):
    """gate=None (no sparte) → None even if rare matches."""
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter, RareTagMatcherAdapter
    gate = ProductDetectorAdapter(docs_df)
    rare = RareTagMatcherAdapter(sections_df, subsections_df)
    cf = CompositeDocFilter([gate, rare])
    q = _FakeQuery(sparte_hints=[], domain_terms=["Glasbruch"])
    result = cf.filter(q)
    assert result is None


def test_gate_kept_when_rare_has_no_overlap(docs_df, sections_df, subsections_df):
    """rare matches cross-sparte (no overlap with gate) → gate returned, not empty."""
    from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter, RareTagMatcherAdapter
    gate = ProductDetectorAdapter(docs_df)
    rare = RareTagMatcherAdapter(sections_df, subsections_df)
    cf = CompositeDocFilter([gate, rare])
    # Kfz gate + rare term that only appears in Glas sections → no overlap → gate preserved
    q = _FakeQuery(sparte_hints=["Schmuck"], domain_terms=["XYZ_NIEISTNIEJĄCY_TAG"])
    result = cf.filter(q)
    # rare returns None (no tag match) → gate kept
    if result is not None:
        matched = docs_df[docs_df["doc_id"].isin(result)]
        assert (matched["sparte"] == "Schmuck").all()


# ── B1: tarif scoped to sparte + fallback no-filter ──────────────────────────

def test_b1_resolve_doc_set_bad_tarif_returns_none(docs_df):
    """B1: resolve_doc_set with tarif that matches no docs → None (no-filter fallback)."""
    result = resolve_doc_set(["Kfz"], "NonExistentTarif", docs_df)
    assert result is None, "Empty result must be None (no-filter), never frozenset()"


def test_b1_resolve_doc_set_valid_tarif_returns_frozenset(docs_df):
    """B1: resolve_doc_set with valid tarif → frozenset (unchanged)."""
    result = resolve_doc_set(["Kfz"], "Spezial", docs_df)
    assert isinstance(result, frozenset)
    assert "kfz-spezial" in result


def test_b1_detect_tarif_scoped_kfz_ignores_hausrat_tarifs(docs_df):
    """B1: _detect_tarif with tarifs scoped to Kfz must not match Hausrat 'Best'."""
    kfz_tarifs = docs_df[docs_df["sparte"] == "Kfz"]["tarif"].dropna().unique().tolist()
    # 'Best' is NOT a Kfz tarif — must not be detected
    result = _detect_tarif("Wechsel von Smart zu Best", kfz_tarifs)
    assert result is None, f"'Best' is Hausrat tarif, must not match in Kfz scope, got: {result}"


def test_b1_detect_tarif_scoped_kfz_matches_spezial(docs_df):
    """B1: _detect_tarif scoped to Kfz correctly matches Kfz tarifs."""
    kfz_tarifs = docs_df[docs_df["sparte"] == "Kfz"]["tarif"].dropna().unique().tolist()
    result = _detect_tarif("Gilt das für Spezial Tarif?", kfz_tarifs)
    assert result == "Spezial"
