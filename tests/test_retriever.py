"""Tests for retriever — updated for D5 refactor (DocFilter + dual-view)."""
import pytest
from unittest.mock import MagicMock
import numpy as np

from src.retriever import Retriever, RetrievalResult


# ── RetrievalResult ───────────────────────────────────────────────────────────

def test_retrieval_result_fields():
    r = RetrievalResult(
        section_id=1,
        doc_id="doc1",
        sparte="Kfz",
        tarif="Spezial",
        heading="A Versicherte Risiken",
        markdown="## A ...",
        pruned_markdown="## A ...",
        breadcrumb="Kfz > Spezial > §A",
        score=0.87,
        section_types=["WHAT_IS_INSURED"],
        topic_tags=["Deckungsumfang"],
    )
    assert r.section_id == 1
    assert r.score == 0.87
    assert r.sparte == "Kfz"
    assert r.pruned_markdown == "## A ..."


# ── Fixtures ──────────────────────────────────────────────────────────────────

_SECTIONS = [
    {
        "section_id": 1, "doc_id": "kfz_spezial", "sparte": "Kfz", "tarif": "Spezial",
        "heading": "A Versicherte Risiken", "markdown": "Versichert ist das Fahrzeug.",
        "breadcrumb": "Kfz > Spezial > §A", "section_types": ["WHAT_IS_INSURED"],
        "topic_tags": ["Deckungsumfang"], "is_retrieval_unit": True,
    },
    {
        "section_id": 2, "doc_id": "hausrat_smart", "sparte": "Hausrat", "tarif": "Smart",
        "heading": "1. Was ist versichert", "markdown": "Versichert ist der Hausrat.",
        "breadcrumb": "Hausrat > Smart > §1", "section_types": ["WHAT_IS_INSURED"],
        "topic_tags": ["Einbruchdiebstahl"], "is_retrieval_unit": True,
    },
    {
        "section_id": 3, "doc_id": "hausrat_best", "sparte": "Hausrat", "tarif": "Best",
        "heading": "2. Ausschlüsse", "markdown": "Nicht versichert sind Vorsatzschäden.",
        "breadcrumb": "Hausrat > Best > §2", "section_types": ["EXCLUSIONS"],
        "topic_tags": ["Ausschlüsse"], "is_retrieval_unit": True,
    },
    {
        "section_id": 4, "doc_id": "kfz_standard", "sparte": "Kfz", "tarif": "Standard",
        "heading": "B Ausschlüsse", "markdown": "Nicht versichert: Vorsatz.",
        "breadcrumb": "Kfz > Standard > §B", "section_types": ["EXCLUSIONS"],
        "topic_tags": ["Ausschlüsse"], "is_retrieval_unit": True,
    },
    {
        "section_id": 5, "doc_id": "glas_kt2021", "sparte": "Glas", "tarif": "KT2021GLHR",
        "heading": "§1 Versicherte Sachen", "markdown": "Glasscheiben sind versichert.",
        "breadcrumb": "Glas > KT2021GLHR > §1", "section_types": ["WHAT_IS_INSURED"],
        "topic_tags": ["Glasbruch"], "is_retrieval_unit": True,
    },
    {
        "section_id": 6, "doc_id": "schmuck_kt", "sparte": "Schmuck", "tarif": "KT Schmuck",
        "heading": "§3 Ausschlüsse Schmuck", "markdown": "Nicht versichert: Verlust.",
        "breadcrumb": "Schmuck > KT Schmuck > §3", "section_types": ["EXCLUSIONS"],
        "topic_tags": ["Ausschlüsse"], "is_retrieval_unit": True,
    },
    # L1-parent: is_retrieval_unit=False — must be excluded from index
    {
        "section_id": 7, "doc_id": "kfz_spezial", "sparte": "Kfz", "tarif": "Spezial",
        "heading": "Parent section", "markdown": "Parent markdown (L2 children exist).",
        "breadcrumb": "Kfz > Spezial > §Parent", "section_types": ["WHAT_IS_INSURED"],
        "topic_tags": [], "is_retrieval_unit": False,
    },
]


class _AllowedDocFilter:
    """Active DocFilter — returns frozenset (may be empty = active+no-match)."""
    def __init__(self, allowed: set[str]):
        self._allowed = frozenset(allowed)
    def filter(self, query):
        return self._allowed


class _NoFilterDocFilter:
    """Returns None — signals no-filter (search all sections)."""
    def filter(self, query):
        return None


@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    def encode(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([[1.0, 0.0, 0.0, 0.0]] * len(texts))
    emb.encode.side_effect = encode
    return emb


@pytest.fixture
def retriever(mock_embedder):
    return Retriever(sections=_SECTIONS, embedder=mock_embedder)


# ── is_retrieval_unit filter ──────────────────────────────────────────────────

def test_l1_parent_excluded_from_index(retriever):
    """section_id=7 (is_retrieval_unit=False) must not appear in any result."""
    results = retriever.retrieve("test", top_k=10)
    ids = [r.section_id for r in results]
    assert 7 not in ids


def test_l1_parent_not_in_sections(mock_embedder):
    """Retriever internal sections list must not contain is_retrieval_unit=False rows."""
    r = Retriever(sections=_SECTIONS, embedder=mock_embedder)
    assert all(s.get("is_retrieval_unit", True) for s in r._sections)


# ── basic retrieval ───────────────────────────────────────────────────────────

def test_retrieve_returns_list(retriever):
    results = retriever.retrieve("Was ist versichert?", top_k=3)
    assert isinstance(results, list)


def test_retrieve_returns_retrieval_results(retriever):
    results = retriever.retrieve("Was ist versichert?", top_k=2)
    for r in results:
        assert isinstance(r, RetrievalResult)


def test_retrieve_top_k_limit(retriever):
    results = retriever.retrieve("Was ist versichert?", top_k=2)
    assert len(results) <= 2


def test_retrieve_sorted_by_score_desc(retriever):
    results = retriever.retrieve("Was ist versichert?", top_k=4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# ── dual-view: pruned_markdown ────────────────────────────────────────────────

def test_result_has_pruned_markdown(retriever):
    results = retriever.retrieve("Was ist versichert?", top_k=1)
    assert results
    assert hasattr(results[0], "pruned_markdown")


def test_verbatim_markdown_unchanged(retriever):
    """markdown field must equal original markdown from sections dict."""
    results = retriever.retrieve("Was ist versichert?", top_k=4)
    for r in results:
        original = next(s for s in _SECTIONS if s["section_id"] == r.section_id)
        assert r.markdown == original["markdown"]


# ── DocFilter integration ─────────────────────────────────────────────────────

def test_doc_filter_kfz_only(retriever):
    df = _AllowedDocFilter({"kfz_spezial", "kfz_standard"})
    results = retriever.retrieve("Fahrzeugschaden", top_k=4, doc_filter=df)
    assert all(r.sparte == "Kfz" for r in results)


def test_doc_filter_hausrat_smart_only(retriever):
    df = _AllowedDocFilter({"hausrat_smart"})
    results = retriever.retrieve("Einbruch", top_k=4, doc_filter=df)
    assert all(r.doc_id == "hausrat_smart" for r in results)


def test_none_means_no_filter(retriever):
    """doc_filter returning None = no-filter → all retrieval units returned."""
    df = _NoFilterDocFilter()
    results = retriever.retrieve("test", top_k=10, doc_filter=df)
    assert len(results) == 6  # same as without doc_filter


def test_empty_frozenset_returns_empty(retriever):
    """doc_filter returning frozenset() = active filter with no match → []."""
    df = _AllowedDocFilter(set())
    results = retriever.retrieve("test", top_k=4, doc_filter=df)
    assert results == []


def test_no_doc_filter_returns_all_units(retriever):
    """Without doc_filter, retriever searches all is_retrieval_unit=True sections."""
    results = retriever.retrieve("test", top_k=10)
    assert len(results) == 6  # 6 is_retrieval_unit=True, 1 excluded (id=7)


# ── Section type filter ───────────────────────────────────────────────────────

def test_section_type_filter(retriever):
    results = retriever.retrieve(
        "Was ist ausgeschlossen?", top_k=4,
        section_types=["EXCLUSIONS"]
    )
    for r in results:
        assert "EXCLUSIONS" in r.section_types


def test_section_type_filter_coverage(retriever):
    results = retriever.retrieve(
        "Was ist versichert?", top_k=4,
        section_types=["WHAT_IS_INSURED"]
    )
    for r in results:
        assert "WHAT_IS_INSURED" in r.section_types


# ── Empty index / no match ────────────────────────────────────────────────────

def test_doc_filter_no_match_returns_empty(retriever):
    df = _AllowedDocFilter({"nonexistent_doc"})
    results = retriever.retrieve("x", top_k=4, doc_filter=df)
    assert results == []


# ── F2: query_obj passthrough ─────────────────────────────────────────────────

class _CapturingDocFilter:
    """DocFilter that records the query_obj it received."""
    def __init__(self, return_value=None):
        self.received_query = None
        self._return = return_value

    def filter(self, query):
        self.received_query = query
        return self._return


def test_query_obj_forwarded_to_doc_filter(retriever):
    """retrieve_multi passes query_obj to doc_filter.filter()."""
    cap = _CapturingDocFilter(return_value=None)
    _FQ = type("FQ", (), {"sparte_hint": "Kfz", "domain_terms": ["Test"]})()
    retriever.retrieve("test", doc_filter=cap, query_obj=_FQ)
    assert cap.received_query is _FQ


def test_query_obj_none_uses_fallback(retriever):
    """Without query_obj, filter still receives some object (backward compat)."""
    cap = _CapturingDocFilter(return_value=None)
    retriever.retrieve("test", doc_filter=cap)
    assert cap.received_query is not None


def test_cross_branch_no_sparte_returns_all(retriever):
    """query_obj with sparte_hint=None and domain_terms=[] → CompositeDocFilter → None → all sections."""
    # Simulate CompositeDocFilter returning None for null-sparte cross-branch query
    cap = _CapturingDocFilter(return_value=None)
    _FQ = type("FQ", (), {"sparte_hint": None, "domain_terms": []})()
    results = retriever.retrieve("Vergleich Glas Hausrat", top_k=10, doc_filter=cap, query_obj=_FQ)
    assert len(results) == 6  # all retrieval units (no filter)
