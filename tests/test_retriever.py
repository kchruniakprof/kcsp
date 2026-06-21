"""Tests for retriever — updated for D5 refactor (DocFilter + dual-view)."""
import pytest
from unittest.mock import MagicMock
import numpy as np

from src.retriever import Retriever, RetrievalResult, CrossEncoderReranker
from src.model_registry import REGISTRY


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


def _make_onehot_encoder(dim: int = 20):
    """Returns an encode fn assigning unique one-hot vectors per unique text.

    Ensures pairwise cosine = 0 between distinct texts → D2 dedup never fires.
    """
    _cache: dict[str, np.ndarray] = {}
    _counter = [0]

    def encode(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for t in texts:
            if t not in _cache:
                vec = np.zeros(dim, dtype=np.float32)
                vec[_counter[0] % dim] = 1.0
                _cache[t] = vec
                _counter[0] += 1
            result.append(_cache[t])
        return np.array(result, dtype=np.float32)

    return encode


@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.encode.side_effect = _make_onehot_encoder(dim=20)
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
    # A1: top_k=3 (exactly 3 EXCLUSIONS sections exist) — boosted chunks rank first
    results = retriever.retrieve(
        "Was ist ausgeschlossen?", top_k=3,
        section_types=["EXCLUSIONS"]
    )
    for r in results:
        assert "EXCLUSIONS" in r.section_types


def test_section_type_filter_coverage(retriever):
    # A1: top_k=3 (exactly 3 WHAT_IS_INSURED sections exist) — boosted chunks rank first
    results = retriever.retrieve(
        "Was ist versichert?", top_k=3,
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


# ── G5: CrossEncoderReranker ──────────────────────────────────────────────────

# Cycle 1 — model_registry has "reranker" key
def test_registry_has_reranker_key():
    assert "reranker" in REGISTRY
    assert REGISTRY["reranker"]  # non-empty string


# Helper: build minimal RetrievalResult for reranker tests
def _make_result(section_id: int) -> RetrievalResult:
    return RetrievalResult(
        section_id=section_id,
        doc_id=f"doc{section_id}",
        sparte="Kfz",
        tarif="Spezial",
        heading=f"Heading {section_id}",
        markdown=f"Markdown {section_id}",
        pruned_markdown=f"Markdown {section_id}",
        breadcrumb=f"Kfz > §{section_id}",
        score=0.5,
        section_types=["WHAT_IS_INSURED"],
        topic_tags=[],
    )


# Cycle 2 — rerank sorts results by cross-encoder score descending
def test_reranker_sorts_by_score():
    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.1, 0.5]
    reranker._model = mock_model  # inject without lazy load
    results = [_make_result(1), _make_result(2), _make_result(3)]
    ranked = reranker.rerank("query", results)
    assert [r.section_id for r in ranked] == [1, 3, 2]


# Cycle 3 — lazy load: model NOT loaded at __init__
def test_reranker_lazy_load(monkeypatch):
    """CrossEncoderReranker() must not import or load the model at init time."""
    loaded = []

    class FakeCrossEncoder:
        def __init__(self, name):
            loaded.append(name)
        def predict(self, pairs):
            return [0.5] * len(pairs)

    import sys
    fake_st = MagicMock()
    fake_st.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    reranker = CrossEncoderReranker()
    assert loaded == [], "model must not be loaded at __init__"

    results = [_make_result(1)]
    reranker.rerank("q", results)
    assert len(loaded) == 1, "model must be loaded on first rerank()"


# Cycle 4 — Retriever with reranker=None (default) behaves identically (backward-compat)
def test_retriever_without_reranker_backward_compat(mock_embedder):
    """Retriever(reranker=None) must work exactly as before — no behavior change."""
    r = Retriever(sections=_SECTIONS, embedder=mock_embedder)  # no reranker
    results = r.retrieve("Was ist versichert?", top_k=3)
    assert len(results) <= 3
    assert all(isinstance(res, RetrievalResult) for res in results)


# Cycle 5 — Retriever with reranker: pool_k=6, top_k=3 → reranker called with 6, returns 3
def test_retriever_with_reranker_calls_rerank(mock_embedder):
    mock_reranker = MagicMock()
    # reranker.rerank returns exactly top_k results (first 3 of the 6 passed)
    mock_reranker.rerank.side_effect = lambda q, results: results[:3]

    r = Retriever(sections=_SECTIONS, embedder=mock_embedder, reranker=mock_reranker)
    results = r.retrieve("Was ist versichert?", top_k=3, pool_k=6)

    assert mock_reranker.rerank.called
    call_args = mock_reranker.rerank.call_args
    candidates_passed = call_args[0][1]  # positional arg: results list
    assert len(candidates_passed) == 6
    assert len(results) == 3


# Cycle 6 — A2: pool_k policy — pool ≤50 → reranker gets ALL candidates
def test_a2_pool_lte50_reranker_gets_full_pool(mock_embedder):
    """A2: When filtered pool ≤50 sections, reranker receives all of them (not truncated to 20)."""
    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda q, results: results[:3]

    # _SECTIONS has 6 retrieval units (≤50) — reranker must receive all 6
    r = Retriever(sections=_SECTIONS, embedder=mock_embedder, reranker=mock_reranker)
    results = r.retrieve("Was ist versichert?", top_k=3)

    assert mock_reranker.rerank.called
    candidates_passed = mock_reranker.rerank.call_args[0][1]
    assert len(candidates_passed) == 6, (
        f"Pool ≤50: reranker should receive all 6 candidates, got {len(candidates_passed)}"
    )
    assert len(results) == 3


# Cycle 7 — A2: pool_k policy — pool >50 → reranker gets top-30
def test_a2_pool_gt50_reranker_gets_top30():
    """A2: When filtered pool >50 sections, reranker receives only top-30 by dense score."""
    import numpy as np

    # Build 55 sections (>50)
    sections_large = [
        {
            "section_id": 1000 + i,
            "doc_id": f"doc_{i}",
            "sparte": "Kfz",
            "tarif": "X",
            "heading": f"Section {i}",
            "markdown": f"Content {i}.",
            "breadcrumb": f"Kfz > §{i}",
            "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": [],
            "is_retrieval_unit": True,
        }
        for i in range(55)
    ]

    emb = MagicMock()
    # Use orthogonal one-hot embeddings → cosine=0 between sections → no dedup
    emb.encode.side_effect = _make_onehot_encoder(dim=60)

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda q, results: results[:5]

    r = Retriever(sections=sections_large, embedder=emb, reranker=mock_reranker)
    results = r.retrieve("test", top_k=5)

    assert mock_reranker.rerank.called
    candidates_passed = mock_reranker.rerank.call_args[0][1]
    assert len(candidates_passed) == 30, (
        f"Pool >50: reranker should receive 30 candidates, got {len(candidates_passed)}"
    )
    assert len(results) == 5


# Cycle 8 — A2: pool exactly 50 → treated as ≤50 (full pool)
def test_a2_pool_exactly50_gets_full_pool():
    """A2: Pool of exactly 50 sections → reranker receives all 50."""
    import numpy as np

    sections_50 = [
        {
            "section_id": 2000 + i,
            "doc_id": f"doc_{i}",
            "sparte": "Kfz",
            "tarif": "X",
            "heading": f"Section {i}",
            "markdown": f"Content {i}.",
            "breadcrumb": f"Kfz > §{i}",
            "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": [],
            "is_retrieval_unit": True,
        }
        for i in range(50)
    ]

    emb = MagicMock()
    emb.encode.side_effect = _make_onehot_encoder(dim=60)

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda q, results: results[:5]

    r = Retriever(sections=sections_50, embedder=emb, reranker=mock_reranker)
    results = r.retrieve("test", top_k=5)

    assert mock_reranker.rerank.called
    candidates_passed = mock_reranker.rerank.call_args[0][1]
    assert len(candidates_passed) == 50, (
        f"Pool=50: reranker should receive all 50, got {len(candidates_passed)}"
    )


# Cycle 9 — A2: reranker=None (no reranker) still works — backward compat
def test_a2_no_reranker_still_returns_top_k(mock_embedder):
    """A2: Without reranker, retrieve works normally — returns top_k without reranking."""
    r = Retriever(sections=_SECTIONS, embedder=mock_embedder)  # no reranker
    results = r.retrieve("Was ist versichert?", top_k=3)
    assert len(results) <= 3
    assert all(isinstance(res, RetrievalResult) for res in results)


# ── A1: soft section_type boost (no hard-drop) ───────────────────────────────

def test_a1_dense_top_chunk_not_dropped_despite_wrong_type():
    """A1: chunk with SPECIAL_PROVISIONS is NOT dropped when section_types=['WHAT_IS_INSURED'].

    Uses 4 sections: 1 SPECIAL_PROVISIONS (dense #1) + 3 WHAT_IS_INSURED.
    With old hard-drop: typed=[102,103,104] len=3 >= _MIN_SECTION_CHUNKS → 101 dropped.
    With soft-boost: 101 stays in pool, gets no boost, still ranked by dense score → appears.
    """
    import numpy as np

    sections_a1 = [
        {
            "section_id": 101, "doc_id": "docA", "sparte": "Kfz", "tarif": "X",
            "heading": "Sonderbedingungen", "markdown": "Naturgefahren content.",
            "breadcrumb": "A", "section_types": ["SPECIAL_PROVISIONS"],
            "topic_tags": [], "is_retrieval_unit": True,
        },
        {
            "section_id": 102, "doc_id": "docA", "sparte": "Kfz", "tarif": "X",
            "heading": "Was versichert A", "markdown": "Deckung A.",
            "breadcrumb": "B", "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": [], "is_retrieval_unit": True,
        },
        {
            "section_id": 103, "doc_id": "docA", "sparte": "Kfz", "tarif": "X",
            "heading": "Was versichert B", "markdown": "Deckung B.",
            "breadcrumb": "C", "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": [], "is_retrieval_unit": True,
        },
        {
            "section_id": 104, "doc_id": "docA", "sparte": "Kfz", "tarif": "X",
            "heading": "Was versichert C", "markdown": "Deckung C.",
            "breadcrumb": "D", "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": [], "is_retrieval_unit": True,
        },
    ]
    # Embedder: section 101 scores 1.0 (dense #1), rest score 0.3
    emb = MagicMock()
    def encode(texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else list(texts)
        result = []
        for t in texts:
            if "Sonderbedingungen" in t or "Naturgefahren" in t:
                result.append([1.0, 0.0])
            else:
                result.append([0.3, 0.0])
        return np.array(result, dtype=np.float32)
    emb.encode.side_effect = encode

    r = Retriever(sections=sections_a1, embedder=emb)
    # top_k=4: with hard-drop, only 102/103/104 returned; with soft-boost, 101 also included
    results = r.retrieve("Naturgefahren versichert?", top_k=4, section_types=["WHAT_IS_INSURED"])
    ids = [res.section_id for res in results]
    assert 101 in ids, "Dense #1 (SPECIAL_PROVISIONS) must NOT be hard-dropped — soft boost only"


def test_a1_soft_boost_applied_once_not_stacked():
    """A1: chunk matching 2 types gets boost exactly once (+0.04, not +0.08)."""
    import numpy as np

    sections_boost = [
        {
            "section_id": 201, "doc_id": "doc", "sparte": "Kfz", "tarif": "X",
            "heading": "Multi", "markdown": "Content.",
            "breadcrumb": "a", "section_types": ["WHAT_IS_INSURED", "COVERAGE_AMOUNT"],
            "topic_tags": [], "is_retrieval_unit": True,
        },
    ]
    emb = MagicMock()
    emb.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)

    r = Retriever(sections=sections_boost, embedder=emb)
    results = r.retrieve("test", top_k=1, section_types=["WHAT_IS_INSURED", "COVERAGE_AMOUNT"])
    assert len(results) == 1
    assert abs(results[0].score - 1.04) < 0.001, f"Expected ~1.04 (1.0 + 0.04), got {results[0].score}"


def test_a1_no_section_types_no_boost():
    """A1: without section_types, scores are raw cosine (no boost applied)."""
    import numpy as np

    sections_single = [
        {
            "section_id": 301, "doc_id": "doc", "sparte": "Kfz", "tarif": "X",
            "heading": "Test", "markdown": "Content.",
            "breadcrumb": "a", "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": [], "is_retrieval_unit": True,
        },
    ]
    emb = MagicMock()
    emb.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)

    r = Retriever(sections=sections_single, embedder=emb)
    results = r.retrieve("test", top_k=1)  # no section_types
    assert len(results) == 1
    assert abs(results[0].score - 1.0) < 0.001, f"No boost expected, got {results[0].score}"


# ── D1: exact-term force include ──────────────────────────────────────────────

def test_d1_forced_include_in_reranker_pool():
    """D1: domain_term in markdown forces chunk into reranker pool even below pool_k threshold.

    52 sections (>50 → pool_k_effective=30). Section 400 has score=0.0 (rank 52),
    but domain_term='Naturgefahren' appears in its markdown → must be added to candidates.
    """
    import numpy as np

    sections_d1 = [
        {
            "section_id": 400, "doc_id": "kfz", "sparte": "Kfz", "tarif": "X",
            "heading": "Sonderbedingungen", "markdown": "Naturgefahren sind versichert.",
            "breadcrumb": "A", "section_types": ["SPECIAL_PROVISIONS"],
            "topic_tags": ["Naturgefahren"], "is_retrieval_unit": True,
        },
    ] + [
        {
            "section_id": 1000 + i, "doc_id": f"doc_{i}", "sparte": "Kfz", "tarif": "X",
            "heading": f"Section {i}", "markdown": f"Content {i}.",
            "breadcrumb": f"K > §{i}", "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": [], "is_retrieval_unit": True,
        }
        for i in range(51)
    ]

    emb = MagicMock()
    def encode_d1(texts, **kw):
        texts = [texts] if isinstance(texts, str) else list(texts)
        result = []
        for t in texts:
            # Section 400 heading contains "Sonderbedingungen" → mark with zero embedding
            if "sonderbedingungen" in t.lower():
                result.append([0.0, 0.0])
            else:
                result.append([1.0, 0.0])
        return np.array(result, dtype=np.float32)
    emb.encode.side_effect = encode_d1

    mock_reranker = MagicMock()
    captured = []
    def capture(q, results):
        captured.extend(results)
        return results[:5]
    mock_reranker.rerank.side_effect = capture

    r = Retriever(sections=sections_d1, embedder=emb, reranker=mock_reranker)

    # Query does NOT contain "Naturgefahren" text but domain_terms does
    _FQ = type("FQ", (), {
        "domain_terms": ["Naturgefahren"],
        "sparte_hints": ["Kfz"],
        "normalized_query": "Was ist durch Sturm abgedeckt?",
    })()

    r.retrieve("Was ist durch Sturm abgedeckt?", top_k=5, query_obj=_FQ)

    assert mock_reranker.rerank.called
    pool_ids = [res.section_id for res in captured]
    assert 400 in pool_ids, (
        f"Section 400 (Naturgefahren) must be forced into reranker pool despite score=0, "
        f"pool was: {pool_ids[:10]}..."
    )


def test_d1_generic_blocklist_not_forced():
    """D1: term in GENERIC_BLOCKLIST must NOT trigger force-include."""
    import numpy as np

    sections_bl = [
        {
            "section_id": 500, "doc_id": "doc", "sparte": "Kfz", "tarif": "X",
            "heading": "Versicherung",
            "markdown": "Versicherung und Versicherungsnehmer und Schaden.",
            "breadcrumb": "A", "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": [], "is_retrieval_unit": True,
        },
    ] + [
        {
            "section_id": 600 + i, "doc_id": f"d{i}", "sparte": "Kfz", "tarif": "X",
            "heading": f"S{i}", "markdown": f"Neutral content {i}.",
            "breadcrumb": f"B{i}", "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": [], "is_retrieval_unit": True,
        }
        for i in range(51)  # total 52 → pool_k_effective=30
    ]

    emb = MagicMock()
    # Use orthogonal one-hot embeddings → cosine=0 between sections → no dedup
    emb.encode.side_effect = _make_onehot_encoder(dim=60)

    mock_reranker = MagicMock()
    captured = []
    def capture(q, results):
        captured.extend(results)
        return results[:5]
    mock_reranker.rerank.side_effect = capture

    r = Retriever(sections=sections_bl, embedder=emb, reranker=mock_reranker)

    _FQ = type("FQ", (), {
        "domain_terms": ["Versicherung", "Schaden"],  # both in GENERIC_BLOCKLIST
        "sparte_hints": [],
        "normalized_query": "Versicherung",
    })()
    r.retrieve("Versicherung", top_k=5, query_obj=_FQ)

    # Pool = top-30 (52 sections, pool_k=30). Section 500 has same score as others.
    # Blocklist terms must not add section 500 as a FORCED include beyond normal pool_k.
    # If 500 is in pool, it's only because it was in the top-30 by dense score, not forced.
    assert mock_reranker.rerank.called
    # The key check: no extra candidates beyond pool_k=30 (forced would add >30)
    assert len(captured) == 30, (
        f"Blocklist terms must not force-include beyond pool_k=30, got {len(captured)}"
    )


def test_d1_forced_not_bypass_doc_filter():
    """D1: forced-include only within DocFilter gate — Hausrat section not in results for Kfz query."""
    import numpy as np

    sections_gate = [
        {
            "section_id": 700, "doc_id": "kfz", "sparte": "Kfz", "tarif": "X",
            "heading": "Naturgefahren Kfz", "markdown": "Naturgefahren für Kfz versichert.",
            "breadcrumb": "A", "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": ["Naturgefahren"], "is_retrieval_unit": True,
        },
        {
            "section_id": 701, "doc_id": "hausrat", "sparte": "Hausrat", "tarif": "Y",
            "heading": "Naturgefahren Hausrat", "markdown": "Naturgefahren im Haushalt.",
            "breadcrumb": "B", "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": ["Naturgefahren"], "is_retrieval_unit": True,
        },
    ]

    emb = MagicMock()
    emb.encode.side_effect = lambda texts, **kw: np.array(
        [[1.0, 0.0]] * (len(texts) if not isinstance(texts, str) else 1)
    )

    r = Retriever(sections=sections_gate, embedder=emb)  # no reranker

    kfz_filter = _AllowedDocFilter({"kfz"})  # DocFilter: Kfz only
    _FQ = type("FQ", (), {
        "domain_terms": ["Naturgefahren"],
        "sparte_hints": ["Kfz"],
        "normalized_query": "Naturgefahren Kfz",
    })()

    results = r.retrieve("Naturgefahren Kfz", top_k=5, doc_filter=kfz_filter, query_obj=_FQ)

    result_ids = [res.section_id for res in results]
    assert 701 not in result_ids, (
        f"Hausrat section 701 must NOT be forced in — it's outside DocFilter gate. Results: {result_ids}"
    )
    assert 700 in result_ids, "Kfz Naturgefahren section 700 must be in results"


# ── D2: near-duplicate dedup ──────────────────────────────────────────────────

def _make_near_dup_sections(n: int, *, identical_emb: bool = True) -> tuple[list[dict], np.ndarray]:
    """Build n nearly-identical sections with pre-computed embeddings."""
    secs = [
        {
            "section_id": 800 + i,
            "doc_id": f"hausrat_{i}",
            "sparte": "Hausrat",
            "tarif": f"Tarif{i}",
            "heading": f"Was ist versichert {i}",
            "markdown": "Versichert ist der gesamte Hausrat des Versicherungsnehmers.",
            "breadcrumb": f"Hausrat > §1 > {i}",
            "section_types": ["WHAT_IS_INSURED"],
            "topic_tags": [],
            "is_retrieval_unit": True,
        }
        for i in range(n)
    ]
    # All sections get emb [1.0, 0.0] (identical), or slightly varied if not identical_emb
    if identical_emb:
        embs = np.array([[1.0, 0.0]] * n, dtype=np.float32)
    else:
        # cosine ~0.95 — below dedup threshold
        embs = np.array([[1.0, 0.0], [0.95, 0.31]], dtype=np.float32)
    return secs, embs


def test_d2_registry_has_dedup_threshold():
    """D2: REGISTRY must have 'dedup_threshold' key with value 0.98."""
    assert "dedup_threshold" in REGISTRY
    assert float(REGISTRY["dedup_threshold"]) == 0.98


def test_d2_retrieval_result_has_shared_tarifs():
    """D2: RetrievalResult must have shared_tarifs field (list[str])."""
    r = RetrievalResult(
        section_id=1, doc_id="d", sparte="K", tarif="T",
        heading="H", markdown="M", pruned_markdown="M", breadcrumb="B",
        score=0.9, section_types=[], topic_tags=[],
    )
    assert hasattr(r, "shared_tarifs")
    assert isinstance(r.shared_tarifs, list)


def test_d2_near_dups_collapsed_to_one():
    """D2: 4 near-identical chunks (cosine=1.0 > 0.98) → 1 representative in pool."""
    import numpy as np

    secs, embs = _make_near_dup_sections(4)
    # Give increasing scores so representative is well-defined
    scores = np.array([0.7, 0.9, 0.8, 0.6], dtype=np.float32)

    emb = MagicMock()
    call_count = [0]
    def encode_d2(texts, **kw):
        texts = [texts] if isinstance(texts, str) else list(texts)
        count = call_count[0]
        call_count[0] += 1
        if count == 0:
            # Building index — return pre-computed section embs
            return embs
        else:
            # Query encoding
            return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)
    emb.encode.side_effect = encode_d2

    mock_reranker = MagicMock()
    pool_received = []
    def capture(q, results):
        pool_received.extend(results)
        return results[:1]
    mock_reranker.rerank.side_effect = capture

    r = Retriever(sections=secs, embedder=emb, reranker=mock_reranker)
    results = r.retrieve("Was ist versichert?", top_k=1)

    # 4 near-identical sections → collapsed to 1 before reranker
    assert mock_reranker.rerank.called
    assert len(pool_received) == 1, (
        f"4 near-dups should collapse to 1, reranker received {len(pool_received)}"
    )
    rep = pool_received[0]
    assert len(rep.shared_tarifs) == 4, (
        f"shared_tarifs must list all 4 tarifs, got {rep.shared_tarifs}"
    )


def test_d2_representative_is_highest_score():
    """D2: representative of deduplicated cluster is chunk with max boosted score."""
    import numpy as np

    secs, embs = _make_near_dup_sections(3)
    # Assign distinct scores: section 801 gets highest score
    # We need to control scores — use embs that produce known cosine with query
    # query emb = [1,0], so score = emb[0]
    embs = np.array([[0.7, 0.0], [0.9, 0.0], [0.5, 0.0]], dtype=np.float32)

    emb = MagicMock()
    call_count = [0]
    def encode_d2(texts, **kw):
        texts = [texts] if isinstance(texts, str) else list(texts)
        count = call_count[0]
        call_count[0] += 1
        if count == 0:
            return embs
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)
    emb.encode.side_effect = encode_d2

    mock_reranker = MagicMock()
    pool_received = []
    def capture(q, results):
        pool_received.extend(results)
        return results[:1]
    mock_reranker.rerank.side_effect = capture

    r = Retriever(sections=secs, embedder=emb, reranker=mock_reranker)
    results = r.retrieve("Was?", top_k=1)

    assert pool_received, "Reranker must have been called"
    rep = pool_received[0]
    assert rep.section_id == 801, (
        f"Representative must be section 801 (highest score 0.9), got {rep.section_id}"
    )


def test_d2_below_threshold_not_collapsed():
    """D2: chunks with cosine 0.95 (below 0.98 threshold) are NOT collapsed."""
    import numpy as np

    secs = [
        {
            "section_id": 810, "doc_id": "doc_a", "sparte": "K", "tarif": "T1",
            "heading": "A", "markdown": "Content A.", "breadcrumb": "A",
            "section_types": [], "topic_tags": [], "is_retrieval_unit": True,
        },
        {
            "section_id": 811, "doc_id": "doc_b", "sparte": "K", "tarif": "T2",
            "heading": "B", "markdown": "Content B.", "breadcrumb": "B",
            "section_types": [], "topic_tags": [], "is_retrieval_unit": True,
        },
    ]
    # cosine ≈ 0.951: [1,0] · [0.951, 0.309] = 0.951 (below threshold 0.98)
    embs = np.array([[1.0, 0.0], [0.951, 0.309]], dtype=np.float32)

    emb = MagicMock()
    call_count = [0]
    def encode_d2(texts, **kw):
        texts = [texts] if isinstance(texts, str) else list(texts)
        count = call_count[0]
        call_count[0] += 1
        if count == 0:
            return embs
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)
    emb.encode.side_effect = encode_d2

    mock_reranker = MagicMock()
    pool_received = []
    def capture(q, results):
        pool_received.extend(results)
        return results[:1]
    mock_reranker.rerank.side_effect = capture

    r = Retriever(sections=secs, embedder=emb, reranker=mock_reranker)
    results = r.retrieve("test", top_k=1)  # top_k=1 < pool=2 → reranker called

    assert mock_reranker.rerank.called
    assert len(pool_received) == 2, (
        f"cosine=0.95 < 0.98 threshold → 2 candidates must remain, got {len(pool_received)}"
    )
    assert all(res.shared_tarifs == [] for res in pool_received), (
        "Non-deduped chunks must have empty shared_tarifs"
    )
