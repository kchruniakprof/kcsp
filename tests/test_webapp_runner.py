"""Unit tests for KcspRunner — uses fake sub-component RAGAssistant."""
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

import pandas as pd

from src.webapp_runner import KcspRunner
from src.tracing import TraceCollector, set_active_collector
from src.query_expansion import Intent


# ── Fake sub-components ───────────────────────────────────────────────────────

class FakeExpandedQuery:
    intent = Intent.COVERAGE_QUERY
    normalized_query = "Was ist versichert?"
    paraphrases = ["Was wird abgedeckt?", "Was ist im Schutz enthalten?", "Was deckt die Versicherung ab?"]
    domain_terms = ["Versicherung", "Deckung"]
    section_types = ["coverage"]
    sparte_hints = ["Hausrat"]
    confidence_score = 0.9
    chain_of_thought = ["Query is factual.", "Hausrat sparte detected."]

    @property
    def primary_sparte(self):
        return "Hausrat"


class FakeOOSExpandedQuery:
    intent = Intent.OUT_OF_SCOPE
    normalized_query = "Tell me a joke"
    paraphrases = []
    domain_terms = []
    section_types = []
    sparte_hints = []
    confidence_score = 0.5
    chain_of_thought = []

    @property
    def primary_sparte(self):
        return None


@dataclass
class FakeRetrievalResult:
    section_id: int
    heading: str
    markdown: str
    breadcrumb: str
    score: float


@dataclass
class FakeGeneratedAnswer:
    answer: str
    sources: list
    breadcrumbs: list
    mode: object


class FakeCriticResult:
    verdict = MagicMock()
    reason = "Answer is correct."
    confidence = 0.95
    answer = "Test answer about insurance"
    retried = False
    used_ensemble = False
    chain_of_thought = ["Checked source 1.", "Answer matches."]

    def __init__(self):
        from src.critic import CriticVerdict
        self.verdict = CriticVerdict.PASS


class FakeQueryExpansion:
    def expand(self, query):
        if "joke" in query.lower():
            return FakeOOSExpandedQuery()
        return FakeExpandedQuery()


class FakeRetriever:
    def retrieve_multi(self, queries, top_k, section_types, doc_filter, query_obj):
        return [
            FakeRetrievalResult(1, "Hausrat Grundschutz", "# Grundschutz\nFeuer und Wasser", "Hausrat > Grundschutz", 0.92),
            FakeRetrievalResult(2, "Hausrat Ausschlüsse", "# Ausschlüsse\nKrieg ist ausgeschlossen", "Hausrat > Ausschlüsse", 0.85),
            FakeRetrievalResult(3, "Hausrat Prämien", "# Prämien\n20€ monatlich", "Hausrat > Prämien", 0.72),
        ]


class FakeGenerator:
    def generate(self, query, sections, mode):
        from src.generator import AnswerMode
        return FakeGeneratedAnswer(
            answer="Test answer about insurance",
            sources=[1, 2, 3],
            breadcrumbs=["Hausrat > Grundschutz", "Hausrat > Ausschlüsse", "Hausrat > Prämien"],
            mode=mode,
        )


class FakeCritic:
    def evaluate(self, query, answer, sections):
        return FakeCriticResult()


def _make_fake_rag(enable_cross_sell=False):
    rag = MagicMock()
    rag._query_expansion = FakeQueryExpansion()
    rag._retriever = FakeRetriever()
    rag._generator = FakeGenerator()
    rag._critic = FakeCritic()
    rag._top_k = 5
    rag._enable_cross_sell = enable_cross_sell
    rag._documents_df = None
    rag._sections_df = None
    rag._subsections_df = None
    rag._ensemble_critic = None
    rag._enable_ensemble = False
    return rag


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_stage_callback_called_with_query_expansion():
    stages = []
    runner = KcspRunner(rag=_make_fake_rag())
    collector = TraceCollector()
    set_active_collector(collector)
    runner.run("Was ist versichert?", stage_callback=lambda s: stages.append(s))
    assert "query_expansion" in stages
    set_active_collector(None)


def test_answer_markdown_populated():
    runner = KcspRunner(rag=_make_fake_rag())
    collector = TraceCollector()
    set_active_collector(collector)
    answer = runner.run("Test query")
    assert answer.answer_markdown == "Test answer about insurance"
    set_active_collector(None)


def test_abstained_maps_correctly():
    runner = KcspRunner(rag=_make_fake_rag())
    collector = TraceCollector()
    set_active_collector(collector)
    answer = runner.run("Tell me a joke")
    assert answer.abstained is True
    assert answer.audit_metadata["critic_verdict"] == "ABSTAIN"
    set_active_collector(None)


def test_cited_sources_populated():
    runner = KcspRunner(rag=_make_fake_rag())
    collector = TraceCollector()
    set_active_collector(collector)
    answer = runner.run("Coverage question")
    assert len(answer.cited_sources) == 3
    assert 1 in answer.cited_sources
    set_active_collector(None)


def test_cost_eur_nonnegative():
    runner = KcspRunner(rag=_make_fake_rag())
    collector = TraceCollector()
    set_active_collector(collector)
    runner.run("Question")
    assert collector.total_cost_eur >= 0
    set_active_collector(None)


def test_query_expansion_detail_in_collector():
    runner = KcspRunner(rag=_make_fake_rag())
    collector = TraceCollector()
    set_active_collector(collector)
    runner.run("Was ist versichert?")
    qe = collector.query_expansion_detail
    assert qe is not None
    assert qe["intent"] == "COVERAGE_QUERY"
    assert len(qe["chain_of_thought"]) > 0
    assert len(qe["paraphrases"]) > 0
    set_active_collector(None)


def test_selected_chunks_have_markdown():
    runner = KcspRunner(rag=_make_fake_rag())
    collector = TraceCollector()
    set_active_collector(collector)
    answer = runner.run("Was ist versichert?")
    chunks = answer.audit_metadata.get("selected_chunks", [])
    assert len(chunks) == 3
    assert all("markdown" in c for c in chunks)
    assert all(c["markdown"] for c in chunks)
    set_active_collector(None)


def test_critic_cot_in_audit_metadata():
    runner = KcspRunner(rag=_make_fake_rag())
    collector = TraceCollector()
    set_active_collector(collector)
    answer = runner.run("Was ist versichert?")
    assert isinstance(answer.audit_metadata.get("critic_cot"), list)
    assert len(answer.audit_metadata["critic_cot"]) > 0
    set_active_collector(None)


def test_all_stages_emitted():
    stages = []
    runner = KcspRunner(rag=_make_fake_rag())
    collector = TraceCollector()
    set_active_collector(collector)
    runner.run("Was ist versichert?", stage_callback=lambda s: stages.append(s))
    for expected in ("query_expansion", "retrieval", "generation", "critic"):
        assert expected in stages, f"Missing stage: {expected}"
    set_active_collector(None)
