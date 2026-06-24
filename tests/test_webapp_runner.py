"""Unit tests for KcspRunner — uses fake RAGAssistant."""
from dataclasses import dataclass
from typing import Optional
from src.webapp_runner import KcspRunner
from src.tracing import TraceCollector, set_active_collector


@dataclass
class FakeFinalAnswer:
    answer: str
    sources: list
    breadcrumbs: list
    intent: str
    abstained: bool
    cross_sell: Optional[list] = None


class FakeRAGAssistant:
    def ask(self, query: str) -> FakeFinalAnswer:
        return FakeFinalAnswer(
            answer="Test answer about insurance",
            sources=[1, 2, 3],
            breadcrumbs=["Hausrat > Was ist versichert", "Hausrat > Ausschlüsse", "Hausrat > Prämien"],
            intent="FACTUAL",
            abstained=False,
        )


class FakeAbstainRAGAssistant:
    def ask(self, query: str) -> FakeFinalAnswer:
        return FakeFinalAnswer(
            answer="Keine gesicherte Antwort möglich.",
            sources=[],
            breadcrumbs=[],
            intent="OUT_OF_SCOPE",
            abstained=True,
        )


def test_stage_callback_called_with_query_expansion():
    stages = []
    runner = KcspRunner(rag=FakeRAGAssistant())
    collector = TraceCollector()
    set_active_collector(collector)
    runner.run("Was ist versichert?", stage_callback=lambda s: stages.append(s))
    assert "query_expansion" in stages
    set_active_collector(None)


def test_answer_markdown_populated():
    runner = KcspRunner(rag=FakeRAGAssistant())
    collector = TraceCollector()
    set_active_collector(collector)
    answer = runner.run("Test query")
    assert answer.answer_markdown == "Test answer about insurance"
    set_active_collector(None)


def test_abstained_maps_correctly():
    runner = KcspRunner(rag=FakeAbstainRAGAssistant())
    collector = TraceCollector()
    set_active_collector(collector)
    answer = runner.run("Tell me a joke")
    assert answer.abstained is True
    assert answer.audit_metadata["critic_verdict"] == "ABSTAIN"
    set_active_collector(None)


def test_cited_sources_populated():
    runner = KcspRunner(rag=FakeRAGAssistant())
    collector = TraceCollector()
    set_active_collector(collector)
    answer = runner.run("Coverage question")
    assert len(answer.cited_sources) == 3
    assert 1 in answer.cited_sources
    set_active_collector(None)


def test_cost_eur_nonnegative():
    runner = KcspRunner(rag=FakeRAGAssistant())
    collector = TraceCollector()
    set_active_collector(collector)
    runner.run("Question")
    assert collector.total_cost_eur >= 0
    set_active_collector(None)
