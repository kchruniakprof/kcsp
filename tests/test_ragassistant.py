"""Tests for ragassistant orchestrator — TDD cycle."""
import pytest
from unittest.mock import MagicMock, patch

from src.ragassistant import RAGAssistant, FinalAnswer
from src.query_expansion import ExpandedQuery, Intent
from src.retriever import RetrievalResult
from src.generator import GeneratedAnswer, AnswerMode
from src.critic import CriticResult, CriticVerdict


# ---------------------------------------------------------------------------
# FinalAnswer
# ---------------------------------------------------------------------------

def test_final_answer_fields():
    fa = FinalAnswer(
        answer="Versichert ist der Hausrat.",
        sources=[1, 2],
        breadcrumbs=["Hausrat > Smart > §1"],
        intent=Intent.COVERAGE_QUERY,
        abstained=False,
        cross_sell=None,
    )
    assert fa.abstained is False
    assert fa.intent == Intent.COVERAGE_QUERY


def test_final_answer_abstained_flag():
    fa = FinalAnswer(
        answer="",
        sources=[],
        breadcrumbs=[],
        intent=Intent.OUT_OF_SCOPE,
        abstained=True,
        cross_sell=None,
    )
    assert fa.abstained is True


# ---------------------------------------------------------------------------
# Helpers — mocked sub-components
# ---------------------------------------------------------------------------

def _make_expanded(intent=Intent.COVERAGE_QUERY, sparte_hint="Hausrat") -> ExpandedQuery:
    return ExpandedQuery(
        original_query="Was ist versichert?",
        normalized_query="Was ist versichert?",
        detected_language="de",
        intent=intent,
        sparte_hint=sparte_hint,
        chain_of_thought=["step 1", "step 2", "step 3"],
        paraphrases=["Para 1", "Para 2", "Para 3"],
        domain_terms=["Hausrat", "Deckung"],
        confidence_score=0.9,
    )


def _make_sections() -> list[RetrievalResult]:
    return [RetrievalResult(
        section_id=1, doc_id="hausrat_smart", sparte="Hausrat", tarif="Smart",
        heading="1. Was ist versichert",
        markdown="## 1. Was ist versichert\nVersichert ist der Hausrat.",
        pruned_markdown="## 1. Was ist versichert\nVersichert ist der Hausrat.",
        breadcrumb="Hausrat > Smart > §1",
        score=0.9, section_types=["WHAT_IS_INSURED"], topic_tags=["Deckungsumfang"],
    )]


def _make_generated(answer="## 1. Was ist versichert\nVersichert.") -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=answer, sources=[1], mode=AnswerMode.VERBATIM, breadcrumbs=["Hausrat > Smart > §1"]
    )


def _make_critic_pass() -> CriticResult:
    return CriticResult(verdict=CriticVerdict.PASS, reason="ok", confidence=0.9)


def _make_critic_abstain() -> CriticResult:
    return CriticResult(verdict=CriticVerdict.ABSTAIN, reason="Halluzination.", confidence=0.1)


@pytest.fixture
def rag(monkeypatch):
    """RAGAssistant with all sub-components mocked."""
    qe = MagicMock()
    qe.expand.return_value = _make_expanded()

    retriever = MagicMock()
    retriever.retrieve_multi.return_value = _make_sections()

    generator = MagicMock()
    generator.generate.return_value = _make_generated()

    critic = MagicMock()
    critic.evaluate.return_value = _make_critic_pass()

    return RAGAssistant(
        query_expansion=qe,
        retriever=retriever,
        generator=generator,
        critic=critic,
    )


# ---------------------------------------------------------------------------
# Basic orchestration
# ---------------------------------------------------------------------------

def test_ask_returns_final_answer(rag):
    result = rag.ask("Was ist versichert?")
    assert isinstance(result, FinalAnswer)


def test_ask_answer_nonempty_on_pass(rag):
    result = rag.ask("Was ist versichert?")
    assert result.answer.strip() != ""


def test_ask_abstained_false_on_pass(rag):
    result = rag.ask("Was ist versichert?")
    assert result.abstained is False


def test_ask_sources_populated_on_pass(rag):
    result = rag.ask("Was ist versichert?")
    assert len(result.sources) > 0


def test_ask_breadcrumbs_populated(rag):
    result = rag.ask("Was ist versichert?")
    assert len(result.breadcrumbs) > 0


def test_ask_intent_set(rag):
    result = rag.ask("Was ist versichert?")
    assert result.intent == Intent.COVERAGE_QUERY


# ---------------------------------------------------------------------------
# Abstain path
# ---------------------------------------------------------------------------

def test_ask_abstains_when_critic_abstains(monkeypatch):
    qe = MagicMock(); qe.expand.return_value = _make_expanded()
    retriever = MagicMock(); retriever.retrieve_multi.return_value = _make_sections()
    generator = MagicMock(); generator.generate.return_value = _make_generated()
    critic = MagicMock(); critic.evaluate.return_value = _make_critic_abstain()

    rag = RAGAssistant(query_expansion=qe, retriever=retriever,
                       generator=generator, critic=critic)
    result = rag.ask("Lebensversicherung?")
    assert result.abstained is True


def test_ask_abstains_when_no_sections(monkeypatch):
    qe = MagicMock(); qe.expand.return_value = _make_expanded()
    retriever = MagicMock(); retriever.retrieve_multi.return_value = []
    generator = MagicMock()
    critic = MagicMock()

    rag = RAGAssistant(query_expansion=qe, retriever=retriever,
                       generator=generator, critic=critic)
    result = rag.ask("unbekannte Frage")
    assert result.abstained is True
    generator.generate.assert_not_called()


def test_ask_out_of_scope_abstains(monkeypatch):
    qe = MagicMock()
    qe.expand.return_value = _make_expanded(intent=Intent.OUT_OF_SCOPE, sparte_hint=None)
    retriever = MagicMock(); retriever.retrieve_multi.return_value = []
    generator = MagicMock()
    critic = MagicMock()

    rag = RAGAssistant(query_expansion=qe, retriever=retriever,
                       generator=generator, critic=critic)
    result = rag.ask("Wer gewinnt die WM?")
    assert result.abstained is True


# ---------------------------------------------------------------------------
# Cross-sell hint
# ---------------------------------------------------------------------------

def test_cross_sell_glas_when_hausrat(monkeypatch):
    """When sparte=Hausrat retrieved, cross_sell may hint Glas/Schmuck."""
    qe = MagicMock()
    qe.expand.return_value = _make_expanded(sparte_hint="Hausrat")
    retriever = MagicMock(); retriever.retrieve_multi.return_value = _make_sections()
    generator = MagicMock(); generator.generate.return_value = _make_generated()
    critic = MagicMock(); critic.evaluate.return_value = _make_critic_pass()

    rag = RAGAssistant(query_expansion=qe, retriever=retriever,
                       generator=generator, critic=critic,
                       enable_cross_sell=True)
    result = rag.ask("Was ist im Hausrat Smart versichert?")
    # cross_sell may be None or a list of sparte strings
    assert result.cross_sell is None or isinstance(result.cross_sell, list)


# ---------------------------------------------------------------------------
# Pipeline wiring — sub-components called
# ---------------------------------------------------------------------------

def test_pipeline_calls_query_expansion(rag):
    rag.ask("test")
    rag._query_expansion.expand.assert_called_once_with("test")


def test_pipeline_calls_retriever(rag):
    rag.ask("test")
    assert rag._retriever.retrieve_multi.called


def test_pipeline_calls_generator(rag):
    rag.ask("test")
    assert rag._generator.generate.called


def test_pipeline_calls_critic(rag):
    rag.ask("test")
    assert rag._critic.evaluate.called


# ---------------------------------------------------------------------------
# D6: abstain on empty retrieval (doc_filter path)
# ---------------------------------------------------------------------------

def test_ask_abstains_when_retriever_returns_empty_list():
    qe = MagicMock(); qe.expand.return_value = _make_expanded()
    retriever = MagicMock(); retriever.retrieve_multi.return_value = []
    generator = MagicMock()
    critic = MagicMock()
    rag = RAGAssistant(query_expansion=qe, retriever=retriever,
                       generator=generator, critic=critic)
    result = rag.ask("test")
    assert result.abstained is True
    generator.generate.assert_not_called()


def test_retrieve_multi_called_without_sparte_param(rag):
    """retrieve_multi must not be called with sparte= (removed in D5)."""
    rag.ask("test")
    call_kwargs = rag._retriever.retrieve_multi.call_args
    all_kwargs = call_kwargs[1] if call_kwargs[1] else {}
    assert "sparte" not in all_kwargs, "ragassistant must not pass sparte= to retrieve_multi"
