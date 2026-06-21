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
    sparte_hints = [sparte_hint] if sparte_hint is not None else []
    return ExpandedQuery(
        original_query="Was ist versichert?",
        normalized_query="Was ist versichert?",
        detected_language="de",
        intent=intent,
        sparte_hints=sparte_hints,
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


# ---------------------------------------------------------------------------
# G4: sparte_hints + tarif wiring through RAGAssistant
# ---------------------------------------------------------------------------

import pandas as pd


def _make_docs_df():
    return pd.DataFrame([
        {"doc_id": "kfz-spezial",  "sparte": "Kfz",    "tarif": "Spezial"},
        {"doc_id": "kfz-standard", "sparte": "Kfz",    "tarif": "Standard"},
        {"doc_id": "hausrat-best", "sparte": "Hausrat", "tarif": "Best"},
        {"doc_id": "hausrat-smart","sparte": "Hausrat", "tarif": "Smart"},
        {"doc_id": "glas-1",       "sparte": "Glas",    "tarif": "KT2021GLHR"},
    ])


def _make_sections_df():
    return pd.DataFrame([
        {"doc_id": "kfz-spezial", "topic_tags": ["Unfallschaden"]},
        {"doc_id": "hausrat-best","topic_tags": ["Hausrat"]},
    ])


def _make_subs_df():
    return pd.DataFrame([
        {"doc_id": "kfz-standard", "topic_tags": []},
    ])


def _rag_with_dfs(**kwargs):
    """RAGAssistant with sub-components mocked + real DataFrames injected."""
    qe = MagicMock()
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
        documents_df=_make_docs_df(),
        sections_df=_make_sections_df(),
        subsections_df=_make_subs_df(),
        **kwargs,
    )


def _get_doc_filter_from_call(rag):
    """Extract doc_filter passed to retrieve_multi."""
    call_kwargs = rag._retriever.retrieve_multi.call_args
    return call_kwargs[1].get("doc_filter") if call_kwargs[1] else None


def test_doc_filter_built_when_documents_df_provided():
    rag = _rag_with_dfs()
    rag._query_expansion.expand.return_value = _make_expanded()
    rag.ask("Was ist versichert?")
    doc_filter = _get_doc_filter_from_call(rag)
    assert doc_filter is not None, "doc_filter must be built when documents_df provided"


def test_tarif_in_query_narrows_doc_filter():
    """'Spezial' in query → doc_filter restricts to Kfz Spezial doc only."""
    rag = _rag_with_dfs()
    expanded = _make_expanded(sparte_hint="Kfz")
    # Override normalized_query to mention Spezial
    expanded = ExpandedQuery(
        original_query="Kfz Spezial Deckung",
        normalized_query="Kfz Spezial Deckungsumfang",
        detected_language="de",
        intent=Intent.COVERAGE_QUERY,
        sparte_hints=["Kfz"],
        chain_of_thought=["step 1", "step 2", "step 3"],
        paraphrases=["Para 1", "Para 2", "Para 3"],
        domain_terms=["Kfz", "Spezial"],
        confidence_score=0.9,
    )
    rag._query_expansion.expand.return_value = expanded
    rag.ask("Kfz Spezial Deckung")
    doc_filter = _get_doc_filter_from_call(rag)
    result = doc_filter.filter(expanded)
    assert result is not None
    assert "kfz-spezial" in result
    assert "kfz-standard" not in result, "Standard must be excluded when Spezial in query"


def test_multi_sparte_gate_excludes_non_hinted_sparte():
    """sparte_hints=['Kfz','Hausrat'] → gate never admits Glas docs, even if rare would."""
    rag = _rag_with_dfs()
    expanded = ExpandedQuery(
        original_query="Vergleich Kfz und Hausrat",
        normalized_query="Vergleich Kfz und Hausrat",
        detected_language="de",
        intent=Intent.COMPARISON,
        sparte_hints=["Kfz", "Hausrat"],
        chain_of_thought=["step 1", "step 2", "step 3"],
        paraphrases=["Para 1", "Para 2", "Para 3"],
        domain_terms=[],  # no rare terms → gate returned as-is
        confidence_score=0.85,
    )
    rag._query_expansion.expand.return_value = expanded
    rag.ask("Vergleich Kfz und Hausrat")
    doc_filter = _get_doc_filter_from_call(rag)
    result = doc_filter.filter(expanded)
    assert result is not None
    docs_df = _make_docs_df()
    matched = docs_df[docs_df["doc_id"].isin(result)]
    sparten = set(matched["sparte"].unique())
    assert "Kfz" in sparten, "Kfz must be in gate for multi-sparte query"
    assert "Hausrat" in sparten, "Hausrat must be in gate for multi-sparte query"
    assert "Glas" not in sparten, "Glas must be excluded (not in sparte_hints)"


def test_cross_sell_uses_primary_sparte():
    """Cross-sell looks up _CROSS_SELL_MAP by primary_sparte (first of sparte_hints)."""
    rag = _rag_with_dfs(enable_cross_sell=True)
    expanded = ExpandedQuery(
        original_query="Was deckt Hausrat ab?",
        normalized_query="Was deckt Hausrat ab?",
        detected_language="de",
        intent=Intent.COVERAGE_QUERY,
        sparte_hints=["Hausrat"],
        chain_of_thought=["step 1", "step 2", "step 3"],
        paraphrases=["Para 1", "Para 2", "Para 3"],
        domain_terms=["Hausrat"],
        confidence_score=0.9,
    )
    rag._query_expansion.expand.return_value = expanded
    result = rag.ask("Was deckt Hausrat ab?")
    assert result.cross_sell is not None
    assert "Glas" in result.cross_sell or "Schmuck" in result.cross_sell
