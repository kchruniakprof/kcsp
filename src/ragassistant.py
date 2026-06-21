"""
RAGAssistant: orchestrator for the full RAG pipeline.
Flow: QueryExpansion → Retriever → Generator → Critic → FinalAnswer.
Abstains when: OUT_OF_SCOPE intent, no sections retrieved, or Critic=ABSTAIN.
Cross-sell: when Hausrat retrieved, hint Glas/Schmuck if enabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from typing import TYPE_CHECKING

import pandas as pd

from src.critic import Critic, CriticVerdict, run_critic
from src.doc_filter import CompositeDocFilter, ProductDetectorAdapter, RareTagMatcherAdapter, _detect_tarif
from src.generator import AnswerMode, Generator
from src.observability import get_logger
from src.query_expansion import Intent, QueryExpansion
from src.retriever import Retriever
from src.schemas import SectionRecord

_log = get_logger("ragassistant")


_CROSS_SELL_MAP: dict[str, list[str]] = {
    "Hausrat": ["Glas", "Schmuck"],
    "Kfz":     [],
    "Glas":    ["Schmuck"],
    "Schmuck": [],
}

_ABSTAIN_ANSWER = (
    "Ich kann auf Basis der vorliegenden Bedingungen keine gesicherte Antwort geben. "
    "Bitte wenden Sie sich an den zuständigen Produktspezialisten."
)


@dataclass
class FinalAnswer:
    answer: str
    sources: list[int]
    breadcrumbs: list[str]
    intent: Intent
    abstained: bool
    cross_sell: Optional[list[str]] = None


class RAGAssistant:
    def __init__(
        self,
        query_expansion: QueryExpansion,
        retriever: Retriever,
        generator: Generator,
        critic: Critic,
        top_k: int = 5,
        enable_cross_sell: bool = False,
        documents_df: Optional[pd.DataFrame] = None,
        sections_df: Optional[pd.DataFrame] = None,
        subsections_df: Optional[pd.DataFrame] = None,
        ensemble_critic: Optional[Critic] = None,
        enable_ensemble: bool = False,
    ) -> None:
        self._query_expansion = query_expansion
        self._retriever = retriever
        self._generator = generator
        self._critic = critic
        self._top_k = top_k
        self._enable_cross_sell = enable_cross_sell
        self._documents_df = documents_df
        self._sections_df = sections_df
        self._subsections_df = subsections_df
        self._ensemble_critic = ensemble_critic
        self._enable_ensemble = enable_ensemble

    def ask(self, query: str) -> FinalAnswer:
        _log.info("pipeline_start", query=query)

        # 1. Expand + classify
        expanded = self._query_expansion.expand(query)

        # 2. OUT_OF_SCOPE short-circuit
        if expanded.intent == Intent.OUT_OF_SCOPE:
            _log.info("pipeline_abstain", reason="out_of_scope", query=query)
            return FinalAnswer(
                answer=_ABSTAIN_ANSWER, sources=[], breadcrumbs=[],
                intent=expanded.intent, abstained=True,
            )

        # 3. Retrieve
        mode = (
            AnswerMode.COMPARE
            if expanded.intent == Intent.COMPARISON
            else AnswerMode.VERBATIM
        )
        queries = [expanded.normalized_query] + list(expanded.paraphrases or [])

        # Build DocFilter when parquet DataFrames available
        doc_filter = None
        if self._documents_df is not None and self._sections_df is not None and self._subsections_df is not None:
            tarif_names = list(self._documents_df["tarif"].dropna().unique())
            detected_tarif = _detect_tarif(expanded.normalized_query, tarif_names)
            adapters = [
                ProductDetectorAdapter(self._documents_df, tarif=detected_tarif),
                RareTagMatcherAdapter(self._sections_df, self._subsections_df),
            ]
            doc_filter = CompositeDocFilter(adapters)

        results = self._retriever.retrieve_multi(
            queries=queries,
            top_k=self._top_k,
            section_types=expanded.section_types or None,
            doc_filter=doc_filter,
            query_obj=expanded,
        )

        if not results:
            _log.info("pipeline_abstain", reason="empty_retrieval",
                      normalized_query=expanded.normalized_query,
                      sparte_hint=expanded.primary_sparte)
            return FinalAnswer(
                answer=_ABSTAIN_ANSWER, sources=[], breadcrumbs=[],
                intent=expanded.intent, abstained=True,
            )

        # 4. Build typed section records (full markdown for both Generator and Critic)
        sections: list[SectionRecord] = [
            SectionRecord(
                section_id=r.section_id,
                heading=r.heading,
                markdown=r.markdown,
                breadcrumb=r.breadcrumb,
            )
            for r in results
        ]

        generated = self._generator.generate(expanded.normalized_query, sections, mode=mode)

        # 5. Critic — skipped for COMPARE
        if mode != AnswerMode.COMPARE:
            critic_sections = sections

            def _generate_fn() -> str:
                return self._generator.generate(
                    expanded.normalized_query, generator_sections, mode=mode
                ).answer

            critic_result = run_critic(
                query=expanded.normalized_query,
                answer=generated.answer,
                sections=critic_sections,
                critic=self._critic,
                generate_fn=_generate_fn,
                ensemble_critic=self._ensemble_critic,
                enable_ensemble=self._enable_ensemble,
            )
        else:
            critic_result = None

        if critic_result is not None and critic_result.verdict == CriticVerdict.ABSTAIN:
            _log.info("pipeline_abstain", reason="critic_abstain",
                      critic_reason=critic_result.reason,
                      critic_confidence=critic_result.confidence)
            return FinalAnswer(
                answer=_ABSTAIN_ANSWER,
                sources=[r.section_id for r in results],
                breadcrumbs=[r.breadcrumb for r in results],
                intent=expanded.intent,
                abstained=True,
            )

        # 6. Cross-sell hint — append to answer text so it's visible
        cross_sell: Optional[list[str]] = None
        # Use regenerated answer if critic ran REGEN→PASS; else use original generated answer
        answer_text = (
            critic_result.answer
            if critic_result is not None and critic_result.answer is not None
            else generated.answer
        )
        if self._enable_cross_sell and expanded.primary_sparte:
            hints = _CROSS_SELL_MAP.get(expanded.primary_sparte, [])
            cross_sell = hints if hints else None
            if cross_sell:
                hint_str = ", ".join(cross_sell)
                answer_text += (
                    f"\n\n---\n**Ergänzende Produkte:** {hint_str}"
                )

        _log.info("pipeline_done", intent=expanded.intent.value,
                  mode=mode.value, abstained=False,
                  answer_len=len(answer_text),
                  sources=[r.section_id for r in results],
                  cross_sell=cross_sell)
        return FinalAnswer(
            answer=answer_text,
            sources=generated.sources,
            breadcrumbs=generated.breadcrumbs,
            intent=expanded.intent,
            abstained=False,
            cross_sell=cross_sell,
        )
