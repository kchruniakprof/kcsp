"""KcspRunner: adapter between webapp chat.py and RAGAssistant pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.tracing import TraceCollector, get_active_collector, set_active_collector

_ABSTAIN_ANSWER = (
    "Ich kann auf Basis der vorliegenden Bedingungen keine gesicherte Antwort geben. "
    "Bitte wenden Sie sich an den zuständigen Produktspezialisten."
)


@dataclass
class KcspAnswer:
    """Mimics dkv's answer object interface expected by chat.py run_rag_in_background."""

    answer_markdown: str
    abstained: bool
    cited_sources: list[int]
    retrieved_doc_ids: list[int]
    used_brute_force: bool = False
    audit_metadata: dict = field(default_factory=dict)


def _ensure_tracing_clients(rag: Any) -> None:
    """One-time idempotent wrapping of RAG component clients with TracingClient.

    Safe on the lru-cached RAG singleton: TracingClient is stateless per-call
    (uses thread-local collector), so wrapping once is enough for all requests.
    Silently skips components that lack _client (e.g. test fakes).
    """
    from src.tracing_client import TracingClient

    def _wrap(component: Any, name: str) -> None:
        client = getattr(component, "_client", None)
        model = getattr(component, "_model", "unknown")
        if client is None or isinstance(client, TracingClient):
            return
        component._client = TracingClient(client, model_id=model, provider="groq", name=name)

    _wrap(rag._query_expansion, "query_expansion")
    _wrap(rag._generator, "generator")
    _wrap(rag._critic, "critic")
    if rag._ensemble_critic is not None:
        _wrap(rag._ensemble_critic, "critic_ensemble")


def _stage_step(collector: TraceCollector, steps_before: int) -> dict | None:
    """Aggregate TraceCollector steps added since steps_before into a summary dict."""
    delta = collector.steps[steps_before:]
    if not delta:
        return None
    pt = sum(s.prompt_tokens for s in delta)
    ct = sum(s.completion_tokens for s in delta)
    return {
        "model": delta[-1].model_id,
        "provider": delta[-1].provider,
        "tokens_prompt": pt,
        "tokens_completion": ct,
        "cost_eur": sum(s.cost_eur for s in delta),
        "duration_ms": sum(s.duration_ms for s in delta),
    }


class KcspRunner:
    """
    Wraps RAGAssistant with:
    - stage_callback support (emits 4 stage names)
    - TraceCollector integration
    - KcspAnswer return type compatible with dkv chat.py
    """

    def __init__(self, rag: Any = None) -> None:
        if rag is None:
            from src.promptfoo_provider import _get_rag
            rag = _get_rag()
        self._rag = rag

    def run(
        self,
        query: str,
        stage_callback: Optional[Callable[[str], None]] = None,
    ) -> KcspAnswer:
        """Run RAG pipeline step-by-step, capturing intermediates for the trace."""
        collector = get_active_collector()
        if collector is None:
            collector = TraceCollector()
            set_active_collector(collector)

        rag = self._rag
        _ensure_tracing_clients(rag)

        # ── Stage 1: Query Expansion ──────────────────────────────────────────
        if stage_callback:
            stage_callback("query_expansion")

        _steps_before_qe = len(collector.steps)
        expanded = rag._query_expansion.expand(query)

        from src.query_expansion import Intent
        from src.generator import AnswerMode

        _mode = (
            AnswerMode.COMPARE
            if expanded.intent == Intent.COMPARISON
            else AnswerMode.VERBATIM
        )

        collector.query_expansion_detail = {
            "intent": expanded.intent.value,
            "mode": _mode.value,
            "sparte_hints": list(expanded.sparte_hints or []),
            "paraphrases": list(expanded.paraphrases or []),
            "domain_terms": list(expanded.domain_terms or []),
            "section_types": list(expanded.section_types or []),
            "confidence": expanded.confidence_score,
            "chain_of_thought": list(expanded.chain_of_thought or []),
            "step": _stage_step(collector, _steps_before_qe),
        }

        if expanded.intent == Intent.OUT_OF_SCOPE:
            return KcspAnswer(
                answer_markdown=_ABSTAIN_ANSWER,
                abstained=True,
                cited_sources=[],
                retrieved_doc_ids=[],
                audit_metadata={"critic_verdict": "ABSTAIN", "critic_reasoning": [],
                                "critic_cot": [], "critic_confidence": None,
                                "abstain_reason": "out_of_scope", "retried": False,
                                "used_ensemble": False, "early_abstain": True,
                                "generator_confidence": None, "generator_cot": [],
                                "selector_confidence": None, "selector_cot": None,
                                "pruning_detail": None, "selected_chunks": []},
            )

        # ── Stage 2: Retrieval ────────────────────────────────────────────────
        if stage_callback:
            stage_callback("retrieval")

        from src.doc_filter import (
            CompositeDocFilter, ProductDetectorAdapter, RareTagMatcherAdapter, _detect_tarif,
        )

        mode = _mode
        queries = [expanded.normalized_query] + list(expanded.paraphrases or [])

        doc_filter = None
        detected_tarif = None
        if (
            rag._documents_df is not None
            and rag._sections_df is not None
            and rag._subsections_df is not None
        ):
            sparte_hints = list(expanded.sparte_hints or [])
            if sparte_hints:
                tarif_names = list(
                    rag._documents_df[rag._documents_df["sparte"].isin(sparte_hints)]["tarif"]
                    .dropna().unique()
                )
            else:
                tarif_names = list(rag._documents_df["tarif"].dropna().unique())
            detected_tarif = _detect_tarif(expanded.normalized_query, tarif_names)
            adapters = [
                ProductDetectorAdapter(rag._documents_df, tarif=detected_tarif),
                RareTagMatcherAdapter(rag._sections_df, rag._subsections_df),
            ]
            doc_filter = CompositeDocFilter(adapters)

        collector.query_expansion_detail["detected_tarif"] = detected_tarif
        collector.query_expansion_detail["doc_filter_active"] = doc_filter is not None

        results = rag._retriever.retrieve_multi(
            queries=queries,
            top_k=rag._top_k,
            section_types=expanded.section_types or None,
            doc_filter=doc_filter,
            query_obj=expanded,
        )

        if not results:
            return KcspAnswer(
                answer_markdown=_ABSTAIN_ANSWER,
                abstained=True,
                cited_sources=[],
                retrieved_doc_ids=[],
                audit_metadata={"critic_verdict": "ABSTAIN", "critic_reasoning": [],
                                "critic_cot": [], "critic_confidence": None,
                                "abstain_reason": "empty_retrieval", "retried": False,
                                "used_ensemble": False, "early_abstain": True,
                                "generator_confidence": None, "generator_cot": [],
                                "selector_confidence": None, "selector_cot": None,
                                "pruning_detail": None,
                                "selected_chunks": [],
                                "detected_tarif": detected_tarif},
            )

        from src.schemas import SectionRecord
        sections: list[SectionRecord] = [
            SectionRecord(
                section_id=r.section_id,
                heading=r.heading,
                markdown=r.markdown,
                breadcrumb=r.breadcrumb,
            )
            for r in results
        ]

        selected_chunks = [
            {
                "chunk_id": str(r.section_id),
                "heading": r.heading,
                "breadcrumb": r.breadcrumb,
                "markdown": r.markdown,
                "score": float(r.score),
                "section_id": r.section_id,
            }
            for r in results
        ]

        # ── Stage 3: Generation ───────────────────────────────────────────────
        if stage_callback:
            stage_callback("generation")

        _steps_before_gen = len(collector.steps)
        generated = rag._generator.generate(expanded.normalized_query, sections, mode=mode)
        _gen_step = _stage_step(collector, _steps_before_gen)

        # ── Stage 4: Critic ───────────────────────────────────────────────────
        critic_result = None
        if mode != AnswerMode.COMPARE:
            if stage_callback:
                stage_callback("critic")

            from src.critic import run_critic, CriticVerdict

            def _generate_fn() -> str:
                return rag._generator.generate(
                    expanded.normalized_query, sections, mode=mode
                ).answer

            _steps_before_critic = len(collector.steps)
            critic_result = run_critic(
                query=expanded.normalized_query,
                answer=generated.answer,
                sections=sections,
                critic=rag._critic,
                generate_fn=_generate_fn,
                ensemble_critic=rag._ensemble_critic,
                enable_ensemble=rag._enable_ensemble,
            )
            _critic_step = _stage_step(collector, _steps_before_critic)

        from src.critic import CriticVerdict
        if critic_result is not None and critic_result.verdict == CriticVerdict.ABSTAIN:
            return KcspAnswer(
                answer_markdown=_ABSTAIN_ANSWER,
                abstained=True,
                cited_sources=[r.section_id for r in results],
                retrieved_doc_ids=[r.section_id for r in results],
                audit_metadata={
                    "critic_verdict": "ABSTAIN",
                    "critic_reasoning": [critic_result.reason] if critic_result.reason else [],
                    "critic_cot": list(critic_result.chain_of_thought or []),
                    "critic_confidence": critic_result.confidence,
                    "abstain_reason": "critic_abstain",
                    "retried": critic_result.retried,
                    "used_ensemble": critic_result.used_ensemble,
                    "early_abstain": False,
                    "generator_confidence": None,
                    "generator_cot": [],
                    "selector_confidence": None,
                    "selector_cot": None,
                    "pruning_detail": None,
                    "selected_chunks": selected_chunks,
                    "detected_tarif": detected_tarif,
                    "generator_step": _gen_step,
                    "critic_step": _critic_step if "_critic_step" in dir() else None,
                },
            )

        # ── Cross-sell ────────────────────────────────────────────────────────
        from src.ragassistant import _CROSS_SELL_MAP
        answer_text = (
            critic_result.answer
            if critic_result is not None and critic_result.answer is not None
            else generated.answer
        )
        if rag._enable_cross_sell and expanded.primary_sparte:
            hints = _CROSS_SELL_MAP.get(expanded.primary_sparte, [])
            if hints:
                hint_str = ", ".join(hints)
                answer_text += f"\n\n---\n**Ergänzende Produkte:** {hint_str}"

        audit_metadata: dict[str, Any] = {
            "critic_verdict": critic_result.verdict.value if critic_result else "PASS",
            "critic_reasoning": [critic_result.reason] if critic_result and critic_result.reason else [],
            "critic_cot": list(critic_result.chain_of_thought or []) if critic_result else [],
            "critic_confidence": critic_result.confidence if critic_result else None,
            "abstain_reason": None,
            "retried": critic_result.retried if critic_result else False,
            "used_ensemble": critic_result.used_ensemble if critic_result else False,
            "early_abstain": False,
            "generator_confidence": None,
            "generator_cot": [],
            "selector_confidence": None,
            "selector_cot": None,
            "pruning_detail": None,
            "selected_chunks": selected_chunks,
            "detected_tarif": detected_tarif,
            "generator_step": _gen_step,
            "critic_step": _critic_step if critic_result is not None else None,
        }

        return KcspAnswer(
            answer_markdown=answer_text,
            abstained=False,
            cited_sources=generated.sources,
            retrieved_doc_ids=[r.section_id for r in results],
            used_brute_force=False,
            audit_metadata=audit_metadata,
        )
