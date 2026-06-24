"""KcspRunner: adapter between webapp chat.py and RAGAssistant.ask()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.tracing import TraceCollector, get_active_collector, set_active_collector


@dataclass
class KcspAnswer:
    """Mimics dkv's answer object interface expected by chat.py run_rag_in_background."""

    answer_markdown: str
    abstained: bool
    cited_sources: list[int]
    retrieved_doc_ids: list[int]
    used_brute_force: bool = False
    audit_metadata: dict = field(default_factory=dict)


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
        """Run RAG pipeline with stage callbacks."""
        if stage_callback:
            stage_callback("query_expansion")

        # Ensure TraceCollector is active
        collector = get_active_collector()
        if collector is None:
            collector = TraceCollector()
            set_active_collector(collector)

        final_answer = self._rag.ask(query)

        # Emit remaining stages after ask() (retrospective — answer already done)
        # These update current_stage in DB via chat.py's _update_stage
        if stage_callback:
            stage_callback("retrieval")
            stage_callback("generation")
            stage_callback("critic")

        # Map FinalAnswer → KcspAnswer
        sources: list[int] = final_answer.sources or []
        breadcrumbs: list[str] = final_answer.breadcrumbs or []

        audit_metadata: dict[str, Any] = {
            "critic_verdict": "ABSTAIN" if final_answer.abstained else "PASS",
            "critic_reasoning": None,
            "critic_cot": None,
            "critic_confidence": None,
            "abstain_reason": None,
            "retried": None,
            "used_ensemble": None,
            "early_abstain": None,
            "generator_confidence": None,
            "generator_cot": None,
            "selector_confidence": None,
            "selector_cot": None,
            "pruning_detail": None,
            "selected_chunks": [
                {
                    "chunk_id": str(sid),
                    "heading": bc,
                    "breadcrumb": bc,
                    "markdown": "",
                    "score": 0.0,
                    "section_id": sid,
                }
                for sid, bc in zip(sources, breadcrumbs)
            ],
        }

        return KcspAnswer(
            answer_markdown=final_answer.answer,
            abstained=final_answer.abstained,
            cited_sources=sources,
            retrieved_doc_ids=sources,
            used_brute_force=False,
            audit_metadata=audit_metadata,
        )
