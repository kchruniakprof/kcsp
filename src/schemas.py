"""Shared data schemas for the KCSP retrieval pipeline."""
from typing import TypedDict


class SectionRecord(TypedDict):
    """Typed boundary for a retrieved section passed between pipeline modules.

    All three consumers (Generator, Critic, RAGAssistant) receive this type.
    markdown must be the full verbatim text (r.markdown), never the pruned view.
    """

    section_id: int
    heading: str
    markdown: str
    breadcrumb: str
