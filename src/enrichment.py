"""
Enrichment: LLM batch — generate title, description, questions, topic_tags per
retrieval unit (leaf section / subsection).

Provider: OpenRouter (OpenAI-compatible API) + instructor (pydantic-validated
structured output, auto-retry on malformed).

Design decisions (grill session 2026-06-20):
  - Output fields in GERMAN (coherent with QueryExpansion->DE + BGE-M3 space,
    no legal-term translation drift). System prompt in ENGLISH (LLM reasons
    better; instructs German output).
  - Core-4 fields only — every field has a consumer:
      title, description, questions -> retrieval embedding
      topic_tags                    -> Rare-tag Matcher (exact, must be DE)
  - Whitelist: these fields NEVER reach the generator / user. Retrieval-only.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.llm_providers import openrouter_client
from src.model_registry import REGISTRY

DEFAULT_MODEL = REGISTRY["enrichment"]


class SectionDetails(BaseModel):
    """Core-4 enrichment. All text fields in German; domain terms verbatim."""

    title: str = Field(
        ...,
        description=(
            "Concise, informative German section title (max 10 words). "
            "Unique and specific, e.g. 'Ausschlüsse beim Kaskoschutz', not just "
            "'Ausschlüsse'."
        ),
    )
    description: str = Field(
        ...,
        description="Short German summary of the section's content (1-2 sentences).",
    )
    questions: list[str] = Field(
        default_factory=list,
        description=(
            "5 to 10 real-world customer questions this section answers, in "
            "natural German, the way an ERGO agent or client would actually "
            "phrase them. Cover common cases and edge cases."
        ),
    )
    topic_tags: list[str] = Field(
        default_factory=list,
        description=(
            "Rare, highly specific German domain terms / product names found in "
            "the text (e.g. 'SF-Klasse', 'Glasbruch', 'Tierbiss', 'Fahrerschutz', "
            "'GAP', 'Neuwertentschädigung'). Keep them VERBATIM in German for "
            "exact rare-tag matching. Ignore generic terms (e.g. 'Schaden', "
            "'Kosten', 'Versicherung'). Empty list if none."
        ),
    )
    section_types: list[str] = Field(
        default_factory=list,
        description=(
            "Multi-label classification using the enum-16 schema. "
            "Choose ALL that apply from: INSURER_ID, PRODUCT_STRUCTURE, RISK_OBJECT, "
            "WHAT_IS_INSURED, EXCLUSIONS, LIMITS_COMPENSATION, CLAIMS_SETTLEMENT, "
            "INSURED_PERSONS, WHERE_COVERED, OBLIGATIONS, PAYMENT, CONTRACT_FORMATION, "
            "TERM_CANCELLATION, PRICING_DISCOUNT, COMPLAINTS_LAW, SPECIAL_PROVISIONS. "
            "Empty list if none apply (keyword labels used as fallback)."
        ),
    )


_SYSTEM_PROMPT = """\
You are an expert analyst of ERGO German P&C insurance conditions (Bedingungen) \
for the lines Kfz, Hausrat, Glas and Schmuck.

Your task: analyse one section of a Bedingungen document and produce structured \
metadata that lets an ERGO agent find this content through a search engine. The \
agent does not want to read the full document — they want the exact paragraph \
answering a concrete question (e.g. "Was zahlt die Teilkasko bei Glasbruch?", \
"Wie hoch ist die Selbstbeteiligung?").

GUIDELINES
- Precise title: unique and informative (e.g. "Obliegenheiten nach dem \
Schadenfall", not "Obliegenheiten").
- Questions: phrase them as real ERGO agents / customers would ask, covering \
routine cases and edge cases (Ausschlüsse, Wartezeiten, Auslandsschutz).
- Rare terms: extract only highly specific domain / product terms verbatim. Skip \
generic words.

IMPORTANT: Every output field MUST be written in GERMAN (proper nouns and domain \
terms stay verbatim). Do not translate German legal terminology. Be precise and \
practical.
"""


def _build_prompt(section: dict[str, Any]) -> str:
    return (
        f"Sparte: {section.get('sparte', '')}\n"
        f"Tarif: {section.get('tarif', '')}\n"
        f"Breadcrumb: {section.get('breadcrumb', '')}\n"
        f"Überschrift: {section.get('heading', '')}\n\n"
        f"Inhalt (Auszug, max 2500 Zeichen):\n"
        f"{str(section.get('markdown', ''))[:2500]}"
    )


def enrich_section(
    section: dict[str, Any],
    client: Optional[Any] = None,
    model: str = DEFAULT_MODEL,
) -> SectionDetails:
    c = client if client is not None else openrouter_client()
    return c.chat.completions.create(
        model=model,
        temperature=0.0,
        response_model=SectionDetails,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(section)},
        ],
    )


if __name__ == "__main__":
    # Smoke test — one section, prints structured output. Real $ (LLM call).
    sample = {
        "sparte": "Kfz",
        "tarif": "Spezial",
        "breadcrumb": "Kfz > Spezial > A.1",
        "heading": "Für welche Fahrzeugarten gelten diese Bedingungen?",
        "markdown": (
            "Diese Bedingungen gelten für Kraftfahrzeuge und Anhänger. "
            "Für die Versicherungen des Kfz-Handels und -Handwerks gelten nur "
            "die Bestimmungen der Abschnitte B.1, 2 und 4."
        ),
    }
    r = enrich_section(sample)
    print("title:      ", r.title)
    print("description:", r.description)
    print("topic_tags: ", r.topic_tags)
    print("questions:")
    for q in r.questions:
        print("  -", q)
