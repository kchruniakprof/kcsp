"""Tests for src/enrichment.py — SectionDetails + enrich_section (mocked LLM)."""
import pytest
from unittest.mock import MagicMock

from src.enrichment import SectionDetails, enrich_section


# ── SectionDetails ────────────────────────────────────────────────────────────

def test_section_details_fields():
    sd = SectionDetails(
        title="Versicherte Risiken",
        description="Welche Risiken sind im Kfz-Tarif Spezial versichert.",
        questions=["Was ist versichert?", "Was gilt bei Diebstahl?"],
        topic_tags=["Deckungsumfang", "Kfz"],
    )
    assert sd.title == "Versicherte Risiken"
    assert isinstance(sd.topic_tags, list)
    assert len(sd.questions) == 2


def test_section_details_empty_topic_tags_ok():
    sd = SectionDetails(title="T", description="D", questions=["Q?"], topic_tags=[])
    assert sd.topic_tags == []


# ── enrich_section with mocked client ────────────────────────────────────────

_FAKE_SECTION = {
    "doc_id": "kfz_spezial",
    "heading": "A Versicherte Risiken",
    "markdown": "Versichert ist das Fahrzeug gegen folgende Schäden...",
    "breadcrumb": "Kfz > Spezial > §A",
    "sparte": "Kfz",
    "tarif": "Spezial",
}

_FAKE_DETAILS = SectionDetails(
    title="Versicherte Risiken Kfz",
    description="Deckungsumfang der Kfz-Spezialversicherung.",
    questions=["Was ist versichert?", "Was gilt bei Brand?"],
    topic_tags=["Deckungsumfang", "Kfz", "Risiken"],
)


def _make_mock_client(result: SectionDetails) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = result
    return client


def test_enrich_section_returns_section_details():
    client = _make_mock_client(_FAKE_DETAILS)
    result = enrich_section(_FAKE_SECTION, client=client)
    assert isinstance(result, SectionDetails)


def test_enrich_section_title_nonempty():
    client = _make_mock_client(_FAKE_DETAILS)
    result = enrich_section(_FAKE_SECTION, client=client)
    assert result.title.strip() != ""


def test_enrich_section_description_nonempty():
    client = _make_mock_client(_FAKE_DETAILS)
    result = enrich_section(_FAKE_SECTION, client=client)
    assert result.description.strip() != ""


def test_enrich_section_has_questions():
    client = _make_mock_client(_FAKE_DETAILS)
    result = enrich_section(_FAKE_SECTION, client=client)
    assert len(result.questions) >= 1


def test_enrich_section_uses_temperature_zero():
    client = _make_mock_client(_FAKE_DETAILS)
    enrich_section(_FAKE_SECTION, client=client)
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs.get("temperature", 999) == 0.0
