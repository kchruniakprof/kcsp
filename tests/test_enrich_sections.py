"""Tests for enrich_sections — TDD B1 (mocked LLM, no real API calls)."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd

from src.enrichment import SectionDetails

_FAKE_DETAILS = SectionDetails(
    title="Versicherte Risiken",
    description="Deckungsumfang der Kfz-Spezialversicherung.",
    questions=["Was ist versichert?", "Was gilt bei Brand?", "Gilt das auch im Ausland?",
               "Was ist bei Diebstahl?", "Wie läuft die Entschädigung ab?"],
    topic_tags=["Deckungsumfang", "Kfz"],
)

_SEC_ROWS = [
    {"section_id": 1, "doc_id": "kfz_spezial", "heading": "§A", "markdown": "Text A",
     "breadcrumb": "Kfz > §A", "sparte": "Kfz", "tarif": "Spezial",
     "is_retrieval_unit": True,
     "title": None, "description": None, "questions": None, "topic_tags": None},
    {"section_id": 2, "doc_id": "kfz_spezial", "heading": "§B Parent", "markdown": "Text B",
     "breadcrumb": "Kfz > §B", "sparte": "Kfz", "tarif": "Spezial",
     "is_retrieval_unit": False,  # L1-parent — must be skipped
     "title": None, "description": None, "questions": None, "topic_tags": None},
    {"section_id": 3, "doc_id": "kfz_spezial", "heading": "§C", "markdown": "Text C",
     "breadcrumb": "Kfz > §C", "sparte": "Kfz", "tarif": "Spezial",
     "is_retrieval_unit": True,
     "title": None, "description": None, "questions": None, "topic_tags": None},
]


def _make_sections_df():
    return pd.DataFrame(_SEC_ROWS)


# ── imports ───────────────────────────────────────────────────────────────────

def test_enrich_sections_importable():
    from src.enrich_sections import enrich_sections
    assert enrich_sections is not None


# ── iterates only retrieval units ─────────────────────────────────────────────

def test_skips_l1_parents(tmp_path):
    """enrich_sections must call enrich_section only for is_retrieval_unit=True rows."""
    from src.enrich_sections import enrich_sections
    sections_df = _make_sections_df()
    calls = []

    def mock_enrich(section, client=None, model=None):
        calls.append(section["section_id"])
        return _FAKE_DETAILS

    with patch("src.enrich_sections.enrich_section", side_effect=mock_enrich):
        enrich_sections(
            sections_df=sections_df,
            output_dir=tmp_path,
            client=MagicMock(),
            auto_yes=True,
        )

    assert 2 not in calls, "section_id=2 (is_retrieval_unit=False) must be skipped"
    assert 1 in calls
    assert 3 in calls


# ── core-4 fields written to output ──────────────────────────────────────────

def test_core4_fields_written_to_parquet(tmp_path):
    """After enrichment, title/description/questions/topic_tags must be non-null."""
    from src.enrich_sections import enrich_sections
    sections_df = _make_sections_df()

    with patch("src.enrich_sections.enrich_section", return_value=_FAKE_DETAILS):
        enrich_sections(
            sections_df=sections_df,
            output_dir=tmp_path,
            client=MagicMock(),
            auto_yes=True,
        )

    out = pd.read_parquet(tmp_path / "sections_enriched.parquet")
    retrieval_rows = out[out["is_retrieval_unit"] == True]
    assert retrieval_rows["title"].notna().all()
    assert retrieval_rows["description"].notna().all()
    assert retrieval_rows["questions"].notna().all()


# ── checkpoint skip-done ──────────────────────────────────────────────────────

def test_checkpoint_skip_already_enriched(tmp_path):
    """Sections listed in checkpoint must be skipped on re-run."""
    from src.enrich_sections import enrich_sections
    sections_df = _make_sections_df()
    checkpoint_file = tmp_path / "enrichment_checkpoint.json"
    checkpoint_file.write_text(json.dumps({"1": True}))  # section_id=1 done

    calls = []
    def mock_enrich(section, client=None, model=None):
        calls.append(section["section_id"])
        return _FAKE_DETAILS

    with patch("src.enrich_sections.enrich_section", side_effect=mock_enrich):
        enrich_sections(
            sections_df=sections_df,
            output_dir=tmp_path,
            client=MagicMock(),
            auto_yes=True,
        )

    assert 1 not in calls, "section_id=1 already in checkpoint — must be skipped"
    assert 3 in calls


def test_checkpoint_written_after_each_success(tmp_path):
    """Checkpoint file updated after each successful enrichment."""
    from src.enrich_sections import enrich_sections
    sections_df = _make_sections_df()

    with patch("src.enrich_sections.enrich_section", return_value=_FAKE_DETAILS):
        enrich_sections(
            sections_df=sections_df,
            output_dir=tmp_path,
            client=MagicMock(),
            auto_yes=True,
        )

    checkpoint_file = tmp_path / "enrichment_checkpoint.json"
    assert checkpoint_file.exists()
    data = json.loads(checkpoint_file.read_text())
    assert "1" in data and "3" in data
    assert "2" not in data  # L1-parent not enriched → not in checkpoint


# ── cost-gate ─────────────────────────────────────────────────────────────────

def test_cost_gate_aborts_on_no(tmp_path, monkeypatch):
    """Without --yes/auto_yes=True, prompts user; 'n' → aborts without calling LLM."""
    from src.enrich_sections import enrich_sections
    sections_df = _make_sections_df()
    monkeypatch.setattr("builtins.input", lambda _: "n")
    calls = []
    with patch("src.enrich_sections.enrich_section", side_effect=lambda *a, **kw: calls.append(1)):
        enrich_sections(
            sections_df=sections_df,
            output_dir=tmp_path,
            client=MagicMock(),
            auto_yes=False,
        )
    assert calls == [], "LLM must not be called when user says 'n'"


def test_auto_yes_skips_prompt(tmp_path, monkeypatch):
    """auto_yes=True must skip the cost-gate prompt entirely."""
    from src.enrich_sections import enrich_sections
    input_called = []
    monkeypatch.setattr("builtins.input", lambda _: input_called.append(1) or "n")
    with patch("src.enrich_sections.enrich_section", return_value=_FAKE_DETAILS):
        enrich_sections(
            sections_df=_make_sections_df(),
            output_dir=tmp_path,
            client=MagicMock(),
            auto_yes=True,
        )
    assert input_called == [], "input() must not be called when auto_yes=True"


# ── error handling ────────────────────────────────────────────────────────────

def test_failed_section_written_to_error_log(tmp_path):
    """Sections that exhaust max_retries go to enrichment_errors.json, not crash."""
    from src.enrich_sections import enrich_sections
    sections_df = pd.DataFrame([_SEC_ROWS[0]])  # only section_id=1

    def mock_enrich(section, client=None, model=None):
        raise RuntimeError("instructor max_retries exceeded")

    with patch("src.enrich_sections.enrich_section", side_effect=mock_enrich):
        enrich_sections(
            sections_df=sections_df,
            output_dir=tmp_path,
            client=MagicMock(),
            auto_yes=True,
        )

    error_file = tmp_path / "enrichment_errors.json"
    assert error_file.exists()
    errors = json.loads(error_file.read_text())
    assert any(e["section_id"] == 1 for e in errors)
