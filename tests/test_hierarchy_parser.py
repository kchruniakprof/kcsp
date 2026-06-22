"""Tests for hierarchy_parser — TDD cycle."""
import re
from pathlib import Path

import pytest

from src.hierarchy_parser import ENUM_16, Document, Section, parse_all, parse_document

CORPUS = Path("D:/_FUN/kcsp/v1/sources/output_md")

KFZ_SPEZIAL = CORPUS / "50064516_Bedingungen_AKB_Spezial_06_2025_final.md"
KFZ_STANDARD = CORPUS / "50064517_Bedingungen_Kfz_06_2025_Final.md"
HAUSRAT_SMART = CORPUS / "50078174_Bedingungen_Hausrat_Smart_11.23_ERGO-final.md"
HAUSRAT_BEST = CORPUS / "50078175_Bedingungen_Hausrat_Best 11.23_ERGO-final.md"
HAUSRAT_NATUR = CORPUS / "50078196_Bedingungen_Hausrat_Best_Naturgefahren 11.23_ERGO-f.md"
HAUSRAT_FAHRAD = CORPUS / "50078197_Bedingungen_Hausrat_Best_Fahrraddiebstahl 11.23_ERG.md"
GLAS = CORPUS / "50078198_Bedingungen_Hausrat_Glasversicherung 11.23_ERGO-fin.md"
SCHMUCK = CORPUS / "ERGO-Schmucksachen-KT-Bedingungen-11.23-50070699-nur PDF_fin.md"


# --- Document catalog & metadata ---

def test_parse_all_returns_8_documents():
    docs = parse_all(CORPUS)
    assert len(docs) == 8


def test_kfz_spezial_metadata():
    doc = parse_document(KFZ_SPEZIAL)
    assert doc.sparte == "Kfz"
    assert doc.tarif == "Spezial"
    assert doc.numbering_scheme == "GDV_AKB_LETTERS"
    assert doc.related_sparte is None


def test_kfz_standard_metadata():
    doc = parse_document(KFZ_STANDARD)
    assert doc.sparte == "Kfz"
    assert doc.tarif == "Standard"


def test_hausrat_smart_metadata():
    doc = parse_document(HAUSRAT_SMART)
    assert doc.sparte == "Hausrat"
    assert doc.tarif == "Smart"
    assert doc.related_sparte is None


def test_glas_metadata():
    doc = parse_document(GLAS)
    assert doc.sparte == "Glas"
    assert doc.related_sparte == "Hausrat"


def test_schmuck_metadata():
    doc = parse_document(SCHMUCK)
    assert doc.sparte == "Schmuck"
    assert doc.related_sparte == "Hausrat"


def test_doc_id_matches_filename_stem():
    doc = parse_document(KFZ_SPEZIAL)
    assert doc.doc_id == KFZ_SPEZIAL.stem


# --- Section codes: Kfz (A-N) ---

def test_kfz_has_all_14_akb_sections():
    doc = parse_document(KFZ_SPEZIAL)
    codes = {s.section_code for s in doc.sections}
    for letter in "ABCDEFGHIJKLMN":
        assert letter in codes, f"Missing Kfz AKB section {letter!r}"


def test_kfz_akb_section_codes_are_letters():
    doc = parse_document(KFZ_SPEZIAL)
    non_preamble = [s for s in doc.sections if s.section_code != "0"]
    for s in non_preamble:
        assert re.match(r'^[A-N]$', s.section_code), \
            f"Unexpected Kfz section code {s.section_code!r}"


# --- Section codes: Hausrat (1-30) ---

def test_hausrat_has_sections_1_through_30():
    doc = parse_document(HAUSRAT_SMART)
    codes = {s.section_code for s in doc.sections}
    for i in range(1, 31):
        assert str(i) in codes, f"Missing Hausrat section {i}"


def test_hausrat_section_codes_are_numbers():
    doc = parse_document(HAUSRAT_SMART)
    non_preamble = [s for s in doc.sections if s.section_code != "0"]
    for s in non_preamble:
        assert s.section_code.isdigit(), f"Non-numeric Hausrat code {s.section_code!r}"


# --- Section codes: Glas (1-16) ---

def test_glas_has_sections_1_through_16():
    doc = parse_document(GLAS)
    codes = {s.section_code for s in doc.sections}
    for i in range(1, 17):
        assert str(i) in codes, f"Missing Glas section {i}"


# --- Breadcrumb ---

def test_kfz_breadcrumb_format():
    doc = parse_document(KFZ_SPEZIAL)
    section_a = next(s for s in doc.sections if s.section_code == "A")
    assert section_a.breadcrumb.startswith("Kfz > Spezial > §A ")


def test_hausrat_breadcrumb_format():
    doc = parse_document(HAUSRAT_SMART)
    section_1 = next(s for s in doc.sections if s.section_code == "1")
    assert section_1.breadcrumb.startswith("Hausrat > Smart > §1 ")


def test_glas_breadcrumb_format():
    doc = parse_document(GLAS)
    section_1 = next(s for s in doc.sections if s.section_code == "1")
    assert section_1.breadcrumb.startswith("Glas > KT2021GLHR > §1 ")


# --- Markdown content ---

def test_all_sections_have_nonempty_markdown():
    docs = parse_all(CORPUS)
    for doc in docs:
        for s in doc.sections:
            assert s.markdown.strip(), \
                f"Empty markdown in {doc.doc_id} §{s.section_code}"


def test_kfz_section_c_markdown_contains_haftpflicht():
    doc = parse_document(KFZ_SPEZIAL)
    sec_c = next(s for s in doc.sections if s.section_code == "C")
    assert "Haftpflicht" in sec_c.markdown


# --- section_types (multi-label, enum-16) ---

def test_hausrat_section1_is_multilabel():
    doc = parse_document(HAUSRAT_SMART)
    sec1 = next(s for s in doc.sections if s.section_code == "1")
    assert len(sec1.section_types) >= 2, \
        f"Expected multi-label on §1, got {sec1.section_types}"


def test_all_16_section_types_present_in_corpus():
    docs = parse_all(CORPUS)
    found = set()
    for doc in docs:
        for s in doc.sections:
            found.update(s.section_types)
    for t in ENUM_16:
        assert t in found, f"section_type {t!r} not found anywhere in corpus"


def test_section_types_are_valid_enum_values():
    docs = parse_all(CORPUS)
    for doc in docs:
        for s in doc.sections:
            for t in s.section_types:
                assert t in ENUM_16, f"Unknown section_type {t!r} in {doc.doc_id} §{s.section_code}"


# --- topic_tags (empty at parse time) ---

def test_topic_tags_empty_at_parse_time():
    doc = parse_document(KFZ_SPEZIAL)
    for s in doc.sections:
        assert s.topic_tags == [], f"topic_tags should be empty before enrichment"


# --- confidence_score ---

def test_confidence_score_is_1():
    doc = parse_document(KFZ_SPEZIAL)
    for s in doc.sections:
        assert s.confidence_score == 1.0


# --- Sanitization applied before parsing ---

def test_no_ligature_artifacts_in_headings():
    docs = parse_all(CORPUS)
    for doc in docs:
        for s in doc.sections:
            assert not re.search(r'fl [a-zäöü]|fi [a-zäöü]', s.heading), \
                f"Ligature artifact in heading: {s.heading!r} ({doc.doc_id})"


def test_no_ligature_artifacts_in_breadcrumbs():
    docs = parse_all(CORPUS)
    for doc in docs:
        for s in doc.sections:
            assert not re.search(r'fl [a-zäöü]|fi [a-zäöü]', s.breadcrumb), \
                f"Ligature in breadcrumb: {s.breadcrumb!r}"


# --- Global uniqueness of section_id ---

def test_section_ids_globally_unique():
    docs = parse_all(CORPUS)
    all_ids = [s.section_id for doc in docs for s in doc.sections]
    assert len(all_ids) == len(set(all_ids)), "Duplicate section_ids across documents"


# --- C1: Preamble + FreeText sections ---

def test_c1_q7_preamble_1_0_exists():
    doc = parse_document(HAUSRAT_SMART)
    codes = {s.section_code for s in doc.sections}
    assert "1.0" in codes, "§1.0 preamble not found in Hausrat Smart"


def test_c1_q7_preamble_1_0_contains_wallboxen():
    doc = parse_document(HAUSRAT_SMART)
    sec = next(s for s in doc.sections if s.section_code == "1.0")
    assert "Wallboxen" in sec.markdown or "wallbox" in sec.markdown.lower()


def test_c1_q7_preamble_1_0_level_and_parent():
    doc = parse_document(HAUSRAT_SMART)
    sec_1 = next(s for s in doc.sections if s.section_code == "1")
    sec_1_0 = next(s for s in doc.sections if s.section_code == "1.0")
    assert sec_1_0.level == 2
    assert sec_1_0.parent_section_id == sec_1.section_id


def test_c1_q7_preamble_breadcrumb_vorbemerkung():
    doc = parse_document(HAUSRAT_SMART)
    sec_1_0 = next(s for s in doc.sections if s.section_code == "1.0")
    assert "Vorbemerkung" in sec_1_0.breadcrumb


def test_c1_q1_kfz_standard_safe_drive_accessible():
    doc = parse_document(KFZ_STANDARD)
    safe_drive = [s for s in doc.sections if "Safe Drive" in s.heading or "Safe Drive" in s.breadcrumb]
    assert safe_drive, "No section with 'Safe Drive' found in KFZ Standard after C1"


def test_c1_q1_safe_drive_section_is_level_2():
    doc = parse_document(KFZ_STANDARD)
    safe_drive = [s for s in doc.sections if "Safe Drive" in s.heading or "Safe Drive" in s.breadcrumb]
    assert any(s.level == 2 for s in safe_drive), "No level-2 section with 'Safe Drive'"


def test_c1_q2_ev_wechselpraemie_has_parent_section_e():
    """Regression guard: EV-Wechselprämie content already in E.5 (parent=§E)."""
    doc = parse_document(KFZ_STANDARD)
    sec_e = next(s for s in doc.sections if s.section_code == "E")
    ev_sects = [s for s in doc.sections if "lektrofahrzeug-Wechselpr" in s.markdown]
    assert ev_sects, "No section with EV-Wechselprämie content"
    assert any(s.parent_section_id == sec_e.section_id for s in ev_sects), \
        "No EV-Wechselprämie section with §E as parent"


def test_c1_preamble_sections_are_nonempty():
    docs = parse_all(CORPUS)
    preambles = [s for doc in docs for s in doc.sections if s.section_code.endswith(".0")]
    assert preambles, "No preamble sections found after C1"
    for s in preambles:
        assert s.markdown.strip(), f"Empty preamble markdown: {s.doc_id} §{s.section_code}"


# --- Denormalized fields ---

def test_sections_have_denormalized_sparte_and_tarif():
    doc = parse_document(KFZ_SPEZIAL)
    for s in doc.sections:
        assert s.sparte == "Kfz"
        assert s.tarif == "Spezial"
