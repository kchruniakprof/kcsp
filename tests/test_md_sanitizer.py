"""Tests for md_sanitizer — TDD cycle."""
import pytest
from src.md_sanitizer import sanitize


# --- Ligature: fl ---

def test_fl_ligature_inline():
    assert sanitize("Haftpfl icht") == "Haftpflicht"


def test_fl_ligature_start_of_word():
    assert sanitize("Pfl anzenschutz") == "Pflanzenschutz"


def test_fl_ligature_multiple():
    assert sanitize("Haftpfl icht und verpfl ichtet") == "Haftpflicht und verpflichtet"


def test_fl_ligature_in_header():
    assert sanitize("## C Kfz-Haftpfl icht – die Versicherung") == \
        "## C Kfz-Haftpflicht – die Versicherung"


# --- Ligature: fi ---

def test_fi_ligature_inline():
    assert sanitize("fi nden") == "finden"


def test_fi_ligature_mid_word():
    assert sanitize("vorläufi g") == "vorläufig"


def test_fi_ligature_ffi():
    assert sanitize("Sheffi eld") == "Sheffield"


def test_fi_ligature_multiple():
    assert sanitize("fi nden Sie fi nden") == "finden Sie finden"


# --- Mixed ---

def test_fl_and_fi_ligatures_together():
    assert sanitize(
        "Die Kfz-Haftpfl icht leistet. Nähere Angaben fi nden Sie in den Abschnitten."
    ) == "Die Kfz-Haftpflicht leistet. Nähere Angaben finden Sie in den Abschnitten."


def test_real_kfz_sentence():
    raw = "Nähere Angaben über Art, Umfang und Fälligkeit unserer Leistung fi nden Sie in den Abschnitten C bis E AKB Spezial."
    expected = "Nähere Angaben über Art, Umfang und Fälligkeit unserer Leistung finden Sie in den Abschnitten C bis E AKB Spezial."
    assert sanitize(raw) == expected


def test_real_auflieger_sentence():
    raw = "## Mitversicherung von Anhängern, Aufl iegern und abgeschleppten Fahrzeugen"
    expected = "## Mitversicherung von Anhängern, Aufliegern und abgeschleppten Fahrzeugen"
    assert sanitize(raw) == expected


# --- Image placeholders ---

def test_image_placeholder_removed():
    assert sanitize("<!-- image -->") == ""


def test_image_placeholder_removed_from_line():
    assert sanitize("text\n<!-- image -->\nmore text") == "text\n\nmore text"


# --- Whitespace normalization ---

def test_multiple_blank_lines_collapsed():
    assert sanitize("line1\n\n\n\nline2") == "line1\n\nline2"


def test_three_blank_lines_to_one():
    assert sanitize("a\n\n\nb") == "a\n\nb"


def test_trailing_whitespace_stripped():
    assert sanitize("line with trailing   ") == "line with trailing"


def test_trailing_whitespace_per_line():
    assert sanitize("line1   \nline2  ") == "line1\nline2"


# --- Preservation ---

def test_headers_preserved():
    assert sanitize("## §A Beginn des Versicherungsschutzes") == \
        "## §A Beginn des Versicherungsschutzes"


def test_bullets_preserved():
    text = "- Kfz-Haftpflicht\n- Kaskoversicherung"
    assert sanitize(text) == text


def test_checkmark_preserved():
    assert sanitize("✓ covered") == "✓ covered"


def test_no_ligature_unchanged():
    text = "Versicherungsschutz gilt auch für Anhänger."
    assert sanitize(text) == text


def test_uppercase_word_boundary_not_merged():
    # "fl " before uppercase = end of word + next word starts with capital → do NOT merge
    assert sanitize("Kfz Haftpfl ichtversicherung") == "Kfz Haftpflichtversicherung"


# --- Real document smoke test ---

def test_real_file_no_ligature_artifacts(tmp_path):
    """Sanitized output from real Kfz file must not contain known broken patterns."""
    import re
    from pathlib import Path

    kfz_file = Path("D:/_FUN/kcsp/v1/sources/output_md/50064516_Bedingungen_AKB_Spezial_06_2025_final.md")
    raw = kfz_file.read_text(encoding="utf-8")
    result = sanitize(raw)

    broken_patterns = [
        r"fl [a-zäöü]",   # fl ligature still broken
        r"fi [a-zäöü]",   # fi ligature still broken
    ]
    for pattern in broken_patterns:
        matches = re.findall(pattern, result)
        assert not matches, f"Ligature artifact still present: {matches[:3]}"


def test_real_file_headers_intact():
    """Kfz headers like '## C Kfz-Haftpflicht' must survive sanitization."""
    from pathlib import Path

    kfz_file = Path("D:/_FUN/kcsp/v1/sources/output_md/50064516_Bedingungen_AKB_Spezial_06_2025_final.md")
    raw = kfz_file.read_text(encoding="utf-8")
    result = sanitize(raw)

    assert "## C Kfz-Haftpflicht" in result
    assert "## B Zustandekommen des Versicherungsschutzes und vorläufiger Versicherungsschutz" in result
