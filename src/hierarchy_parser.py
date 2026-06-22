"""
Generyczny parser hierarchii Bedingungen z deklaratywną konfiguracją per Sparte.
Musi działać PO md_sanitizer (ligatury w nagłówkach rozbijają regex sekcji).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.constants import SECTION_TYPES_FULL as ENUM_16
from src.md_sanitizer import sanitize

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Section:
    doc_id: str
    section_id: int
    sparte: str
    tarif: str
    section_code: str
    section_types: list[str]
    topic_tags: list[str]
    heading: str
    markdown: str
    breadcrumb: str
    confidence_score: float = 1.0
    level: int = 1
    parent_section_id: Optional[int] = None


@dataclass
class Document:
    doc_id: str
    sparte: str
    tarif: str
    numbering_scheme: str
    related_sparte: Optional[str]
    source_file: str
    sections: list[Section] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Document catalog: filename stem → (sparte, tarif, related_sparte, scheme)
# ---------------------------------------------------------------------------

DOCUMENT_CATALOG: dict[str, tuple[str, str, Optional[str], str]] = {
    "50064516_Bedingungen_AKB_Spezial_06_2025_final":
        ("Kfz", "Spezial", None, "GDV_AKB_LETTERS"),
    "50064517_Bedingungen_Kfz_06_2025_Final":
        ("Kfz", "Standard", None, "GDV_AKB_LETTERS"),
    "50078174_Bedingungen_Hausrat_Smart_11.23_ERGO-final":
        ("Hausrat", "Smart", None, "KT_NUMBERS"),
    "50078175_Bedingungen_Hausrat_Best 11.23_ERGO-final":
        ("Hausrat", "Best", None, "KT_NUMBERS"),
    "50078196_Bedingungen_Hausrat_Best_Naturgefahren 11.23_ERGO-f":
        ("Hausrat", "Best+Naturgefahren", None, "KT_NUMBERS"),
    "50078197_Bedingungen_Hausrat_Best_Fahrraddiebstahl 11.23_ERG":
        ("Hausrat", "Best+Fahrraddiebstahl", None, "KT_NUMBERS"),
    "50078198_Bedingungen_Hausrat_Glasversicherung 11.23_ERGO-fin":
        ("Glas", "KT2021GLHR", "Hausrat", "KT_NUMBERS"),
    "ERGO-Schmucksachen-KT-Bedingungen-11.23-50070699-nur PDF_fin":
        ("Schmuck", "KT Schmuck", "Hausrat", "KT_NUMBERS"),
}

# Per-scheme top-level section pattern: group(1)=code, group(2)=heading text
_SECTION_PATTERNS: dict[str, re.Pattern] = {
    "GDV_AKB_LETTERS": re.compile(r'^## ([A-N])\s+(.+)$', re.MULTILINE),
    "KT_NUMBERS":       re.compile(r'^## (\d+)\.\s+(.+)$', re.MULTILINE),
}

# Per-scheme L2 sub-section pattern: group(1)=sub_code, group(2)=heading text
# KT_NUMBERS:       ## 4.1 Feuer  /  ## 4.1.1 Brand
# GDV_AKB_LETTERS:  ## E.5 Wie regulieren wir?
_L2_PATTERNS: dict[str, re.Pattern] = {
    "GDV_AKB_LETTERS": re.compile(r'^## ([A-N]\.\d+)\s+(.+)$', re.MULTILINE),
    "KT_NUMBERS":       re.compile(r'^## (\d+\.\d+(?:\.\d+)?)\s+(.+)$', re.MULTILINE),
}

# ---------------------------------------------------------------------------
# Section type keyword rules (checked against heading + sub-headings text)
# ---------------------------------------------------------------------------

_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("INSURER_ID",         ["Wer sind wir", "Versicherer", "ERGO Versicherung AG", "Informationen zum Versicherer"]),
    ("PRODUCT_STRUCTURE",  ["Versicherungsbedingungen", "Inhalt", "Allgemeine Bedingungen", "Kundeninformation"]),
    ("RISK_OBJECT",        ["Welche Sachen", "Fahrzeugarten", "Verglasung", "Versicherte Risiken", "Risiken", "Gegenstände"]),
    ("WHAT_IS_INSURED",    ["versichert", "Schutz", "Leistung", "Deckung"]),
    ("EXCLUSIONS",         ["nicht versichert", "nicht entschädigt", "ausgeschlossen", "keine Entschädigung", "Wogegen", "Ausschluss"]),
    ("LIMITS_COMPENSATION",["Entschädigungsgrenze", "Versicherungssumme", "Kosten", "Unterversicherung", "überversichert", "Entschädigungsgrenzen"]),
    ("CLAIMS_SETTLEMENT",  ["Entschädigung erhalte", "Schadensfall", "regulier", "Sachverständigen", "aufgefunden", "Entschädigung erhalte"]),
    ("INSURED_PERSONS",    ["Wer ist versichert", "mitversichert", "Personen", "Halter", "Fahrer", "Beifahrer"]),
    ("WHERE_COVERED",      ["Wo ist", "Wo sind", "Länder", "Gebiet", "örtlich"]),
    ("OBLIGATIONS",        ["Obliegenheit", "Mitwirkungs", "Anzeigepflicht", "Sicherheitsvorschrift", "Gefahrerhöhung"]),
    ("PAYMENT",            ["Beitragszahlung", "Zahlungsintervall", "SEPA", "Lastschrift", "Folgebeitrag"]),
    ("CONTRACT_FORMATION", ["Zustandekommen", "Vertragsabschluss", "Widerruf", "Lebenssituation", "veräußere"]),
    ("TERM_CANCELLATION",  ["Laufzeit", "Kündigung", "beenden", "läuft", "Ablauf", "verjähren", "Versicherungsjahr"]),
    ("PRICING_DISCOUNT",   ["Schadenfreiheit", "Rabatt", "Beitragsberechnung", "Regionalklasse", "Typklasse", "SF-Klasse"]),
    ("COMPLAINTS_LAW",     ["Beschwerde", "Ombudsmann", "Recht gilt", "zuständig", "BaFin", "Schlichtung"]),
    ("SPECIAL_PROVISIONS", ["Mehrfachversicherung", "Ruheversicherung", "Außerbetriebsetzung", "Saison", "fremde Rechnung", "Ausfuhr"]),
]


_PREAMBLE_LIST_RE = re.compile(r'^\s*[-\d]', re.MULTILINE)
_PREAMBLE_SENT_RE = re.compile(r'.{30,}[.!?]')
_FREETEXT_HEAD_RE = re.compile(r'^## (.+)$', re.MULTILINE)
_FREETEXT_MARKER_RE = re.compile(r'Anhang|Sonderbedingungen|Besondere Bedingungen', re.IGNORECASE)


def _is_substantive_preamble(body: str) -> bool:
    s = body.strip()
    if not s:
        return False
    return (len(s) >= 200
            or bool(_PREAMBLE_LIST_RE.search(s))
            or bool(_PREAMBLE_SENT_RE.search(s)))


def _collect_headings(markdown: str) -> str:
    """Return all ## heading lines from markdown joined for keyword search."""
    return " ".join(re.findall(r'^## (.+)$', markdown, re.MULTILINE))


def _assign_types(heading: str, markdown: str) -> list[str]:
    text = (heading + " " + _collect_headings(markdown)).lower()
    types = [t for t, kws in _TYPE_RULES if any(kw.lower() in text for kw in kws)]
    return types or ["SPECIAL_PROVISIONS"]


def _parse_l2(parent: Section, l2_pattern: re.Pattern, sid_start: int) -> list[Section]:
    """Extract L2 sub-sections from a parent L1 section's markdown."""
    matches = list(l2_pattern.finditer(parent.markdown))
    if not matches:
        return []

    subsections: list[Section] = []
    for i, m in enumerate(matches):
        sub_code = m.group(1)
        sub_heading = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(parent.markdown)
        md = parent.markdown[start:end].strip()

        subsections.append(Section(
            doc_id=parent.doc_id,
            section_id=sid_start + i,
            sparte=parent.sparte,
            tarif=parent.tarif,
            section_code=sub_code,
            section_types=_assign_types(sub_heading, md),
            topic_tags=[],
            heading=sub_heading,
            markdown=md,
            breadcrumb=f"{parent.breadcrumb} > §{sub_code} {sub_heading}",
            confidence_score=1.0,
            level=2,
            parent_section_id=parent.section_id,
        ))
    return subsections


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_document(file_path: Path, _id_start: int = 1) -> Document:
    path = Path(file_path)
    doc_id = path.stem

    entry = DOCUMENT_CATALOG.get(doc_id)
    if entry is None:
        raise ValueError(f"Unknown document stem: {doc_id!r}. Add to DOCUMENT_CATALOG.")

    sparte, tarif, related_sparte, scheme = entry
    pattern = _SECTION_PATTERNS[scheme]

    raw = path.read_text(encoding="utf-8")
    text = sanitize(raw)

    matches = list(pattern.finditer(text))
    sections: list[Section] = []
    sid = _id_start

    # Preamble: everything before first top-level section
    if matches:
        preamble_md = text[: matches[0].start()].strip()
        if preamble_md:
            preamble_types = _assign_types(
                "Versicherungsbedingungen Inhalt Allgemeine Bedingungen Versicherer", preamble_md
            )
            sections.append(Section(
                doc_id=doc_id, section_id=sid, sparte=sparte, tarif=tarif,
                section_code="0", section_types=preamble_types, topic_tags=[],
                heading="Einleitung",
                markdown=preamble_md,
                breadcrumb=f"{sparte} > {tarif} > Einleitung",
                confidence_score=1.0,
            ))
            sid += 1

    l2_pattern = _L2_PATTERNS.get(scheme)

    # Top-level sections
    for i, m in enumerate(matches):
        code = m.group(1)
        heading = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        md = text[start:end].strip()

        l1 = Section(
            doc_id=doc_id, section_id=sid, sparte=sparte, tarif=tarif,
            section_code=code, section_types=_assign_types(heading, md), topic_tags=[],
            heading=heading,
            markdown=md,
            breadcrumb=f"{sparte} > {tarif} > §{code} {heading}",
            confidence_score=1.0,
            level=1,
            parent_section_id=None,
        )
        sections.append(l1)
        sid += 1

        if l2_pattern:
            l2_in_l1 = list(l2_pattern.finditer(md))

            # ── Rule 1: L1 preamble → {code}.0 section ───────────────────────
            if l2_in_l1:
                pre_raw = md[:l2_in_l1[0].start()]
                pre_body = re.sub(r'^## [^\n]+\n?', '', pre_raw, count=1).strip()
                if _is_substantive_preamble(pre_body):
                    pre_code = f"{code}.0"
                    sections.append(Section(
                        doc_id=doc_id, section_id=sid, sparte=sparte, tarif=tarif,
                        section_code=pre_code,
                        section_types=_assign_types(heading, pre_body),
                        topic_tags=[],
                        heading="Vorbemerkung",
                        markdown=pre_body,
                        breadcrumb=f"{l1.breadcrumb} > §{pre_code} Vorbemerkung",
                        confidence_score=1.0,
                        level=2,
                        parent_section_id=l1.section_id,
                    ))
                    sid += 1

            l2_sections = _parse_l2(l1, l2_pattern, sid_start=sid)
            sections.extend(l2_sections)
            sid += len(l2_sections)

            # ── Rule 2: tail free-text sections (marker-only) ─────────────
            # Only emit headings whose text OR first 200 chars of body
            # contain the attachment-marker regex.
            if l2_in_l1:
                tail = md[l2_in_l1[-1].start():]
                l2_pos_in_tail = {m.start() for m in l2_pattern.finditer(tail)}
                ft_heads = [h for h in _FREETEXT_HEAD_RE.finditer(tail)
                            if h.start() not in l2_pos_in_tail]
                ft_count = 0
                for j, ft_m in enumerate(ft_heads):
                    ft_heading = ft_m.group(1).strip()
                    ft_start = ft_m.start()
                    ft_end = ft_heads[j + 1].start() if j + 1 < len(ft_heads) else len(tail)
                    ft_md = tail[ft_start:ft_end].strip()
                    if not ft_md:
                        continue
                    # Gate: emit only if heading or first 200 chars of body has marker
                    body_after = ft_md[len(f"## {ft_heading}"):].strip()[:200]
                    if not (_FREETEXT_MARKER_RE.search(ft_heading) or
                            _FREETEXT_MARKER_RE.search(body_after)):
                        continue
                    ft_count += 1
                    ft_code = f"{code}-FT{ft_count}"
                    sections.append(Section(
                        doc_id=doc_id, section_id=sid, sparte=sparte, tarif=tarif,
                        section_code=ft_code,
                        section_types=_assign_types(ft_heading, ft_md),
                        topic_tags=[],
                        heading=ft_heading,
                        markdown=ft_md,
                        breadcrumb=f"{l1.breadcrumb} > §{ft_code} {ft_heading}",
                        confidence_score=1.0,
                        level=2,
                        parent_section_id=l1.section_id,
                    ))
                    sid += 1

    return Document(
        doc_id=doc_id, sparte=sparte, tarif=tarif,
        numbering_scheme=scheme, related_sparte=related_sparte,
        source_file=str(path), sections=sections,
    )


def parse_all(directory: Path) -> list[Document]:
    directory = Path(directory)
    docs: list[Document] = []
    sid = 1
    for stem, _ in DOCUMENT_CATALOG.items():
        candidates = list(directory.glob("*.md"))
        match = next((p for p in candidates if p.stem == stem), None)
        if match:
            doc = parse_document(match, _id_start=sid)
            sid += len(doc.sections)
            docs.append(doc)
    return docs
