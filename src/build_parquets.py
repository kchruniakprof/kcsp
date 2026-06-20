"""
Orchestrator: sanitizer → parser → emit:
  documents.parquet   — 8 dokumentów
  sections.parquet    — sekcje L1 (z is_retrieval_unit)
  subsections.parquet — subsekcje L2 (z is_retrieval_unit + parent_section_id)

Embedding wydzielone do build_embeddings.py (etap C).
Walidacja schematu po build.
"""
import re
from pathlib import Path

import pandas as pd

from src.hierarchy_parser import DOCUMENT_CATALOG, parse_all


def strip_noise(md: str) -> str:
    """Remove ToC, marketing footer, and large table blocks (>5 lines)."""
    m = re.search(r"^## Stichwortverzeichnis", md, re.MULTILINE)
    if m:
        md = md[:m.start()].rstrip()
    m = re.search(r"^## Wir sind immer", md, re.MULTILINE)
    if m:
        md = md[:m.start()].rstrip()
    lines = md.splitlines()
    cleaned, table_buf = [], []
    for line in lines:
        if line.strip().startswith("|"):
            table_buf.append(line)
        else:
            if table_buf:
                if len(table_buf) <= 5:
                    cleaned.extend(table_buf)
                table_buf = []
            cleaned.append(line)
    if table_buf and len(table_buf) <= 5:
        cleaned.extend(table_buf)
    return "\n".join(cleaned).strip()

def build(corpus_dir: Path, output_dir: Path) -> None:
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = parse_all(corpus_dir)
    if not docs:
        raise RuntimeError(f"No documents parsed from {corpus_dir}")

    # --- documents.parquet ---
    doc_rows = [
        {
            "doc_id": d.doc_id,
            "sparte": d.sparte,
            "tarif": d.tarif,
            "numbering_scheme": d.numbering_scheme,
            "related_sparte": d.related_sparte,  # None → NaN in parquet
            "source_file": d.source_file,
        }
        for d in docs
    ]
    docs_df = pd.DataFrame(doc_rows)
    docs_df.to_parquet(output_dir / "documents.parquet", index=False)

    # --- sections: collect rows ---
    sec_rows = [
        {
            "doc_id": s.doc_id,
            "section_id": s.section_id,
            "sparte": s.sparte,
            "tarif": s.tarif,
            "section_code": s.section_code,
            "section_types": s.section_types,
            "topic_tags": s.topic_tags,
            "heading": s.heading,
            "markdown": s.markdown,
            "breadcrumb": s.breadcrumb,
            "confidence_score": s.confidence_score,
            "level": s.level,
            "parent_section_id": s.parent_section_id,
        }
        for d in docs
        for s in d.sections
    ]

    # --- strip noise from stored markdown ---
    for row in sec_rows:
        row["markdown"] = strip_noise(row["markdown"])

    all_df = pd.DataFrame(sec_rows)

    # --- is_retrieval_unit: L2 always True; L1 True only if it has no children ---
    l2_parent_ids = set(
        all_df[all_df["level"] == 2]["parent_section_id"].dropna().astype(int)
    )
    all_df["is_retrieval_unit"] = all_df.apply(
        lambda r: True if r["level"] == 2 else int(r["section_id"]) not in l2_parent_ids,
        axis=1,
    )

    # --- sections.parquet (L1 only) ---
    secs_df = (
        all_df[all_df["level"] == 1]
        .drop(columns=["level", "parent_section_id"])
        .reset_index(drop=True)
    )
    secs_df.to_parquet(output_dir / "sections.parquet", index=False)

    # --- subsections.parquet (L2 only) ---
    subs_df = (
        all_df[all_df["level"] == 2]
        .drop(columns=["level"])
        .reset_index(drop=True)
    )
    subs_df.to_parquet(output_dir / "subsections.parquet", index=False)

    _validate(docs_df, secs_df, subs_df)


def _validate(docs: pd.DataFrame, secs: pd.DataFrame, subs: pd.DataFrame) -> None:
    # documents
    assert len(docs) == 8, f"Expected 8 documents, got {len(docs)}"
    assert docs["doc_id"].is_unique
    assert {"doc_id", "sparte", "tarif", "numbering_scheme", "source_file"}.issubset(docs.columns)
    assert set(docs["sparte"].unique()) == {"Kfz", "Hausrat", "Glas", "Schmuck"}

    # sections (L1)
    assert secs["section_id"].is_unique, "Duplicate section_ids"
    required_sec_cols = {"doc_id", "section_id", "sparte", "tarif", "section_code",
                         "section_types", "topic_tags", "heading", "markdown",
                         "breadcrumb", "confidence_score", "is_retrieval_unit"}
    assert required_sec_cols.issubset(secs.columns), f"Missing cols: {required_sec_cols - set(secs.columns)}"
    assert "embedding" not in secs.columns, "sections.parquet must not contain embedding (use build_embeddings.py)"
    for col in ["doc_id", "section_code", "heading", "markdown", "breadcrumb"]:
        assert secs[col].notna().all(), f"Nulls in sections.{col}"
    assert len(secs[~secs["doc_id"].isin(docs["doc_id"])]) == 0, "Orphan sections"

    # subsections (L2)
    assert subs["section_id"].is_unique, "Duplicate section_ids in subsections"
    required_sub_cols = {"doc_id", "section_id", "sparte", "tarif", "section_code",
                         "section_types", "topic_tags", "heading", "markdown",
                         "breadcrumb", "confidence_score", "parent_section_id", "is_retrieval_unit"}
    assert required_sub_cols.issubset(subs.columns), f"Missing cols: {required_sub_cols - set(subs.columns)}"
    assert "embedding" not in subs.columns, "subsections.parquet must not contain embedding"
    for col in ["doc_id", "section_code", "heading", "markdown", "breadcrumb"]:
        assert subs[col].notna().all(), f"Nulls in subsections.{col}"
    assert subs["parent_section_id"].notna().all(), "Nulls in subsections.parent_section_id"
    orphan_parents = subs[~subs["parent_section_id"].isin(secs["section_id"])]
    assert len(orphan_parents) == 0, f"Subsections with missing parent: {orphan_parents['section_id'].tolist()}"

    # is_retrieval_unit counts
    total_retrieval = secs["is_retrieval_unit"].sum() + subs["is_retrieval_unit"].sum()
    assert 350 <= total_retrieval <= 420, f"is_retrieval_unit count {total_retrieval} out of expected range 350–420"
    assert subs["is_retrieval_unit"].all(), "All L2 subsections must have is_retrieval_unit=True"


if __name__ == "__main__":
    build(
        corpus_dir=Path("sources/output_md"),
        output_dir=Path("parquet"),
    )
