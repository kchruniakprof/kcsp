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
from typing import Optional

import pandas as pd

from src.hierarchy_parser import DOCUMENT_CATALOG, parse_all, _assign_types

_PAGE_REF_RE = re.compile(r"\([A-Z]+\.?\d*\)\s*\d+")
_SINGLE_LETTER_RE = re.compile(r"^[A-Z]$")

# --- oversized retrieval-unit splitting -----------------------------------
# Monster chunks (e.g. §E Kasko 24k, §N+Anhang/Safe-Drive 16k) glob many plain
# '## ' sub-headings that carry no L2 code, hiding deep facts from the embed
# window. Split such units at '## ' boundaries into ~target-sized child units.
_SPLIT_MAX_CHARS = 4000   # split level-2 units longer than this
_SPLIT_TARGET = 1800      # accumulate '## ' blocks up to ~this per child
_HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)


def _split_md_by_headings(markdown: str, target: int = _SPLIT_TARGET) -> list[tuple[str, str]]:
    """Split markdown at '## ' boundaries into (heading, chunk) accumulating
    blocks up to ~target chars. Returns [] when fewer than 2 headings."""
    heads = list(_HEADING_RE.finditer(markdown))
    if len(heads) < 2:
        return []
    blocks: list[tuple[str, str]] = []
    for i, h in enumerate(heads):
        start = h.start()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(markdown)
        blocks.append((h.group(1).strip(), markdown[start:end].strip()))

    chunks: list[tuple[str, str]] = []
    cur_head: Optional[str] = None
    cur_parts: list[str] = []
    cur_len = 0
    for head, body in blocks:
        if cur_parts and cur_len + len(body) > target:
            chunks.append((cur_head, "\n\n".join(cur_parts)))
            cur_head, cur_parts, cur_len = None, [], 0
        if cur_head is None:
            cur_head = head
        cur_parts.append(body)
        cur_len += len(body)
    if cur_parts:
        chunks.append((cur_head, "\n\n".join(cur_parts)))
    return chunks if len(chunks) > 1 else []


def _split_oversized_units(sec_rows: list[dict]) -> list[dict]:
    """Replace oversized level-2 rows with '## '-boundary child rows.
    Reassigns contiguous section_ids and remaps parent_section_id."""
    out: list[dict] = []
    tmp_id = 10_000_000  # temporary unique ids for new children
    for row in sec_rows:
        if row.get("level") == 2 and len(str(row.get("markdown", ""))) > _SPLIT_MAX_CHARS:
            chunks = _split_md_by_headings(str(row["markdown"]))
            if chunks:
                base_bc = row["breadcrumb"]
                for n, (chunk_head, chunk_md) in enumerate(chunks, 1):
                    tmp_id += 1
                    head = chunk_head or row["heading"]
                    out.append({
                        **row,
                        "section_id": tmp_id,
                        "section_code": f"{row['section_code']}#{n}",
                        "section_types": _assign_types(head, chunk_md),
                        "heading": head,
                        "markdown": chunk_md,
                        "breadcrumb": f"{base_bc} > {head}",
                    })
                continue
        out.append(row)

    # --- renumber contiguously, remap parent_section_id ---
    old2new = {r["section_id"]: i + 1 for i, r in enumerate(out)}
    for i, r in enumerate(out):
        r["section_id"] = i + 1
        p = r.get("parent_section_id")
        if p is not None and not pd.isna(p):
            r["parent_section_id"] = old2new.get(int(p), r["section_id"])
    return out


def is_index_section(heading: str, body: str) -> bool:
    """True when section is an alphabetical index (no insurance content)."""
    if _SINGLE_LETTER_RE.match(heading.strip()):
        return True
    lines = [l for l in body.splitlines() if l.strip()]
    if not lines:
        return False
    matched = sum(1 for l in lines if _PAGE_REF_RE.search(l))
    return matched / len(lines) >= 0.5


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

    # --- split oversized level-2 units (de-glob monster chunks) ---
    # Applied as a post-enrichment step via apply_oversized_split.py so that
    # existing Core-4 enrichment + checkpoints are preserved (see that script).
    # Not wired into build() to keep the L1/L2 id-space and counts stable.

    all_df = pd.DataFrame(sec_rows)

    # --- is_retrieval_unit: L2 always True; L1 True only if it has no children ---
    l2_parent_ids = set(
        all_df[all_df["level"] == 2]["parent_section_id"].dropna().astype(int)
    )
    all_df["is_retrieval_unit"] = all_df.apply(
        lambda r: True if r["level"] == 2 else int(r["section_id"]) not in l2_parent_ids,
        axis=1,
    )

    # --- override: alphabetical index sections are never retrieval units ---
    index_mask = all_df.apply(
        lambda r: is_index_section(r["heading"], r["markdown"]), axis=1
    )
    all_df.loc[index_mask, "is_retrieval_unit"] = False

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
