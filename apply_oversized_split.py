"""Split oversized retrieval units in existing (enriched) parquets.

Preserves existing enrichment: adds split children with NEW unique section_ids
(no renumbering → enrichment checkpoint stays valid), marks the original
oversized unit is_retrieval_unit=False. New children are enriched separately
and embeddings rebuilt afterwards.
"""
from pathlib import Path
import pandas as pd

from src.build_parquets import _split_md_by_headings, _SPLIT_MAX_CHARS
from src.hierarchy_parser import _assign_types

PQ = Path("parquet")


def _next_id(*dfs) -> int:
    return int(max(int(df["section_id"].max()) for df in dfs)) + 1


def main() -> None:
    secs = pd.read_parquet(PQ / "sections.parquet")
    subs = pd.read_parquet(PQ / "subsections.parquet")
    nid = _next_id(secs, subs)

    new_children: list[dict] = []
    n_split = 0

    for df, has_parent in ((secs, False), (subs, True)):
        for idx, row in df.iterrows():
            if not bool(row.get("is_retrieval_unit", False)):
                continue
            md = str(row.get("markdown", ""))
            if len(md) <= _SPLIT_MAX_CHARS:
                continue
            chunks = _split_md_by_headings(md)
            if not chunks:
                continue
            n_split += 1
            df.at[idx, "is_retrieval_unit"] = False  # parent leaves the index
            base_bc = row["breadcrumb"]
            parent_id = int(row["section_id"])
            _tt = row.get("topic_tags")
            parent_tags = list(_tt) if _tt is not None and len(_tt) else []
            for n, (chunk_head, chunk_md) in enumerate(chunks, 1):
                head = chunk_head or str(row["heading"])
                child = {
                    "doc_id": row["doc_id"],
                    "section_id": nid,
                    "sparte": row["sparte"],
                    "tarif": row.get("tarif"),
                    "section_code": f"{row['section_code']}#{n}",
                    "section_types": _assign_types(head, chunk_md),
                    "topic_tags": parent_tags,
                    "heading": head,
                    "markdown": chunk_md,
                    "breadcrumb": f"{base_bc} > {head}",
                    "confidence_score": 1.0,
                    "is_retrieval_unit": True,
                    "parent_section_id": parent_id,
                    "title": None,
                    "description": None,
                    "questions": [],
                    "embedding": None,
                }
                new_children.append(child)
                nid += 1

    print(f"[split] oversized units split: {n_split} -> new children: {len(new_children)}")

    if new_children:
        children_df = pd.DataFrame(new_children)
        # align columns to subs
        for col in subs.columns:
            if col not in children_df.columns:
                children_df[col] = None
        children_df = children_df[subs.columns]
        subs = pd.concat([subs, children_df], ignore_index=True)

    secs.to_parquet(PQ / "sections.parquet", index=False)
    subs.to_parquet(PQ / "subsections.parquet", index=False)
    ru = int((secs["is_retrieval_unit"] == True).sum() + (subs["is_retrieval_unit"] == True).sum())
    print(f"[split] retrieval units now: {ru}  (secs={len(secs)} subs={len(subs)})")


if __name__ == "__main__":
    main()
