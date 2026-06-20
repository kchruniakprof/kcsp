"""
build_embeddings: load enriched parquet → compose embed-text → encode BGE-M3 → save.

Embed-text composition (ADR-006):
  heading \\n title \\n description \\n Q1 \\n … \\n Q5 \\n markdown[:400]

Only is_retrieval_unit=True rows receive embeddings.
L1-parents (is_retrieval_unit=False) get embedding=None.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_BGE_MODEL_NAME = "BAAI/bge-m3"
_BATCH_SIZE = 32


def _load_model() -> Any:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_BGE_MODEL_NAME)


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    try:
        import pandas as _pd
        if _pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _embed_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in ("heading", "title", "description"):
        val = row.get(field)
        if not _is_missing(val) and str(val).strip():
            parts.append(str(val).strip())

    raw_qs = row.get("questions")
    if _is_missing(raw_qs):
        questions: list = []
    else:
        questions = list(raw_qs)

    for q in questions[:5]:
        if q and str(q).strip():
            parts.append(str(q).strip())

    md = row.get("markdown")
    body = str(md)[:400] if not _is_missing(md) else ""
    if body.strip():
        parts.append(body)

    return "\n".join(parts)


def _embed_dataframe(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Add 'embedding' column; retrieval-unit rows get vectors, parents get None."""
    df = df.copy()
    df["embedding"] = None

    mask = df["is_retrieval_unit"] == True
    retrieval_df = df[mask]

    if len(retrieval_df) == 0:
        return df

    texts = [_embed_text(row) for _, row in retrieval_df.iterrows()]
    embeddings = model.encode(
        texts,
        batch_size=_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for i, (idx, _) in enumerate(retrieval_df.iterrows()):
        df.at[idx, "embedding"] = embeddings[i]

    return df


def _validate(df: pd.DataFrame) -> None:
    mask = df["is_retrieval_unit"] == True
    embs = df.loc[mask, "embedding"]
    for e in embs:
        if e is None:
            raise ValueError("Missing embedding for is_retrieval_unit=True row")
        arr = np.array(e, dtype=np.float32)
        assert arr.shape == (1024,), f"Expected (1024,), got {arr.shape}"
        norm = float(np.linalg.norm(arr))
        assert abs(norm - 1.0) < 1e-3, f"Embedding not normalized: norm={norm}"


def build_embeddings(
    parquet_dir: Path,
    validate: bool = True,
) -> None:
    parquet_dir = Path(parquet_dir)
    model = _load_model()

    for fname in ("sections.parquet", "subsections.parquet"):
        fpath = parquet_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        enriched = _embed_dataframe(df, model)
        if validate:
            _validate(enriched)
        enriched.to_parquet(fpath, index=False)
        print(f"[build_embeddings] Written: {fpath} (embeddings added)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build BGE-M3 embeddings for retrieval units.")
    parser.add_argument("--parquet-dir", default="parquet")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    build_embeddings(Path(args.parquet_dir), validate=not args.no_validate)
