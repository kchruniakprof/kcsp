"""
build_embeddings: load enriched parquet → compose embed-text → encode Qwen3-Embedding-8B → save.

Embed-text composition (ADR-006):
  heading \n title \n description \n Q1 \n … \n Q5 \n markdown[:400]

Only is_retrieval_unit=True rows receive embeddings.
L1-parents (is_retrieval_unit=False) get embedding=None.

E1: switched from BGE-M3 (local FP16) to Qwen3-Embedding-8B via OpenRouter API.
Dense: OpenRouter /v1/embeddings, model=qwen/qwen3-embedding-8b, full 4096-dim.
Sparse: BM25 (bm25_encoder), IDF computed over full corpus (sections + subsections).
IDF saved to parquet/bm25_idf.pkl for runtime query encoding.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.bm25_encoder import build_bm25_sparse

_MODEL_NAME = os.environ.get("EMBED_MODEL", "accounts/fireworks/models/qwen3-embedding-8b")
_BATCH_SIZE = 16
_EMBED_DIM = int(os.environ.get("EMBED_DIM", "4096"))


def _openrouter_client() -> Any:
    import openai
    base_url = os.environ.get("EMBED_BASE_URL", "https://api.fireworks.ai/inference/v1")
    key_var = os.environ.get("EMBED_API_KEY_ENV", "FIREWORKS_API_KEY")
    key = os.environ.get(key_var, "")
    if not key:
        raise RuntimeError(f"{key_var} missing from .env")
    return openai.OpenAI(api_key=key, base_url=base_url)


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
    body = str(md)[:2500] if not _is_missing(md) else ""
    if body.strip():
        parts.append(body)

    return "\n".join(parts)


def _encode_batch(client: Any, texts: list[str]) -> list[list[float]]:
    """Encode one batch via Fireworks. Returns list of float vectors."""
    for attempt in range(6):
        try:
            response = client.embeddings.create(
                model=_MODEL_NAME,
                input=texts,
                encoding_format="float",
            )
            return [e.embedding for e in response.data]
        except Exception as exc:
            if attempt == 5:
                raise
            wait = 5 * (2 ** attempt)  # 5, 10, 20, 40, 80s
            print(f"[build_embeddings] API error (attempt {attempt+1}): {exc}. Retry in {wait}s...")
            time.sleep(wait)
    return []  # unreachable


def _embed_dataframe(
    df: pd.DataFrame,
    client: Any,
    bm25_sparse: list[dict[int, float]],
    all_texts_offset: int,
) -> tuple[pd.DataFrame, list]:
    """Add 'embedding' column (dense); align sparse from precomputed bm25_sparse.

    bm25_sparse is indexed over ALL retrieval units across both parquets.
    all_texts_offset: index into bm25_sparse for the first retrieval unit in this df.
    Returns (df_with_embedding, sparse_list_aligned_to_df_rows).
    """
    df = df.copy()
    df["embedding"] = None

    mask = df["is_retrieval_unit"] == True
    retrieval_df = df[mask]
    sparse_list: list = [None] * len(df)

    if len(retrieval_df) == 0:
        return df, sparse_list

    texts = [_embed_text(row) for _, row in retrieval_df.iterrows()]
    all_dense: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i: i + _BATCH_SIZE]
        print(f"[build_embeddings] Encoding batch {i//  _BATCH_SIZE + 1}/{(len(texts)-1)//_BATCH_SIZE + 1} ({len(batch)} texts)...")
        vecs = _encode_batch(client, batch)
        all_dense.extend(vecs)
        time.sleep(2)

    for i, (idx, _) in enumerate(retrieval_df.iterrows()):
        vec = np.array(all_dense[i], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        df.at[idx, "embedding"] = vec
        sparse_list[idx] = bm25_sparse[all_texts_offset + i]

    return df, sparse_list


def _validate(df: pd.DataFrame) -> None:
    mask = df["is_retrieval_unit"] == True
    embs = df.loc[mask, "embedding"]
    for e in embs:
        if e is None:
            raise ValueError("Missing embedding for is_retrieval_unit=True row")
        arr = np.array(e, dtype=np.float32)
        assert arr.shape == (_EMBED_DIM,), f"Expected ({_EMBED_DIM},), got {arr.shape}"
        norm = float(np.linalg.norm(arr))
        assert abs(norm - 1.0) < 1e-3, f"Embedding not normalized: norm={norm}"


def build_embeddings(
    parquet_dir: Path,
    validate: bool = True,
) -> None:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    parquet_dir = Path(parquet_dir)
    client = _openrouter_client()

    # ── Step 1: collect all retrieval-unit texts for global BM25 IDF ──────────
    fnames = [f for f in ("sections.parquet", "subsections.parquet")
              if (parquet_dir / f).exists()]
    all_texts: list[str] = []
    dfs: list[pd.DataFrame] = []
    masks: list[Any] = []

    for fname in fnames:
        df = pd.read_parquet(parquet_dir / fname)
        mask = df["is_retrieval_unit"] == True
        texts = [_embed_text(row) for _, row in df[mask].iterrows()]
        all_texts.extend(texts)
        dfs.append(df)
        masks.append(mask)

    print(f"[build_embeddings] Building BM25 IDF over {len(all_texts)} retrieval units...")
    bm25_sparse, idf = build_bm25_sparse(all_texts)

    # Save global IDF for runtime query encoding
    idf_path = parquet_dir / "bm25_idf.pkl"
    with open(idf_path, "wb") as f:
        pickle.dump(idf, f)
    print(f"[build_embeddings] IDF written: {idf_path} ({len(idf)} terms)")

    # ── Step 2: encode dense + write per-parquet ───────────────────────────────
    offset = 0
    for df, mask, fname in zip(dfs, masks, fnames):
        fpath = parquet_dir / fname
        n_units = int(mask.sum())
        enriched, sparse_list = _embed_dataframe(
            df, client, bm25_sparse, all_texts_offset=offset
        )
        offset += n_units

        if validate:
            _validate(enriched)

        enriched.to_parquet(fpath, index=False)
        sparse_path = parquet_dir / f"{fpath.stem}_sparse.pkl"
        with open(sparse_path, "wb") as f:
            pickle.dump(sparse_list, f)
        print(f"[build_embeddings] Written: {fpath} + {sparse_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Qwen3-Embedding-8B embeddings via OpenRouter.")
    parser.add_argument("--parquet-dir", default="parquet")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    build_embeddings(Path(args.parquet_dir), validate=not args.no_validate)
