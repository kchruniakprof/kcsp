"""BM25 sparse encoder — replaces BGE-M3 lexical weights for RRF fusion.

Build time: build_bm25_sparse(texts) → (sparse_list, idf_dict)
  sparse_list[i] = {word_hash: tf_component}  → stored in *_sparse.pkl
  idf_dict       = {word_hash: idf_value}      → stored in bm25_idf.pkl

Query time: encode_query_sparse(query, idf_dict) → {word_hash: idf_value}

RRF dot-product: sum(tf_component * idf_value) ≈ BM25 score.
No external dependencies — stdlib + built-in hash only.
"""
from __future__ import annotations

import math
import re


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _word_hash(word: str) -> int:
    """FNV-1a 24-bit hash — same collision rate as BGE-M3 vocab (~32k terms)."""
    h = 2166136261
    for c in word.encode("utf-8"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFF
    return h


def build_bm25_sparse(
    texts: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[list[dict[int, float]], dict[int, float]]:
    """Return (sparse_list, idf_dict) for a corpus of texts.

    sparse_list[i]: {word_hash: tf_bm25_component} for document i
    idf_dict:       {word_hash: idf} for the whole corpus
    """
    tokenized = [_tokenize(t) for t in texts]
    N = len(tokenized)
    avgdl = sum(len(t) for t in tokenized) / max(N, 1)

    df: dict[int, int] = {}
    for tokens in tokenized:
        for h in {_word_hash(w) for w in tokens}:
            df[h] = df.get(h, 0) + 1

    idf: dict[int, float] = {
        h: math.log((N - n + 0.5) / (n + 0.5) + 1)
        for h, n in df.items()
    }

    sparse_list: list[dict[int, float]] = []
    for tokens in tokenized:
        dl = len(tokens)
        raw: dict[int, int] = {}
        for w in tokens:
            h = _word_hash(w)
            raw[h] = raw.get(h, 0) + 1
        sparse_list.append({
            h: (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avgdl, 1)))
            for h, f in raw.items()
        })

    return sparse_list, idf


def encode_query_sparse(query: str, idf: dict[int, float]) -> dict[int, float]:
    """Return {word_hash: idf} for query terms present in the corpus vocabulary."""
    result: dict[int, float] = {}
    for w in _tokenize(query):
        h = _word_hash(w)
        if h in idf:
            result[h] = idf[h]
    return result
