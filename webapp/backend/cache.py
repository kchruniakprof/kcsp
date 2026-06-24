"""Answer cache utilities: norm_query, build_version_tag, make_query_hash."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def norm_query(query: str) -> str:
    """Normalize a query for cache key: lowercase, trim, collapse whitespace."""
    return re.sub(r"\s+", " ", query.strip().lower())


def build_version_tag(
    src_contents: dict[str, str],
    model_ids: list[str],
    corpus_chunk_ids: list[str],
) -> str:
    """Compute a stable 64-char hex tag that changes when src/models/corpus change.

    Inputs sorted before hashing so order of file dict / list items is irrelevant.
    Changing webapp/ alone does NOT change the tag — only src/, model IDs, corpus.
    """
    src_hash = hashlib.sha256(
        "".join(v for _, v in sorted(src_contents.items())).encode()
    ).hexdigest()
    models_str = "|".join(sorted(model_ids))
    corpus_hash = hashlib.sha256(
        "|".join(sorted(corpus_chunk_ids)).encode()
    ).hexdigest()
    combined = f"{src_hash}:{models_str}:{corpus_hash}"
    return hashlib.sha256(combined.encode()).hexdigest()


def make_query_hash(norm_q: str, vtag: str) -> str:
    """Primary key for answer_cache: sha256(norm_query + NUL + version_tag)."""
    return hashlib.sha256(f"{norm_q}\x00{vtag}".encode()).hexdigest()


def read_src_contents(src_dir: Path) -> dict[str, str]:
    """Read all *.py files in src_dir → {filename: content} for version_tag."""
    result: dict[str, str] = {}
    for p in sorted(src_dir.glob("*.py")):
        try:
            result[p.name] = p.read_text(encoding="utf-8")
        except Exception:
            pass
    return result
