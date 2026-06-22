"""Tests for build_embeddings — TDD C1 (BGE-M3 mocked)."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── _embed_text unit tests ─────────────────────────────────────────────────────

def test_embed_text_importable():
    from src.build_embeddings import _embed_text
    assert _embed_text is not None


def test_embed_text_full_core4():
    from src.build_embeddings import _embed_text
    row = {
        "heading": "§A Versicherte Risiken",
        "title": "Deckungsumfang Kfz-Spezial",
        "description": "Welche Schäden sind abgedeckt.",
        "questions": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?", "Q6?"],
        "markdown": "A" * 600,
    }
    text = _embed_text(row)
    assert "§A Versicherte Risiken" in text
    assert "Deckungsumfang Kfz-Spezial" in text
    assert "Welche Schäden sind abgedeckt." in text
    assert "Q1?" in text
    assert "Q5?" in text
    # Q6 must NOT be in embed text (only first 5 questions)
    assert "Q6?" not in text


def test_embed_text_missing_title_no_crash():
    from src.build_embeddings import _embed_text
    row = {
        "heading": "§B",
        "title": None,
        "description": "Beschreibung.",
        "questions": ["Q?"],
        "markdown": "short",
    }
    text = _embed_text(row)
    assert "§B" in text
    assert "Beschreibung." in text
    assert text  # not empty


def test_embed_text_no_placeholder_for_missing_fields():
    from src.build_embeddings import _embed_text
    row = {
        "heading": "§C",
        "title": None,
        "description": None,
        "questions": None,
        "markdown": "text",
    }
    text = _embed_text(row)
    # No placeholder text like "None", "null", "N/A"
    assert "None" not in text
    assert "null" not in text


def test_embed_text_markdown_truncated_at_400():
    from src.build_embeddings import _embed_text
    long_md = "X" * 600
    row = {
        "heading": "§D",
        "title": "T",
        "description": "D",
        "questions": ["Q?"],
        "markdown": long_md,
    }
    text = _embed_text(row)
    # The markdown portion must be exactly 400 chars
    assert "X" * 400 in text
    assert "X" * 401 not in text


def test_embed_text_short_markdown_uses_full():
    from src.build_embeddings import _embed_text
    row = {
        "heading": "§E",
        "title": None,
        "description": None,
        "questions": None,
        "markdown": "Kurz",
    }
    text = _embed_text(row)
    assert "Kurz" in text


def test_embed_text_questions_max_5():
    from src.build_embeddings import _embed_text
    row = {
        "heading": "§F",
        "title": None,
        "description": None,
        "questions": [f"Q{i}?" for i in range(10)],
        "markdown": "md",
    }
    text = _embed_text(row)
    for i in range(5):
        assert f"Q{i}?" in text
    for i in range(5, 10):
        assert f"Q{i}?" not in text


def test_embed_text_newline_separator():
    from src.build_embeddings import _embed_text
    row = {
        "heading": "H",
        "title": "T",
        "description": "D",
        "questions": ["Q?"],
        "markdown": "M",
    }
    text = _embed_text(row)
    assert "\n" in text


# ── build_embeddings integration (mocked BGE-M3) ─────────────────────────────

def _make_enriched_df():
    return pd.DataFrame([
        {
            "section_id": 1, "doc_id": "kfz", "sparte": "Kfz", "tarif": "Spezial",
            "heading": "§A", "markdown": "Text A", "breadcrumb": "Kfz > §A",
            "is_retrieval_unit": True,
            "title": "Titel A", "description": "Beschreibung A",
            "questions": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
            "topic_tags": ["Kfz"],
        },
        {
            "section_id": 2, "doc_id": "kfz", "sparte": "Kfz", "tarif": "Spezial",
            "heading": "§B Parent", "markdown": "Text B", "breadcrumb": "Kfz > §B",
            "is_retrieval_unit": False,  # L1-parent — no embedding
            "title": None, "description": None, "questions": None, "topic_tags": None,
        },
    ])


def test_build_embeddings_importable():
    from src.build_embeddings import build_embeddings
    assert build_embeddings is not None


def _fake_model_output(n: int):
    return {
        "dense_vecs": np.random.rand(n, 1024).astype("float32"),
        "lexical_weights": [{42: 0.5} for _ in range(n)],
    }


def test_build_embeddings_adds_embedding_column(tmp_path):
    from src.build_embeddings import build_embeddings
    df = _make_enriched_df()
    df.to_parquet(tmp_path / "sections.parquet", index=False)

    fake_model = MagicMock()
    fake_model.encode.return_value = _fake_model_output(1)  # 1 retrieval unit

    with patch("src.build_embeddings._load_model", return_value=fake_model):
        build_embeddings(parquet_dir=tmp_path, validate=False)

    out = pd.read_parquet(tmp_path / "sections.parquet")
    assert "embedding" in out.columns


def test_build_embeddings_saves_sparse_pkl(tmp_path):
    """D3: build_embeddings must save {stem}_sparse.pkl next to the parquet."""
    from src.build_embeddings import build_embeddings
    import pickle
    df = _make_enriched_df()
    df.to_parquet(tmp_path / "sections.parquet", index=False)

    fake_model = MagicMock()
    fake_model.encode.return_value = _fake_model_output(1)

    with patch("src.build_embeddings._load_model", return_value=fake_model):
        build_embeddings(parquet_dir=tmp_path, validate=False)

    sparse_path = tmp_path / "sections_sparse.pkl"
    assert sparse_path.exists(), "sections_sparse.pkl must be saved alongside sections.parquet"
    with open(sparse_path, "rb") as f:
        sparse_list = pickle.load(f)
    assert isinstance(sparse_list, list)
    assert len(sparse_list) == len(df)


def test_l1_parent_has_no_embedding(tmp_path):
    from src.build_embeddings import build_embeddings
    df = _make_enriched_df()
    df.to_parquet(tmp_path / "sections.parquet", index=False)

    fake_model = MagicMock()
    fake_model.encode.return_value = _fake_model_output(1)

    with patch("src.build_embeddings._load_model", return_value=fake_model):
        build_embeddings(parquet_dir=tmp_path, validate=False)

    out = pd.read_parquet(tmp_path / "sections.parquet")
    parent_row = out[out["section_id"] == 2]
    assert parent_row["embedding"].isna().all() or parent_row["embedding"].iloc[0] is None


def test_d3_sparse_pkl_retrieval_unit_has_weights(tmp_path):
    """D3: sparse pkl — retrieval-unit rows must have non-None sparse dicts."""
    from src.build_embeddings import build_embeddings
    import pickle
    df = _make_enriched_df()
    df.to_parquet(tmp_path / "sections.parquet", index=False)

    fake_model = MagicMock()
    fake_model.encode.return_value = _fake_model_output(1)

    with patch("src.build_embeddings._load_model", return_value=fake_model):
        build_embeddings(parquet_dir=tmp_path, validate=False)

    with open(tmp_path / "sections_sparse.pkl", "rb") as f:
        sparse_list = pickle.load(f)

    # section_id=1 is_retrieval_unit=True → sparse[0] must be dict
    assert isinstance(sparse_list[0], dict), "Retrieval unit must have sparse dict"
    # section_id=2 is_retrieval_unit=False → sparse[1] must be None
    assert sparse_list[1] is None, "L1-parent must have None sparse"
