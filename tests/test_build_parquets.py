"""Tests for build_parquets — TDD cycle."""
import pandas as pd
import pytest
from pathlib import Path

from src.build_parquets import build

CORPUS = Path("D:/_FUN/kcsp/v1/sources/output_md")
OUTPUT = Path("D:/_FUN/kcsp/v1/parquet")


@pytest.fixture(scope="module")
def parquets(tmp_path_factory):
    out = tmp_path_factory.mktemp("parquet")
    build(CORPUS, out)
    return out


def test_documents_parquet_has_8_rows(parquets):
    df = pd.read_parquet(parquets / "documents.parquet")
    assert len(df) == 8


def test_documents_doc_id_unique(parquets):
    df = pd.read_parquet(parquets / "documents.parquet")
    assert df["doc_id"].is_unique


def test_documents_schema(parquets):
    df = pd.read_parquet(parquets / "documents.parquet")
    for col in ["doc_id", "sparte", "tarif", "numbering_scheme", "source_file"]:
        assert col in df.columns, f"Missing column {col!r}"


def test_documents_glas_has_related_sparte(parquets):
    df = pd.read_parquet(parquets / "documents.parquet")
    glas = df[df["sparte"] == "Glas"]
    assert len(glas) == 1
    assert glas.iloc[0]["related_sparte"] == "Hausrat"


def test_documents_schmuck_has_related_sparte(parquets):
    df = pd.read_parquet(parquets / "documents.parquet")
    schmuck = df[df["sparte"] == "Schmuck"]
    assert len(schmuck) == 1
    assert schmuck.iloc[0]["related_sparte"] == "Hausrat"


def test_documents_kfz_no_related_sparte(parquets):
    df = pd.read_parquet(parquets / "documents.parquet")
    kfz = df[df["sparte"] == "Kfz"]
    assert all(kfz["related_sparte"].isna())


def test_sections_parquet_nonempty(parquets):
    df = pd.read_parquet(parquets / "sections.parquet")
    assert len(df) > 0


def test_sections_schema(parquets):
    df = pd.read_parquet(parquets / "sections.parquet")
    for col in ["doc_id", "section_id", "sparte", "tarif", "section_code",
                "section_types", "topic_tags", "heading", "markdown",
                "breadcrumb", "confidence_score"]:
        assert col in df.columns, f"Missing column {col!r}"


def test_sections_no_null_in_required_fields(parquets):
    df = pd.read_parquet(parquets / "sections.parquet")
    for col in ["doc_id", "section_code", "heading", "markdown", "breadcrumb"]:
        assert df[col].notna().all(), f"Nulls in required column {col!r}"


def test_sections_section_id_globally_unique(parquets):
    df = pd.read_parquet(parquets / "sections.parquet")
    assert df["section_id"].is_unique


def test_sections_join_documents(parquets):
    docs = pd.read_parquet(parquets / "documents.parquet")
    secs = pd.read_parquet(parquets / "sections.parquet")
    merged = secs.merge(docs, on="doc_id", how="left")
    assert merged["sparte_x"].notna().all(), "sections.doc_id has orphans not in documents"


def test_sections_confidence_score_is_1(parquets):
    df = pd.read_parquet(parquets / "sections.parquet")
    assert (df["confidence_score"] == 1.0).all()


def test_sections_topic_tags_are_list_like(parquets):
    import numpy as np
    df = pd.read_parquet(parquets / "sections.parquet")
    for tags in df["topic_tags"]:
        assert isinstance(tags, (list, np.ndarray)), f"topic_tags must be list-like, got {type(tags)}"


def test_sparten_count(parquets):
    df = pd.read_parquet(parquets / "documents.parquet")
    sparten = set(df["sparte"].unique())
    assert sparten == {"Kfz", "Hausrat", "Glas", "Schmuck"}


# ── is_retrieval_unit ────────────────────────────────────────────────────────

def test_sections_has_is_retrieval_unit_column(parquets):
    df = pd.read_parquet(parquets / "sections.parquet")
    assert "is_retrieval_unit" in df.columns


def test_subsections_has_is_retrieval_unit_column(parquets):
    df = pd.read_parquet(parquets / "subsections.parquet")
    assert "is_retrieval_unit" in df.columns


def test_all_l2_subsections_are_retrieval_units(parquets):
    df = pd.read_parquet(parquets / "subsections.parquet")
    assert df["is_retrieval_unit"].all(), "Every L2 subsection must be a retrieval unit"


def test_l1_parents_are_not_retrieval_units(parquets):
    secs = pd.read_parquet(parquets / "sections.parquet")
    subs = pd.read_parquet(parquets / "subsections.parquet")
    parent_ids = set(subs["parent_section_id"].dropna().astype(int))
    l1_parents = secs[secs["section_id"].isin(parent_ids)]
    assert len(l1_parents) > 0, "Test setup: expected some L1 parents"
    assert not l1_parents["is_retrieval_unit"].any(), "L1 parents must have is_retrieval_unit=False"


def test_l1_leaves_are_retrieval_units(parquets):
    secs = pd.read_parquet(parquets / "sections.parquet")
    subs = pd.read_parquet(parquets / "subsections.parquet")
    parent_ids = set(subs["parent_section_id"].dropna().astype(int))
    l1_leaves = secs[~secs["section_id"].isin(parent_ids)]
    assert len(l1_leaves) > 0, "Test setup: expected some L1 leaves"
    assert l1_leaves["is_retrieval_unit"].all(), "L1 leaves must have is_retrieval_unit=True"


def test_retrieval_unit_count_approx_370(parquets):
    secs = pd.read_parquet(parquets / "sections.parquet")
    subs = pd.read_parquet(parquets / "subsections.parquet")
    total = secs["is_retrieval_unit"].sum() + subs["is_retrieval_unit"].sum()
    assert 350 <= total <= 420, f"Expected ~370 retrieval units, got {total}"


def test_sections_has_no_embedding_column(parquets):
    df = pd.read_parquet(parquets / "sections.parquet")
    assert "embedding" not in df.columns, "sections.parquet must not contain embedding after A1 refactor"
