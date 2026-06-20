"""
enrich_sections: batch Core-4 enrichment with checkpoint/resume/cost-gate.

Usage:
    python -m src.enrich_sections [--yes]

Flow:
  1. Load is_retrieval_unit=True rows from sections_df
  2. Cost-gate: print estimated count, prompt "Proceed? [y/N]" (skipped with --yes / auto_yes)
  3. For each non-checkpointed section: call enrich_section(); write Core-4 back to parquet
  4. Checkpoint updated immediately after each success: parquet/enrichment_checkpoint.json
  5. Failures → parquet/enrichment_errors.json (log + skip; no crash)

L1-parents (is_retrieval_unit=False) are NEVER enriched.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.enrichment import DEFAULT_MODEL, SectionDetails, enrich_section
from src.llm_providers import openrouter_client
from src.model_registry import REGISTRY
from src.observability import get_logger

_log = get_logger("enrich_sections")

_CHECKPOINT_FILE = "enrichment_checkpoint.json"
_ERRORS_FILE = "enrichment_errors.json"


def _load_checkpoint(output_dir: Path) -> dict[str, bool]:
    cp = output_dir / _CHECKPOINT_FILE
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))
    return {}


def _save_checkpoint(output_dir: Path, checkpoint: dict[str, bool]) -> None:
    (output_dir / _CHECKPOINT_FILE).write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )


def _append_error(output_dir: Path, section_id: int, error: str) -> None:
    ef = output_dir / _ERRORS_FILE
    errors: list[dict] = []
    if ef.exists():
        errors = json.loads(ef.read_text(encoding="utf-8"))
    errors.append({"section_id": section_id, "error": error})
    ef.write_text(json.dumps(errors), encoding="utf-8")


def enrich_sections(
    sections_df: pd.DataFrame,
    output_dir: Path,
    client: Optional[Any] = None,
    model: str = DEFAULT_MODEL,
    auto_yes: bool = False,
    output_filename: str = "sections_enriched.parquet",
) -> pd.DataFrame:
    """Enrich all is_retrieval_unit=True rows with Core-4 fields.

    Returns the updated DataFrame (also written to output_dir/output_filename).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    c = client if client is not None else openrouter_client()

    retrieval_df = sections_df[sections_df["is_retrieval_unit"] == True].copy()
    total = len(retrieval_df)

    if not auto_yes:
        print(f"\n[enrich_sections] Will enrich {total} retrieval-unit sections via LLM.")
        print(f"  Model: {model} (OpenRouter)")
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("[enrich_sections] Aborted by user.")
            return sections_df

    checkpoint = _load_checkpoint(output_dir)
    _log.info("start", total=total, already_done=len(checkpoint))

    result_df = sections_df.copy()
    # Ensure Core-4 columns exist with object dtype before assigning list values
    for col in ("title", "description", "questions", "topic_tags"):
        if col not in result_df.columns:
            result_df[col] = None
    # Force object dtype so list assignments don't trigger broadcast errors
    for col in ("questions", "topic_tags"):
        result_df[col] = result_df[col].astype(object)

    for _, row in retrieval_df.iterrows():
        sid = str(row["section_id"])
        if checkpoint.get(sid):
            _log.info("skip_checkpointed", section_id=sid)
            continue

        section = row.to_dict()
        try:
            details: SectionDetails = enrich_section(section, client=c, model=model)
            i = result_df.index[result_df["section_id"] == row["section_id"]][0]
            result_df.at[i, "title"] = details.title
            result_df.at[i, "description"] = details.description
            result_df.at[i, "questions"] = details.questions
            result_df.at[i, "topic_tags"] = details.topic_tags
            checkpoint[sid] = True
            _save_checkpoint(output_dir, checkpoint)
            _log.info("enriched", section_id=sid)
        except Exception as exc:
            _log.error("error", section_id=sid, error=str(exc))
            _append_error(output_dir, int(sid), str(exc))

    result_df.to_parquet(output_dir / output_filename, index=False)
    _log.info("done", output=str(output_dir / output_filename))
    return result_df


if __name__ == "__main__":
    import argparse
    from src.build_parquets import build

    parser = argparse.ArgumentParser(description="Enrich sections with Core-4 fields.")
    parser.add_argument("--parquet-dir", default="parquet", help="Directory with sections.parquet")
    parser.add_argument("--yes", action="store_true", help="Skip cost-gate prompt")
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    secs = pd.read_parquet(parquet_dir / "sections.parquet")
    subs = pd.read_parquet(parquet_dir / "subsections.parquet")

    print("[enrich_sections] Enriching sections.parquet ...")
    enrich_sections(secs, parquet_dir, auto_yes=args.yes, output_filename="sections.parquet")

    print("[enrich_sections] Enriching subsections.parquet ...")
    enrich_sections(subs, parquet_dir, auto_yes=args.yes, output_filename="subsections.parquet")
