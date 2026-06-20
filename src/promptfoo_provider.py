"""
Promptfoo Python provider for ERGO RAGAssistant.

Promptfoo interface:
  call_api(prompt, options, context) -> {"output": str, "metadata": {...}}

CLI entrypoint (for direct testing):
  python src/promptfoo_provider.py "<question>"
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

# ensure project root on sys.path regardless of CWD (promptfoo spawns Python elsewhere)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

# ---------------------------------------------------------------------------
# Lazy RAGAssistant factory — built once per process
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_rag():
    import numpy as np
    import pandas as pd
    from groq import Groq
    from sentence_transformers import SentenceTransformer

    from src.critic import Critic
    from src.generator import Generator
    from src.llm_selector import ContextSelector
    from src.llm_providers import groq_client
    from src.query_expansion import QueryExpansion
    from src.ragassistant import RAGAssistant
    from src.retriever import Retriever

    import instructor
    from src.model_registry import REGISTRY

    api_key = os.environ["GROQ_API_KEY"]
    groq_client_raw = Groq(api_key=api_key)
    instructor_client = instructor.from_groq(groq_client_raw, mode=instructor.Mode.JSON)

    parquet_dir = _ROOT / "parquet"
    secs_df = pd.read_parquet(parquet_dir / "sections.parquet")
    subs_df = pd.read_parquet(parquet_dir / "subsections.parquet")
    docs_df = pd.read_parquet(parquet_dir / "documents.parquet") if (parquet_dir / "documents.parquet").exists() else None

    all_df = pd.concat([secs_df, subs_df], ignore_index=True)

    # Only retrieval units get embeddings — filter before stacking to avoid None
    retrieval_df = all_df[all_df["is_retrieval_unit"] == True].reset_index(drop=True)
    sec_embs = np.stack(retrieval_df["embedding"].values).astype(np.float32)
    sections = retrieval_df.drop(columns=["embedding"]).to_dict("records")

    embedder = SentenceTransformer("BAAI/bge-m3")
    retriever = Retriever(sections=sections, embedder=embedder, sec_embs=sec_embs)

    enable_ensemble = os.environ.get("ENABLE_ENSEMBLE", "false").lower() == "true"
    ensemble_critic = (
        Critic(client=instructor_client, model=REGISTRY["critic_ensemble"], _wrap_instructor=False)
        if enable_ensemble else None
    )

    return RAGAssistant(
        query_expansion=QueryExpansion(api_key=api_key),
        retriever=retriever,
        generator=Generator(client=groq_client_raw),
        critic=Critic(client=instructor_client, model=REGISTRY["critic"], _wrap_instructor=False),
        top_k=5,
        enable_cross_sell=True,
        documents_df=docs_df,
        sections_df=secs_df,
        subsections_df=subs_df,
        ensemble_critic=ensemble_critic,
        enable_ensemble=enable_ensemble,
    )


@lru_cache(maxsize=1)
def _get_selector():
    """Shadow ContextSelector (threshold=None) for E1 calibration."""
    from src.llm_selector import ContextSelector
    return ContextSelector(threshold=None)


# ---------------------------------------------------------------------------
# Promptfoo provider interface
# ---------------------------------------------------------------------------

def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    try:
        rag = _get_rag()
        result = rag.ask(prompt)

        # Shadow-score via ContextSelector for E1 EMBED_THRESHOLD calibration
        selector_confidence: float | None = None
        try:
            from src.context_pruner import PrunedChunk
            from src.retriever import RetrievalResult
            # Re-run retrieval to get candidates for selector scoring
            # Use cached internals to avoid re-encoding
            selector = _get_selector()
            # Build PrunedChunk list from last retrieval results stored in result
            # Since we don't cache last retrieval, run a lightweight shadow pass
            # via the retriever directly — sources are already in result.sources
            # Skip shadow if abstained (no candidates to score)
            if not result.abstained and result.sources:
                # Collect pruned candidates from retriever for shadow scoring
                # We use the retriever's indexed sections to build PrunedChunks
                r = rag._retriever
                candidates = []
                for sec_id in result.sources:
                    sec = next((s for s in r._sections if s["section_id"] == sec_id), None)
                    if sec:
                        from src.context_pruner import ContextPruner
                        pc = r._pruner.prune(sec["markdown"])
                        candidates.append(pc)
                if candidates:
                    sel_result = selector.select(candidates, query=prompt)
                    from src.llm_selector import SelectedChunk
                    if isinstance(sel_result, SelectedChunk):
                        selector_confidence = sel_result.confidence
        except Exception:
            pass  # shadow scoring failure must never break eval

        return {
            "output": result.answer,
            "metadata": {
                "abstained": result.abstained,
                "sources": result.sources,
                "breadcrumbs": result.breadcrumbs,
                "intent": result.intent.value,
                "cross_sell": result.cross_sell,
                "selector_confidence": selector_confidence,
            },
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    result = call_api(prompt, {}, {})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
