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

class _BGE3Embedder:
    """Wraps BGEM3FlagModel — exposes encode() for dense and encode_sparse() for RRF."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(model_name, use_fp16=True)

    def encode(self, texts: Any, normalize_embeddings: bool = True, **kwargs) -> Any:
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        out = self._model.encode(texts, return_dense=True, return_sparse=False, batch_size=12)
        return out["dense_vecs"]

    def encode_sparse(self, texts: Any) -> list:
        if isinstance(texts, str):
            texts = [texts]
        out = self._model.encode(texts, return_dense=False, return_sparse=True, batch_size=12)
        return out["lexical_weights"]


@lru_cache(maxsize=1)
def _get_rag():
    import pickle
    import numpy as np
    import pandas as pd
    from groq import Groq

    from src.critic import Critic
    from src.generator import Generator
    from src.llm_providers import groq_client
    from src.query_expansion import QueryExpansion
    from src.ragassistant import RAGAssistant
    from src.retriever import CrossEncoderReranker, Retriever

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

    # D3: load sparse embeddings if available (sections + subsections pkl)
    sparse_embs: list | None = None
    secs_mask = secs_df["is_retrieval_unit"] == True
    subs_mask = subs_df["is_retrieval_unit"] == True
    secs_sparse_path = parquet_dir / "sections_sparse.pkl"
    subs_sparse_path = parquet_dir / "subsections_sparse.pkl"
    if secs_sparse_path.exists() and subs_sparse_path.exists():
        with open(secs_sparse_path, "rb") as f:
            secs_sparse_all = pickle.load(f)
        with open(subs_sparse_path, "rb") as f:
            subs_sparse_all = pickle.load(f)
        # Filter to retrieval units only, matching order of retrieval_df
        secs_sparse = [s for s, m in zip(secs_sparse_all, secs_mask) if m]
        subs_sparse = [s for s, m in zip(subs_sparse_all, subs_mask) if m]
        sparse_embs = secs_sparse + subs_sparse

    embedder = _BGE3Embedder()
    retriever = Retriever(sections=sections, embedder=embedder, sec_embs=sec_embs,
                          sparse_embs=sparse_embs, reranker=CrossEncoderReranker())

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


# ---------------------------------------------------------------------------
# Promptfoo provider interface
# ---------------------------------------------------------------------------

def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    try:
        rag = _get_rag()
        result = rag.ask(prompt)
        return {
            "output": result.answer,
            "metadata": {
                "abstained": result.abstained,
                "sources": result.sources,
                "breadcrumbs": result.breadcrumbs,
                "intent": result.intent.value,
                "cross_sell": result.cross_sell,
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
