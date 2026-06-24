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

class _Qwen3Embedder:
    """OpenRouter Qwen3-Embedding-8B for dense + BM25 for sparse (RRF-compatible).

    encode()        → OpenRouter API, 4096-dim L2-normalised vectors
    encode_sparse() → BM25 query encoding using corpus IDF from bm25_idf.pkl
    """

    def __init__(self, parquet_dir: Path, api_key: str) -> None:
        import openai
        import pickle as _pickle
        base_url = os.environ.get("EMBED_BASE_URL", "https://api.fireworks.ai/inference/v1")
        self._model = os.environ.get("EMBED_MODEL", "accounts/fireworks/models/qwen3-embedding-8b")
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        idf_path = parquet_dir / "bm25_idf.pkl"
        with open(idf_path, "rb") as f:
            self._idf: dict = _pickle.load(f)

    def encode(self, texts: Any, normalize_embeddings: bool = True, **kwargs) -> Any:
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            encoding_format="float",
        )
        vecs = np.array([e.embedding for e in response.data], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vecs /= norms
        return vecs

    def encode_sparse(self, query: Any) -> dict:
        from src.bm25_encoder import encode_query_sparse
        if isinstance(query, list):
            query = query[0]
        return encode_query_sparse(str(query), self._idf)


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
    from src.retriever import CohereAPIReranker, CrossEncoderReranker, FireworksReranker, JinaAPIReranker, Retriever

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

    _key_var = os.environ.get("EMBED_API_KEY_ENV", "FIREWORKS_API_KEY")
    or_key = os.environ.get(_key_var, "")
    if not or_key:
        raise RuntimeError(f"{_key_var} missing from .env")
    embedder = _Qwen3Embedder(parquet_dir=parquet_dir, api_key=or_key)
    _reranker_model = os.environ.get("RERANKER_MODEL", "")
    if os.environ.get("DISABLE_RERANKER", "false").lower() == "true":
        _reranker = None
    elif _reranker_model.startswith("jina"):
        _reranker = JinaAPIReranker(
            model=_reranker_model,
            api_key=os.environ.get("JINA_API_KEY", ""),
        )
    elif _reranker_model.startswith("accounts/fireworks/"):
        _reranker = FireworksReranker(
            model=_reranker_model,
            api_key=os.environ.get("FIREWORKS_API_KEY", ""),
        )
    elif _reranker_model:
        _reranker = CohereAPIReranker(
            model=_reranker_model,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        _reranker = CrossEncoderReranker()
    retriever = Retriever(sections=sections, embedder=embedder, sec_embs=sec_embs,
                          sparse_embs=sparse_embs, reranker=_reranker)

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
        top_k=10,
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
