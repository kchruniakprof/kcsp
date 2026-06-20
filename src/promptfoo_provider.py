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
    from src.query_expansion import QueryExpansion
    from src.ragassistant import RAGAssistant
    from src.retriever import Retriever

    api_key = os.environ["GROQ_API_KEY"]
    groq_client = Groq(api_key=api_key)

    parquet_dir = _ROOT / "parquet"
    secs_df = pd.read_parquet(parquet_dir / "sections.parquet")
    subs_df = pd.read_parquet(parquet_dir / "subsections.parquet")
    all_df = pd.concat([secs_df, subs_df], ignore_index=True)

    sec_embs = np.stack(all_df["embedding"].values).astype(np.float32)
    sections = all_df.drop(columns=["embedding"]).to_dict("records")

    # embedder still needed to encode queries at runtime
    embedder = SentenceTransformer("BAAI/bge-m3")
    retriever = Retriever(sections=sections, embedder=embedder, sec_embs=sec_embs)

    return RAGAssistant(
        query_expansion=QueryExpansion(api_key=api_key),
        retriever=retriever,
        generator=Generator(client=groq_client),
        critic=Critic(client=groq_client),
        top_k=5,
        enable_cross_sell=True,
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
