# ADR-008: Retriever Architecture — Port from DKV Pattern + Verbatim Guarantee

**Status:** Accepted  
**Date:** 2026-06-20  
**Deciders:** K. Chruniak  

---

## Context

Current `src/retriever.py`: `heading + markdown[:512]` embed-text, always returns top_k, no similarity threshold, no pre-filter. Known failures: low-relevance chunks returned (no abstain), no product/coverage filter, no reranking.

DKV Belgium (`d:/_FUN/DKV_Belgium/calude/accuracy/src/`) solved analogous problems with: DocFilter, ContextSelector (TopK + BruteForce fallback), similarity threshold, ContextPruner, EmbeddingPruner.

**Critical difference:** DKV generator **rewrites** content → pruning at sentence level is safe. KCSP generator **cites verbatim** (`{markdown, heading, section_code}` from whitelist) → pruning the cited block breaks the verbatim guarantee.

---

## Decision

Port the DKV retriever pattern to KCSP with four adaptations:

### 1. DocFilter (`src/doc_filter.py`)

Protocol + adapters, `CompositeDocFilter` (union of frozenset `doc_id`):

- **`ProductDetectorAdapter`** — maps Sparte/Tarif from `QueryExpansion` output to `doc_id` set via `documents.parquet`. Replaces current ad-hoc sparte/tarif filter in `retriever.py`.
- **`RareTagMatcherAdapter`** — exact-match `topic_tags` from enrichment against query `domain_terms`; returns `doc_id` set. New capability.

### 2. ContextSelector (`src/llm_selector.py`)

- Primary: `TopKReranker` over pre-filtered candidates.
- Fallback: `BruteForceReranker` over full corpus (if primary yields `confidence < threshold`).
- **KCSP-specific:** if `top_score < EMBED_THRESHOLD` → **abstain** (return no-answer) instead of returning low-confidence chunk.
- `EMBED_THRESHOLD`: **calibrate on eval set** — do NOT copy DKV's 0.40 (BGE-M3 ≠ MiniLM; different cosine scale).
- LLM selector model: **Groq** (runtime path, latency-sensitive).

### 3. ContextPruner (`src/context_pruner.py` + `src/embedding_pruner.py`)

Port sentence-level and embedding-level pruning from DKV.

**VERBATIM CONSTRAINT:** Pruner operates on a **`pruned-for-reasoning` view** fed to LLM (selector, critic). The **`verbatim-for-citation` view** (full `markdown` block) passes through untouched to the generator's whitelist. Two views, single chunk object.

- Global bypass: skip pruning if chunk < 2500 chars (DKV pattern — preserve short sections entirely).
- Empty-guard: pruner must never produce empty context (fall back to full chunk).

### 4. Abstain path

Low top-score (`< EMBED_THRESHOLD`) → return structured no-answer without hallucinating. Threshold calibrated in Phase E (`eval_full` re-run post-rebuild).

---

## Module mapping

| DKV source | KCSP target |
|---|---|
| `doc_filter.py` | `src/doc_filter.py` (Protocol + adapters) |
| `reranker_strategy.py` | `src/reranker_strategy.py` (TopK + BruteForce) |
| `llm_selector.py` | `src/llm_selector.py` (Groq model) |
| `context_pruner.py` | `src/context_pruner.py` |
| `embedding_pruner.py` | `src/embedding_pruner.py` |

---

## Rationale

DKV pattern is production-validated for analogous insurance RAG task. Port + adapt is faster than designing from scratch and lower risk. Critical adaptations (verbatim constraint, abstain vs fallback, threshold calibration) address the exact points where blind copying would break KCSP invariants.

---

## Consequences

- `src/retriever.py` refactored to wire DocFilter → ContextSelector → Pruner (two views).
- **Every portedmodule must have a test verifying verbatim passthrough** (pruned view ≠ verbatim view; verbatim view == original `markdown`).
- `EMBED_THRESHOLD` stored in config (not hardcoded); default `None` until calibrated in Phase E.
- `src/ragassistant.py` updated to consume new retriever interface.
- `GROQ_API_KEY` required for selector/critic runtime path.

## Revisit triggers

- Eval abstain rate > 15% on valid questions → lower threshold or improve embedding quality.
- Verbatim guarantee extended to sub-paragraph level → pruner scope may shrink further.
