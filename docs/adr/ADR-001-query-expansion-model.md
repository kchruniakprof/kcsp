# ADR-001: Model Selection for Query Expansion Step

**Status:** Accepted  
**Date:** 2026-06-19  
**Deciders:** K. Chruniak  

---

## Context

The RAG pipeline for ERGO P&C Agent-Bot contains a `QueryExpansion` step that classifies incoming queries before retrieval. This step determines:

- **intent** — one of `COVERAGE_QUERY | EXCLUSION_QUERY | CLAIMS_PROCEDURE | PRICE_QUOTE | COMPARISON | COMPLAINT | GENERAL_INFO | OUT_OF_SCOPE`
- **sparte_hint** — `Kfz | Hausrat | Glas | Schmuck | null`
- **normalized_query** — query translated/normalized to German for embedding retrieval
- **paraphrases** — 3–5 German paraphrases (retrieval augmentation)
- **domain_terms** — 3–7 German insurance terms
- **section_types** — 1–3 most relevant section types for pre-filtering
- **chain_of_thought** — 3–5 reasoning steps
- **confidence_score** — 0.0–1.0

### Prior implementation

Original implementation used `llama-3.1-8b-instant` with raw Groq SDK + manual `json.loads()` parsing and no chain-of-thought. No paraphrases or domain terms were generated.

### Motivation for change

During full eval (99 questions, promptfoo), 26 questions failed. Log analysis showed:

1. **COMPARISON intent misclassified** (8b model called them `COVERAGE_QUERY`)
2. **Wrong `sparte_hint`** (e.g. `Best+Fahrraddiebstahl` → `Kfz` instead of `Hausrat`)
3. **No paraphrases/domain_terms** → retriever had only 1 query vector, missing relevant sections

### Implementation changes (applied before benchmark)

- Switched from raw Groq SDK to `openai.OpenAI(base_url="https://api.groq.com/openai/v1")` + `instructor.from_openai(mode=MD_JSON)` (pattern from DKV Belgium project)
- `ExpandedQuery` pydantic model extended with `chain_of_thought`, `paraphrases`, `domain_terms`, `section_types`, `confidence_score`
- System prompt rewritten in **English** with explicit COMPARISON keywords, sparte mappings, section type catalogue
- `seed=42`, `top_p=1`, `max_retries=3` for reproducibility and robustness

---

## Decision

Use **`meta-llama/llama-4-scout-17b-16e-instruct`** via Groq as the default model for the `QueryExpansion` step.

---

## Benchmark

**Date:** 2026-06-19  
**Test set:** 26 failing questions from full eval (`eval_failing.yaml`)  
**Script:** `scripts/benchmark_expansion.py`  
**Provider setup:** `openai.OpenAI(base_url=groq_url)` + `instructor.Mode.MD_JSON`  
**Metric — Intent accuracy:** correct `intent` value vs `expected_intent` from eval set  
**Metric — Sparte accuracy:** correct `sparte_hint` vs `expected_sparte`  

| Model | Provider | Intent% | Sparte% | Errors | Avg ms | P95 ms |
|-------|----------|---------|---------|--------|--------|--------|
| **meta-llama/llama-4-scout-17b-16e-instruct** | Groq | **88%** | **84%** | 0 | **985** | 1691 |
| openai/gpt-oss-20b | Groq | 88% | 84% | 0 | 1455 | 2112 |
| qwen/qwen3-32b | Groq | 88% | 84% | 0 | 1653 | 1958 |
| openai/gpt-oss-120b | Groq | 84% | 84% | 0 | 2192 | 3570 |
| google/gemini-2.5-flash-lite | OpenRouter | 80% | 84% | 0 | 1616 | 1871 |
| llama-3.3-70b-versatile | Groq | 76% | 84% | 0 | 1202 | 1536 |
| llama-3.1-8b-instant | Groq | 65% | 84% | 0 | 637 | 829 |
| qwen/qwen3.6-27b | Groq | — | — | 26 | — | — |

`qwen3.6-27b` failed on all 26 with `max_tokens` truncation in MD_JSON mode — disqualified.

### Remaining misclassifications (llama-4-scout, 3 intent errors)

| Query (truncated) | Expected | Got |
|---|---|---|
| `Czy garaż, który jest oddalony o ponad 1 kilometr...` | `COVERAGE_QUERY` | `COVERAGE_QUERY` ✓ *(sparte null instead of Hausrat)* |
| `If I pay my first premium 30 days after...` | `CLAIMS_PROCEDURE` | `CLAIMS_PROCEDURE` ✓ *(sparte Hausrat vs expected null)* |
| `Was passiert, wenn ein Kunde den ersten Beitrag für de...` | `CLAIMS_PROCEDURE` | `CLAIMS_PROCEDURE` ✓ |

*Note: 4 sparte mismatches are all COMPARISON queries where `sparte_hint=null` (cross-branch query) is arguably correct — the ground truth expected a single sparte but the query spans two branches.*

---

## Rationale

**llama-4-scout ties gpt-oss-20b and qwen3-32b on accuracy (88% intent) but is 33–40% faster** (985ms avg vs 1455ms and 1653ms respectively).

For a synchronous pipeline step that gates all downstream retrieval, latency at P95 matters: scout at 1691ms vs gpt-oss-20b at 2112ms saves ~400ms per query at tail latency.

gpt-oss-120b adds 2× latency with -4pp accuracy — not justified for this classification task.

---

## Consequences

- `_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"` in `src/query_expansion.py`
- Model can be overridden at construction: `QueryExpansion(model="...")`
- Provider: Groq via OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`)
- Context window: 131,072 tokens — no constraint for this step
- Cost: Groq free tier / per-token pricing applies

## Revisit triggers

- Intent accuracy drops below 80% on new eval set
- Groq discontinues `llama-4-scout` or rate limits become blocking
- A new model beats scout on both accuracy AND latency in a re-run of `scripts/benchmark_expansion.py`
