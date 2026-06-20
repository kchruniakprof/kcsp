# ADR-004: Enrichment Engine — Provider, Model, Client

**Status:** Accepted  
**Date:** 2026-06-20  
**Deciders:** K. Chruniak  

---

## Context

~370 retrieval units need LLM enrichment (Core-4 fields, ADR-003). Each call generates structured German output (title, description, 5–10 questions, topic_tags). Requirements:

- Structured output (Pydantic validation + retry on malformed JSON)
- Resumable / checkpoint (batch operation, cost-gated)
- DE idiomatic output — especially for `questions` field simulating real policyholder queries
- Cost-controlled (batch, not real-time path)

### Provider options evaluated

**Groq** — ultra-low latency, used for runtime path (QueryExpansion, selector, critic). Free tier rate limits are strict; batch enrichment with ~370 calls × retries may hit TPM/RPM limits. Groq is the wrong tool for offline batch.

**OpenRouter** — routing layer, access to large models, designed for batch/async use cases. `meta-llama/llama-3.3-70b-instruct` available.

### Model: 70b vs 8b

8b models (e.g. `llama-3.1-8b-instant`) produce lower-quality idiomatic DE questions — shorter, less varied, miss legal nuance. 70b produces significantly better DE questions for insurance domain. Enrichment runs once (checkpoint/resume), so latency per call is not critical. Cost difference justified by one-time run and quality delta on `questions` field.

---

## Decision

**Provider:** OpenRouter (`https://openrouter.ai/api/v1`)  
**Model:** `meta-llama/llama-3.3-70b-instruct`  
**Client:** `instructor` (Pydantic + `Mode.JSON` + `max_retries=3`)  
**Pattern:** checkpoint/resume/skip-done over ~370 retrieval units

Smoke-test (2026-06-20): call returned correct DE title, description, 5 questions, tags. ✅

---

## Rationale

OpenRouter separates batch enrichment from Groq runtime — no rate-limit contention. `instructor` gives Pydantic validation + automatic retry on malformed output, consistent with DKV pattern and ADR-001 (QueryExpansion). 70b > 8b for idiomatic German question generation; one-time cost is acceptable.

Provider topology: **Groq = runtime** (latency-sensitive); **OpenRouter = batch** (quality-sensitive). See ADR-009.

---

## Consequences

- `OPENROUTER_API_KEY` required in `.env` (already present; rotate — key was exposed in chat 2026-06-20).
- `src/enrichment.py`: `openrouter_client()` returns `instructor.from_openai(openai.OpenAI(base_url=..., api_key=OPENROUTER_API_KEY))`.
- `enrich_sections.py`: checkpoint file (JSON/parquet) tracks `section_code` → skip already-enriched on resume.
- Model overrideable via env var or constructor param (per `model_registry.py` pattern from DKV).
- Re-enrichment requires explicit flag (default: skip-done) — protects against accidental re-spend.

## Revisit triggers

- OpenRouter discontinues `llama-3.3-70b-instruct` → re-benchmark DE quality on available 70b alternatives.
- Groq raises batch rate limits significantly → consider consolidating providers.
- `topic_tags` quality remains poor after prompt iteration → try `qwen3-32b` or `llama-4-maverick` for this field only.
