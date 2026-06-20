# ADR-007: Build Pipeline — Three-Stage Separation

**Status:** Accepted  
**Date:** 2026-06-20  
**Deciders:** K. Chruniak  

---

## Context

Current `build_parquets.py` parses markdown, builds parquet, AND computes embeddings in a single pass. This couples three concerns with very different cost profiles:

| Stage | Cost | Idempotency | Trigger |
|---|---|---|---|
| Parse markdown → parquet | ~0 $ | Deterministic | Corpus update |
| LLM enrichment (Core-4) | ~$X per 370 calls | Non-deterministic | Once, or model change |
| Embed-text → vectors | Model inference only | Deterministic given compose fn | Compose fn change, model change |

Monolithic script prevents: re-embedding without re-enriching, re-parsing without losing enrichment, cost-gating enrichment independently.

---

## Decision

Split into **three independent scripts**:

1. **`build_parquets.py`** (refactored) — sanitize → parse → emit parquet + `is_retrieval_unit`. Zero LLM calls. Zero $ cost.
2. **`enrich_sections.py`** (new) — load parquet → Core-4 via OpenRouter → write back. Checkpoint/resume/skip-done. **Explicit go required from user** (cost gate per PRD).
3. **`build_embeddings.py`** (new/extracted) — load enriched parquet → compose embed-text (ADR-006) → compute BGE-M3 vectors → write `embedding` column.

Dependency chain: 1 → 2 → 3. Each stage is independently re-runnable without re-running upstream if inputs unchanged.

---

## Rationale

**Re-embed without re-enrich:** embedding composition formula may be tuned (e.g. body slice length) without paying LLM cost again. Stage 3 re-runs in minutes; Stage 2 costs money.

**Re-enrich without re-parse:** corpus structure is stable; if enrichment prompt is revised, only Stage 2 re-runs.

**Re-parse without losing enrichment:** if `md_sanitizer` / `hierarchy_parser` are patched, Stage 1 re-runs; Stage 2 checkpoint skips already-enriched rows.

**LLM ≠ deterministic parse:** mixing them in one script makes testing harder — Stage 1 is unit-testable with no mocks; Stage 2 requires integration test with API key.

---

## Consequences

- `build_parquets.py`: remove embedding computation; add `is_retrieval_unit`; remains pure pandas/parse.
- `enrich_sections.py`: new script; checkpoint file at `parquet/enrichment_checkpoint.json` (section_code → enriched bool).
- `build_embeddings.py`: new script; reads `sections.parquet` + `subsections.parquet`, filters `is_retrieval_unit`, writes `embedding` column back.
- CI/make target order: `build_parquets → enrich_sections → build_embeddings`.
- Cost gate: `enrich_sections.py` prints estimated cost before starting and prompts `[y/N]` unless `--yes` flag passed.

## Revisit triggers

- Corpus update rate increases → consider incremental parquet update (append-only) instead of full rebuild.
