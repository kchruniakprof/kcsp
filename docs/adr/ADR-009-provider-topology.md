# ADR-009: LLM Provider Topology — Groq (Runtime) vs OpenRouter (Batch)

**Status:** Accepted  
**Date:** 2026-06-20  
**Deciders:** K. Chruniak  

---

## Context

Pipeline has two distinct LLM usage patterns:

**Runtime (synchronous, latency-sensitive):**
- `QueryExpansion` — gates all retrieval; P95 latency directly visible to user
- `LLMSelector` / `ContextSelector` — reranking within retrieval
- `Critic` — answer quality check

**Batch (offline, quality-sensitive, cost-gated):**
- `enrich_sections.py` — ~370 calls, run once per corpus revision

Two providers available: **Groq** (ultra-low latency, strict rate limits, free/cheap tier) and **OpenRouter** (routing layer, higher latency, access to large models, no strict RPM limits).

---

## Decision

| Path | Provider | Model |
|---|---|---|
| QueryExpansion | Groq | `meta-llama/llama-4-scout-17b-16e-instruct` (ADR-001) |
| LLMSelector / Critic | Groq | TBD at calibration |
| Batch enrichment | OpenRouter | `meta-llama/llama-3.3-70b-instruct` (ADR-004) |

**Provider per step must be independently configurable** — via `model_registry.py` / `llm_providers.py` pattern (port from DKV). No hardcoded provider strings in business logic.

---

## Rationale

Groq's differentiator is sub-second latency on modest models — exactly what runtime path needs. Mixing batch enrichment into Groq would exhaust RPM/TPM free-tier limits and degrade runtime quality.

OpenRouter for batch: no RPM pressure, access to 70b models for better DE generation quality, checkpoint/resume makes provider outage recoverable.

Separation also enables future A/B: swap one step's model without touching others. Provider abstraction (`llm_providers.py`) prevents coupling.

---

## Consequences

- `src/llm_providers.py` (port from DKV): factory functions `groq_client()`, `openrouter_client()` — thin wrappers returning `instructor`-wrapped OpenAI-compatible clients.
- `src/model_registry.py` (port from DKV): dict mapping step → `{provider, model_id, params}`.
- No step imports `groq` or `openrouter` SDK directly — always via registry/provider factory.
- `GROQ_API_KEY` + `OPENROUTER_API_KEY` both required in `.env`.
- Cost tracking: OpenRouter batch spend visible in OpenRouter dashboard; Groq runtime spend via Groq dashboard.

## Revisit triggers

- Groq introduces batch API with higher rate limits → consolidate providers.
- OpenRouter 70b latency becomes acceptable for runtime (<1s P50) → unify on one provider.
- New step added → assign to runtime (Groq) or batch (OpenRouter) bucket based on latency requirement.
