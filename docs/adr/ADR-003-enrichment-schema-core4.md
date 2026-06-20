# ADR-003: Enrichment Schema — Core-4 Fields

**Status:** Accepted  
**Date:** 2026-06-20  
**Deciders:** K. Chruniak  

---

## Context

The DKV Belgium pattern (`data_preparation.ipynb`, `SectionDetails`) includes fields: `title`, `description`, `questions`, `topic_tags`, `practical_applications`, `key_insights`, and generic `tags`. Each field adds LLM cost and latency per section (~370 retrieval units).

We need to decide which fields to generate for KCSP.

### Field consumer analysis

| Field | DKV consumer | KCSP consumer |
|---|---|---|
| `title` | embedding | embedding |
| `description` | embedding | embedding |
| `questions` | embedding | embedding |
| `topic_tags` | label injection UI selector | Rare-tag Matcher (exact match) |
| `practical_applications` | UI label injection | **none** |
| `key_insights` | UI label injection | **none** |
| `tags` (generic) | search facets | **none** |

DKV had a UI selector component that consumed `practical_applications`, `key_insights`, and `tags` for label injection. KCSP pipeline has no such consumer — retriever is the only downstream.

---

## Decision

Generate exactly **Core-4 fields**: `title`, `description`, `questions` (5–10, embed 5), `topic_tags` (rare DE legal terms, verbatim).

Reject `practical_applications`, `key_insights`, generic `tags`.

---

## Rationale

Every extra field adds cost × 370 sections with zero retrieval benefit. Dead fields also bloat the Pydantic model, complicate prompt, and increase token usage per call.

`questions`: generate 5–10, store all in parquet, embed only first 5 — extra questions are a free semantic diversity reserve at query time without embedding cost.

`topic_tags`: kept despite being DE-specific because it gates the Rare-tag Matcher path (US-23 in PRD). Without it, exact-match pre-filter is impossible.

---

## Consequences

- `SectionDetails` pydantic model in `src/enrichment.py`: exactly 4 fields.
- `questions: List[str]` — validated length 5–10.
- `topic_tags: List[str]` — validated non-empty for sections with identifiable legal terms; allowed empty for preamble/general sections.
- Parquet schema: `title`(str), `description`(str), `questions`(List[str]), `topic_tags`(List[str]) — all enrichment columns.
- Embedding composition uses `title + description + questions[:5]` (see ADR-006).
- **Blocklist needed:** `topic_tags` smoke-test returned light generics (`Kraftfahrzeuge`, `Anhänger`). Prompt needs sharpening toward rare/specific terms; blocklist for generics at Rare-tag Matcher side (PRD US-23).

## Revisit triggers

- A new pipeline consumer appears (e.g. UI facets, citation metadata) → add field only when consumer exists.
- `questions` hit-rate analysis shows >5 needed for embedding → increase embed count.
