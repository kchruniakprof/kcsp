# ADR-005: Retrieval Unit Granularity — Leaf Nodes Only

**Status:** Accepted  
**Date:** 2026-06-20  
**Deciders:** K. Chruniak  

---

## Context

Corpus hierarchy:
- **L1 sections** (e.g. "§ A Kfz-Haftpflichtversicherung"): 205 total
  - **53 L1 parents** — have child subsections; their `markdown` = concatenated child text
  - **152 L1 leaves** — no children; standalone atomic content
- **L2 subsections** (e.g. "§ A.1"): 218 total; always leaves

Total retrieval candidates: 205 + 218 = 423 rows in parquet.

### Problem with including L1 parents

Verified (2026-06-20): L1 "A" len=2092 ≈ sum(A.1 + A.2 + A.3)=2062. L1 parent `markdown` is a concatenation of its children. Including parents in the retrieval pool causes:

1. **Duplicate embeddings** — same text appears as parent + N children → top_k saturated with duplicates; real coverage shrinks.
2. **Ambiguous citation** — which `§` do we cite? Parent "§ A" or child "§ A.1"? Verbatim citation guarantee (ADR-008) requires exactly one source per chunk.
3. **Wasted embed slots** — 53 duplicate vectors crowd out genuinely different sections.

---

## Decision

**Retrieval unit = leaf node.**

Pool = **152 L1-leaves** ∪ **218 L2** = **370 units**.

**53 L1-parents excluded from retrieval pool and embedding**, but retained in parquet for breadcrumb construction (`heading`, `section_code` chain for citation context).

New boolean column **`is_retrieval_unit`** computed deterministically in `build_parquets.py`:

```python
is_retrieval_unit = (level == "L2") or (level == "L1" and has_no_children)
```

Retriever and embedder filter on `is_retrieval_unit == True` only.

---

## Rationale

L1 parents carry zero information beyond their children. Their inclusion dilutes top_k and breaks the single-source citation invariant. Deterministic flag means no LLM call needed; rebuild is cheap and reproducible.

Parquet files stay split (`sections.parquet` / `subsections.parquet`) — joins use `parent_section_id` from subsections; breadcrumb traverses via `section_code`.

---

## Consequences

- `build_parquets.py`: add `is_retrieval_unit` column; verify with assert: `sum(is_retrieval_unit) == 370` (±tolerance for corpus updates).
- `src/retriever.py`: load only rows where `is_retrieval_unit == True` for embedding index.
- `enrich_sections.py`: iterate only `is_retrieval_unit == True` rows — saves ~53 × LLM call cost.
- `build_embeddings.py`: filter on `is_retrieval_unit` before computing embed-text.
- L1-parent rows present in parquet with `is_retrieval_unit=False`; `title/description/questions/topic_tags` may be NULL (no enrichment needed).

## Revisit triggers

- Corpus restructured so L1 parents contain unique content (not just child concatenation).
- Evaluation shows missed coverage attributable to a specific L1 parent's unique intro text → consider splitting intro paragraph as separate leaf.
