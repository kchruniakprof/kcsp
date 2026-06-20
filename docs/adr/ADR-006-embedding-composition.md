# ADR-006: Embedding Text Composition

**Status:** Accepted  
**Date:** 2026-06-20  
**Deciders:** K. Chruniak  

---

## Context

Each retrieval unit needs a single embedding vector (BGE-M3, 1024-dim, mean-pooling). The text fed to the embedder ("embed-text") determines retrieval quality. Options:

**Pre-enrichment baseline (current):** `heading + markdown[:512]` — raw legal text, dense legalese, no questions.

**Post-enrichment options:**

| Component | Chars (approx) | Value |
|---|---|---|
| `heading` | ~30 | anchor, section code |
| `title` (enriched) | ~60 | human-readable summary |
| `description` (enriched) | ~200 | semantic digest |
| `questions[:5]` (enriched) | ~400 | query-side language |
| `body[:N]` (raw markdown) | N | legal verbatim anchor |

### body truncation: 2000 vs 400

Current baseline uses `body[:512]` (chars). DKV pattern (`embedder.py`, `_section_text`) used longer body segments. Issue specific to BGE-M3 with mean-pooling: long legalese dilutes the semantic vector — dense German legal clauses ("soweit", "gemäß", "unbeschadet") contribute high-frequency low-signal tokens that flatten the embedding toward a generic "insurance contract" centroid.

Enrichment (`title + description + questions`) carries the semantic load. Short body tail (400 chars) anchors the vector to specific clause structure without drowning the enrichment signal.

---

## Decision

**Embed-text composition:**

```
{heading}\n{title}\n{description}\n{q1}\n{q2}\n{q3}\n{q4}\n{q5}\n{body[:400]}
```

- `questions`: embed first 5; remaining (up to 10) stored in parquet only.
- `body[:400]`: character slice of raw `markdown` (after sanitization).
- All components joined with newline; missing components skipped (not replaced with placeholder).

---

## Rationale

`heading` = mandatory anchor (section identity). `title + description` = semantic digest replacing 2000-char legalese. `questions` = query-side language → dot-product alignment with user queries. `body[:400]` = verbatim clause hook prevents enrichment hallucination from fully decoupling vector from actual text.

400 chars ≈ 2–3 sentences. Enough for clause specificity; not enough to dilute mean-pool vector with boilerplate.

Embed only 5 questions: BGE-M3 context window is 8192 tokens — not the constraint. Quality constraint: questions 6–10 (generated last) tend to be lower-specificity; storing them preserves optionality for future query-time expansion.

---

## Consequences

- `build_embeddings.py`: `_embed_text(row)` function assembles components in this order; unit-tested.
- Re-embedding requires re-running `build_embeddings.py` only (enrichment not re-run — see ADR-007).
- BGE-M3 `max_length=8192` tokens — composition well within limit; no truncation needed at model level.
- Evaluation (Fase E): compare retrieval hit-rate vs pre-enrichment baseline to validate 400-char cutoff.

## Revisit triggers

- Eval shows hit-rate regression vs baseline → try `body[:800]` or `body[:1200]`.
- BGE-M3 replaced by cross-encoder or late-interaction model → embed-text composition may change entirely.
