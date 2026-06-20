# ADR-002: Enrichment Language Strategy

**Status:** Accepted  
**Date:** 2026-06-20  
**Deciders:** K. Chruniak  

---

## Context

Enrichment adds `title`, `description`, `questions`, `topic_tags` to each retrieval unit. These fields are consumed exclusively by the retrieval pipeline (embedding + Rare-tag Matcher) — they never surface verbatim to the user. Two language choices exist for LLM prompting:

- **DE prompt + DE output** — consistent, but LLM instruction-following degrades in German; risk of paraphrase drift away from legal DE terminology.
- **EN prompt + DE output** — LLM understands task in its strongest language, but emits German domain content → legal terms stay verbatim in DE.

Cross-lingual retrieval (Polish/English queries against DE corpus) is a separate concern, addressed by model selection (BGE-M3 multilingual) and QueryExpansion→DE normalization — NOT by translating the corpus.

---

## Decision

**System prompt in English; all enrichment field output in German.**

`topic_tags` must be verbatim DE legal terms (e.g. `Kraftfahrzeuge`, `Haftpflichtschaden`) — exact-match downstream in Rare-tag Matcher means any translation or paraphrase breaks matching.

Cross-lingual gap is closed by:
1. BGE-M3 (multilingual embedding model — same vector space for PL/EN queries and DE text)
2. `QueryExpansion` normalizing user query to German before embedding

Corpus translation is **not** an option — it would break verbatim citation guarantee (ADR-008).

---

## Rationale

Enrichment fields are retrieval-only (whitelist). "Loss of nuance in translation" applies to user-facing answers, not to these fields. EN prompt → better LLM task compliance; DE output → zero terminology drift; BGE-M3 → cross-lingual retrieval without corpus translation.

---

## Consequences

- `enrich_sections.py` system prompt in English with explicit DE output instruction per field.
- `topic_tags` validated: must be German strings, no English mixed in.
- BGE-M3 remains the mandatory embedding model (multilingual); switching to monolingual DE model invalidates this strategy.
- QueryExpansion `normalized_query` must produce German strings (enforced by existing ADR-001 prompt design).

## Revisit triggers

- BGE-M3 replaced by monolingual model → cross-lingual strategy needs rethink.
- Rare-tag Matcher evolves beyond exact match (fuzzy/stemmed) → `topic_tags` language constraint relaxes.
