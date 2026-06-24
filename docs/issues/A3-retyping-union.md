# A3 — LLM retyping union w enrichmencie

> PRD: `docs/PRD-retriever-repair-streams-ABCD.md` — Stream A (rebuild)
> Typ: AFK
> Faza: 2 (aktywuje się przy rebuild enrichmentu po C1)

## Co należy zbudować

Zmiana logiki przypisywania `section_types` w `enrich_sections.py` z „LLM zastępuje keyword" na **union**: `section_types = keyword_types ∪ llm_types`.

**Motywacja:**
Przy miękkim filtrze (A1) over-labeling jest bezpieczny (niepasujący typ = brak boostu, nie kara). Under-labeling jest kosztowny (brak boostu dla poprawnie sklasyfikowanego chunku). Union asymetrycznie faworyzuje pokrycie.

**Mechanika:**
- `_assign_types(heading, markdown)` (z `hierarchy_parser.py`) produkuje keyword-based typy (istniejące).
- LLM enrichment produkuje listę typów multi-label (np. `["WHAT_IS_INSURED", "COVERAGE_AMOUNT"]`).
- Przy zapisie do parquet: `final_types = list(set(keyword_types) | set(llm_types))`.
- Gdy LLM zwróci pusty/null list → fallback do samych keyword_types (nie tracić keyword-signal).

**Uwaga:** Ten issue zmienia tylko `enrich_sections.py`. Efekt widoczny dopiero po re-run enrichmentu (po rebuild parquet z C1) i rebuild embeddings (D3). Kod można commitować wcześniej.

## Kryteria akceptacji

- [ ] `enrich_sections.py`: `section_types` = `set(keyword_types) | set(llm_types)` przy zapisie
- [ ] LLM null/empty → fallback do keyword_types (nie puste `[]`)
- [ ] Over-labeling nie powoduje błędu (miękki filtr A1 obsługuje)
- [ ] Test: chunk z keyword_type `SPECIAL_PROVISIONS` + LLM `WHAT_IS_INSURED` → union `["SPECIAL_PROVISIONS", "WHAT_IS_INSURED"]`
- [ ] Test: LLM null → `section_types = keyword_types` (nie puste)
- [ ] `pytest tests/test_enrich_sections.py` — zielony (lub nowy plik jeśli testy nie istnieją)

## Blokowane przez

- C1 (rebuild parquet wymagany przed re-run enrichmentu; kod można pisać równolegle z C1)
