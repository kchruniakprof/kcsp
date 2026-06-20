# D6 — ragassistant: aktualizacja do nowego interfejsu retrievera

**Typ:** AFK  
**Blokowane przez:** D5  
**Dotyczy US:** 30  

---

## Co należy zbudować

Aktualizacja `src/ragassistant.py`: obsługa nowego interfejsu `Retriever` (D5) — dual-view `RetrievalResult`, abstain path z `ContextSelector`, wiring `CompositeDocFilter`.

**Zmiany:**

1. Konstruowanie `CompositeDocFilter` z `ProductDetectorAdapter` + `RareTagMatcherAdapter` (D2) i przekazywanie do `Retriever.retrieve_multi(doc_filter=...)`
2. Obsługa pustej listy wyników z retrievera (abstain od `ContextSelector`): ragassistant zwraca `FinalAnswer` z `abstained=True` i komunikatem standardowym (nie wywołuje generatora)
3. Generator dostaje `result.markdown` (verbatim) — nie `result.pruned_markdown`
4. Critic dostaje `result.pruned_markdown` jeśli pruner był aktywny — skrócony kontekst dla oceny

Kontrakt `FinalAnswer` bez zmian: `{answer_markdown, breadcrumb, doc_ids, section_types, abstained, audit}`.

---

## Kryteria akceptacji

- [ ] `RAGAssistant` konstruuje `CompositeDocFilter([ProductDetectorAdapter(...), RareTagMatcherAdapter(...)])` na starcie lub per-query
- [ ] `retrieve_multi` wywoływane z `doc_filter=composite_filter`
- [ ] Pusta lista wyników retrievera (abstain) → `FinalAnswer(abstained=True)` bez wywołania generatora ani critica
- [ ] Generator w promptcie: `result.markdown` (verbatim, pełny §)
- [ ] Critic w promptcie: `result.pruned_markdown` (skrócony widok); jeśli pruner bypass → `pruned_markdown == markdown`
- [ ] `FinalAnswer.answer_markdown` pochodzi wyłącznie z generatora operującego na `result.markdown` (whitelist nie zmieniona)
- [ ] Istniejące testy `tests/test_ragassistant.py` przechodzą (lub zaktualizowane)
- [ ] Nowy test: ragassistant z mockiem retriever zwracającym `[]` → `FinalAnswer.abstained == True`
- [ ] Nowy test: `answer_markdown` nie zawiera tekstu z `pruned_markdown` gdy różni się od `markdown` (verbatim guarantee end-to-end)

## Blokowane przez

- D5 (`retriever` z nowym interfejsem dual-view + `DocFilter` wiring)
