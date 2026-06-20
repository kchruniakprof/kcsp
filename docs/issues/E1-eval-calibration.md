# E1 — Eval + kalibracja EMBED_THRESHOLD

**Typ:** HITL  
**Blokowane przez:** C1, D6  
**Dotyczy US:** 31–32  

---

## Co należy zbudować

Re-run `eval_full` po przebudowie retriever pipeline (nowe embeddingi z Core-4 + nowy retriever). Zmierzenie hit-rate vs baseline. Kalibracja `EMBED_THRESHOLD` na eval secie.

**Kroki:**

1. Uruchom `src/build_parquets.py` (A1) → `src/enrich_sections.py` (B1, cost-gate) → `src/build_embeddings.py` (C1) — pełny rebuild parquet
2. Uruchom `eval_full` (`promptfooconfig.full.yaml`) z nowym retriever pipeline (D1–D6)
3. Porównaj retrieval hit-rate z baseline (wynik przed rebuildem z `results_full.json`)
4. Kalibracja `EMBED_THRESHOLD`:
   - Zbierz `top_score` z `ContextSelector` dla wszystkich 100 pytań
   - Wykres: próg vs (abstain-rate na OUT_OF_SCOPE) + (hit-rate na pytaniach w-zakresie)
   - Wybierz próg gdzie abstain-rate na OUT_OF_SCOPE ≈ wysoki, hit-rate na w-zakresie spada < 5pp
5. Ustaw `EMBED_THRESHOLD` w `model_registry` / `.env`

**Decyzja człowieka (HITL):** wybór ostatecznej wartości progu wymaga oceny trade-off abstain vs coverage — nie jest to mechanicznie optymalizowalne bez znajomości priorytetu biznesowego (abstain > miss czy miss > abstain?).

---

## Kryteria akceptacji

- [ ] `eval_full` przebiega bez błędów na nowym pipeline (D1–D6 + C1)
- [ ] Retrieval hit-rate (retrieved_doc_ids accuracy) ≥ baseline z `results_full.json`
- [ ] Retrieval hit-rate na sekcjach z wypełnionymi Core-4 > hit-rate baseline (weryfikacja że enrichment pomógł)
- [ ] Zebrane `top_score` dla wszystkich 100 pytań; histogram / wykres progu
- [ ] `EMBED_THRESHOLD` wybrany i ustawiony w konfiguracji
- [ ] Abstain-rate na 5 pytaniach OUT_OF_SCOPE ≥ 80% po ustawieniu progu
- [ ] Hit-rate na pozostałych 95 pytaniach spada < 5pp vs wersja bez progu
- [ ] Wyniki zapisane w `results_full_enriched.json` (nowy plik, nie nadpisywać baseline)

## Blokowane przez

- C1 (enriched embeddings w parquet)
- D6 (pełny retriever pipeline z DocFilter + ContextSelector + dual-view)
