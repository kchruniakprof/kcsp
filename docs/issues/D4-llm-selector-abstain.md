# D4 — llm_selector: ContextSelector + abstain-on-low-score

**Typ:** AFK  
**Blokowane przez:** D1, D2, D3  
**Dotyczy US:** 21–24  

---

## Co należy zbudować

Nowy moduł `src/llm_selector.py` portowany z DKV `reranker_strategy.py` + `llm_selector.py`, z **krytyczną adaptacją KCSP**: `confidence < EMBED_THRESHOLD` → abstain (nie BruteForce fallback jak w DKV).

**`ContextSelector`** — główna klasa:
- Primary: `TopKReranker` — LLM ocenia kandydatów z DocFilter; zwraca top chunk z `confidence` score
- Ścieżka low-confidence: `confidence < EMBED_THRESHOLD` → abstain (sygnał dla ragassistant że nie ma dobrej odpowiedzi)
- `EMBED_THRESHOLD`: czytany z `model_registry` / env var; `None` = brak abstain (domyślnie przed kalibracją Fazy E)
- Provider: Groq (via `groq_client()` z D1); model z `model_registry["llm_selector"]`

**Różnica vs DKV:** DKV przy `confidence < próg` → `BruteForceReranker` nad całym korpusem. KCSP → abstain. Powód: w domenie ubezpieczeniowej niska pewność = ryzyko halucynacji; lepiej "nie wiem" niż zgadywanie.

Wejście do `ContextSelector`: lista `PrunedChunk` (z D3; ma `pruned_text` dla LLM i `verbatim_text` dla usera). Selektor używa `pruned_text` — nigdy `verbatim_text`.

---

## Kryteria akceptacji

- [ ] `ContextSelector.select(candidates: list[PrunedChunk]) → SelectedChunk | Abstain`
- [ ] `confidence >= EMBED_THRESHOLD` → `SelectedChunk` z `verbatim_text` nienaruszonym
- [ ] `confidence < EMBED_THRESHOLD` → `Abstain` (nie rzuca wyjątku; ragassistant obsługuje)
- [ ] `EMBED_THRESHOLD = None` → nigdy abstain (zachowanie jak dotychczas; bezpieczne przed kalibracją)
- [ ] Selektor używa `pruned_text` z `PrunedChunk` przy wywołaniu LLM; `verbatim_text` przepuszcza nienaruszony do `SelectedChunk`
- [ ] Provider: Groq via `groq_client()` (D1); model z `REGISTRY["llm_selector"]` (D1)
- [ ] Testy: mock Groq → fixture confidence wysoki → `SelectedChunk`; fixture confidence niski + `EMBED_THRESHOLD` ustawiony → `Abstain`; `EMBED_THRESHOLD=None` → zawsze `SelectedChunk`
- [ ] Test verbatim: `selected.verbatim_text == original_markdown` niezależnie od confidence

## Blokowane przez

- D1 (provider factory + model_registry)
- D2 (DocFilter — wejście do selectora pochodzi z przefiltrowanego retrievalu)
- D3 (PrunedChunk — interfejs wejściowy selectora)
