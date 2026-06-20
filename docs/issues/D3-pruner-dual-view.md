# D3 — Pruner: context_pruner + embedding_pruner z dual-view

**Typ:** AFK  
**Blokowane przez:** D1  
**Dotyczy US:** 25–27  

---

## Co należy zbudować

Dwa nowe moduły portowane z wzorca DKV (`context_pruner.py`, `embedding_pruner.py`), z **krytyczną adaptacją KCSP**: pruner NIE modyfikuje verbatim cytatu — produkuje dwa widoki tego samego chunku.

**Dual-view output** — każdy pruner zwraca obiekt z:
- `verbatim_text: str` — oryginalne `markdown` sekcji, nienaruszony
- `pruned_text: str` — skrócony widok dla LLM (selector, critic)

`verbatim_text` idzie do generatora i usera (whitelist). `pruned_text` idzie tylko do wewnętrznych kroków LLM. Nigdy nie są zamieniane.

**`src/context_pruner.py`** — zdaniowy pruning (port DKV):
- Global bypass: chunk < 2500 znaków → `pruned_text == verbatim_text` (nie przetwarza)
- Empty-guard: jeśli wynik pruning jest pusty → fallback do `verbatim_text`
- Pruning działa na kopii tekstu — `verbatim_text` nigdy nie jest modyfikowany

**`src/embedding_pruner.py`** — embedding-based pruning (port DKV):
- Analogiczne dual-view + bypass + empty-guard
- Używa BGE-M3 do oceny zdań; nie re-ładuje modelu jeśli już załadowany

---

## Kryteria akceptacji

- [ ] `ContextPruner.prune(chunk) → PrunedChunk` gdzie `PrunedChunk.verbatim_text == original_markdown` zawsze
- [ ] `ContextPruner.prune(chunk)` gdzie `len(chunk) < 2500` → `pruned_text == verbatim_text`
- [ ] `ContextPruner.prune(chunk)` gdzie wynik pruning pusty → `pruned_text == verbatim_text` (empty-guard)
- [ ] `EmbeddingPruner` analogiczne zachowanie (dual-view, bypass, empty-guard)
- [ ] Testy TDD w `tests/test_context_pruner.py`:
  - `result.verbatim_text == original_markdown` dla długiego chunku (pruner skrócił `pruned_text`)
  - `result.pruned_text == result.verbatim_text` dla chunku < 2500 znaków (bypass)
  - `result.verbatim_text == original_markdown` po empty-guard
- [ ] Żaden z prunerów nie importuje `openai` / `instructor` bezpośrednio (D1 pattern)
- [ ] `src/retriever.py` NIC nie zmienione w tym issue (refaktor = D5)

## Blokowane przez

- D1 (`llm_providers` / `model_registry` — infrastruktura wspólna)
