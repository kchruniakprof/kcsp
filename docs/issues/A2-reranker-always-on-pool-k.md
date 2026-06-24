# A2 — Reranker always-on + pool_k policy

> PRD: `docs/PRD-retriever-repair-streams-ABCD.md` — Stream A (runtime)
> Typ: AFK
> Faza: 1 (runtime, zero rebuild)

## Co należy zbudować

Dwie zmiany aktywujące `CrossEncoderReranker` w każdym wywołaniu produkcyjnym i ustawiające politykę `pool_k`:

**1. Reranker inject w provider**
`CrossEncoderReranker` jest już zbudowany (`src/retriever.py`), ale `reranker=None` w `promptfoo_provider.py` (linia ~66). Zmienić na inject `CrossEncoderReranker()` przy budowaniu `Retriever`. Lazy load modelu zachowany — nie obciąża importu.

**2. pool_k policy**
Obecne: `pool_k=20` hardcoded. Nowa polityka:
- Gdy `len(filtered_pool) ≤ 50` → `pool_k = len(filtered_pool)` (rerank cały pool)
- Gdy `len(filtered_pool) > 50` → `pool_k = 30`

Polityka implementowana wewnątrz `retrieve_multi` po DocFilter i soft-boost (po A1), przed wywołaniem reranker.

**Uwaga:** Reranker ocenia pary `(query, heading + markdown[:512])` — kontrakt `CrossEncoderReranker.rerank()` bez zmian.

## Kryteria akceptacji

- [ ] `promptfoo_provider.py`: `Retriever` budowany z `reranker=CrossEncoderReranker()`, nie `None`
- [ ] `retrieve_multi`: pool_k = `len(filtered)` gdy ≤50, `30` gdy >50
- [ ] Reranker lazy — model nie ładuje się przy imporcie ani `__init__`; ładuje się przy pierwszym `rerank()`
- [ ] Test: pool ≤50 → rerank cały pool (nie truncate do 20)
- [ ] Test: pool >50 → rerank tylko top-30 wg dense score
- [ ] Test: reranker zmienia kolejność wyników (mock CrossEncoder z odwróconymi scorami)
- [ ] `pytest tests/test_retriever.py` — zielony

## Blokowane przez

- A1 (ten sam `retrieve_multi` — soft-boost musi być na miejscu, kolejność operacji: boost → pool_k → rerank)
