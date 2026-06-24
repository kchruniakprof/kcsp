# G5 — CrossEncoderReranker (two-stage retrieval)

> PRD: `docs/PRD-product-scope-retrieval.md` — sekcja C
> Typ: AFK
> Zablokowane przez: brak — start natychmiast (równolegle z G1+G2)

## Co należy zbudować

Dwa etapy retrieval zamiast jednego: bi-encoder zwraca pool top-`pool_k`=20 kandydatów → cross-encoder reranker sortuje → top `top_k`=5 zwracane.

**Nowy komponent `CrossEncoderReranker`** (analogicznie do `ContextPruner` — optional inject):
- Lazy load modelu przy pierwszym wywołaniu (nie przy import / `__init__`)
- `rerank(query, results) -> list[RetrievalResult]` — scores pairs (query, heading + markdown[:512]), zwraca posortowane desc
- Model: `REGISTRY["reranker"]` = `"BAAI/bge-reranker-v2-m3"`

**Zmiany w `Retriever`:**
- Nowy parametr `reranker: Optional[CrossEncoderReranker] = None` w `__init__`
- Nowy parametr `pool_k: int = 20` w `retrieve_multi` (gdy `reranker` nie None i `pool_k > top_k`)
- Gdy reranker inject: bi-encoder zwraca pool_k → reranker → slice top_k
- Gdy reranker=None (default): zachowanie bez zmian (backward-compat)

**`model_registry.py`:**
- Nowy wpis: `"reranker": "BAAI/bge-reranker-v2-m3"`

## Kryteria akceptacji

- [ ] `CrossEncoderReranker` klasa z lazy load — import retriever.py nie ładuje modelu
- [ ] `rerank(query, results)` zwraca poprawną kolejność (mock cross-encoder scores w testach)
- [ ] `Retriever.__init__` przyjmuje `reranker=None` (opcjonalny)
- [ ] `retrieve_multi` z `pool_k=20`, `top_k=5`: zwraca ≤5 wyników po rerankingu
- [ ] Bez reranker (default): zachowanie identyczne z obecnym (testy nie psują się)
- [ ] `model_registry.py` zawiera `"reranker"` wpis
- [ ] Testy `test_retriever.py`: reranker zmienia kolejność wyników (mock); pool_k > top_k → slice; lazy load (patch); backward-compat bez reranker
- [ ] `pytest tests/test_retriever.py` — zielony

## Blokowane przez

Brak — można rozpocząć natychmiast.
