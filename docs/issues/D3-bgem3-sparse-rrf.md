# D3 — BGE-M3 sparse+dense + RRF

> PRD: `docs/PRD-retriever-repair-streams-ABCD.md` — Stream D (rebuild)
> Typ: AFK
> Faza: 2 (wymaga rebuild embeddings po C1+A3)

## Co należy zbudować

Rozszerzenie embeddera o sparse component (leksykalny) i fuzja wyników dense+sparse przez RRF. Zastępuje `sentence-transformers` → `FlagEmbedding` (`BGEM3FlagModel`). Naprawia defense-in-depth dla Q5 i przyszłych novel-compound terminów DE.

**`build_embeddings.py` — swap embeddera**
`BGEM3FlagModel` z `FlagEmbedding` (pip: `FlagEmbedding`), wywołanie:
```python
# z prototypu — koduje decyzję single-pass
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
output = model.encode(texts, return_dense=True, return_sparse=True, batch_size=12)
dense_embs = output["dense_vecs"]      # shape (N, 1024), float32
sparse_embs = output["lexical_weights"]  # list[dict[int, float]] — token_id: weight
```

Przechowywać osobno: `dense_embeddings.npy` (jak dotąd) + `sparse_embeddings.pkl` (list[dict]).
Lazy load model — nie przy imporcie.

**`retriever.py` — RRF fusion w `retrieve_multi`**
Po soft-boost (A1), zastąpić lub rozszerzyć scoring:
1. Dense rank: `cand_embs @ q_dense.T` → posortuj → `dense_rank[i]`
2. Sparse rank: dot-product sparse query × sparse chunks → `sparse_rank[i]`
3. RRF score: `1/(k + dense_rank[i]) + 1/(k + sparse_rank[i])` gdzie `k=60` (standard)
4. Boost `+0.04` (A1) aplikowany na RRF score, nie na raw cosine

Sparse query: `BGEM3FlagModel.encode([query], return_sparse=True)["lexical_weights"][0]`.

**Uwaga:** BM25 **odrzucony** — wymaga dekompozytora kompozycji DE, kruchy na novel compoundy. Sparse BGE-M3 działa sub-word, nie wymaga external tokenizacji.

**Instalacja:** `pip install FlagEmbedding` (dodać do `requirements.txt`).

## Kryteria akceptacji

- [ ] `build_embeddings.py` używa `BGEM3FlagModel` z `return_dense=True, return_sparse=True`
- [ ] Dense i sparse embeddingi przechowywane osobno (`.npy` + `.pkl` lub inny format)
- [ ] `Retriever.__init__` ładuje oba typy embeddingów gdy dostępne
- [ ] `retrieve_multi`: RRF fusion z `k=60`, wynik RRF zastępuje raw cosine jako primary rank
- [ ] Boost `+0.04` (A1) aplikowany na RRF score
- [ ] Gdy sparse embeddingi niedostępne (legacy) → fallback do samego dense (backward-compat)
- [ ] `BGEM3FlagModel` lazy load — nie ładuje się przy imporcie
- [ ] `requirements.txt` zawiera `FlagEmbedding`
- [ ] Test: RRF function jednostkowa: poprawny wzór `1/(k+r)`, suma dense+sparse
- [ ] Test: chunk z wysokim sparse score ale niskim dense → awansuje w RRF vs. samo dense
- [ ] `pytest tests/test_retriever.py` — zielony

## Blokowane przez

- C1 (nowe RU z preambuł/FreeText muszą mieć embeddingi)
- A3 (retyping — enrichment po C1 musi być skończony przed build_embeddings)
- Rebuild kolejność: `build_parquets (C1) → enrich_sections (A3) → build_embeddings (D3)`
