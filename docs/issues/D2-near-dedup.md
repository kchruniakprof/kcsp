# D2 — Near-duplicate dedup + shared_tarifs

> PRD: `docs/PRD-retriever-repair-streams-ABCD.md` — Stream D (runtime)
> Typ: AFK
> Faza: 1 (runtime, zero rebuild)

## Co należy zbudować

Kolaps prawie-identycznych chunków w pool przed rerankingiem. Eliminuje zalew top-5 przez 4 prawie-identyczne sekcje Hausrat (~95% overlap między dokumentami różnych taryf).

**Mechanika:**
- Po force-include (D1), przed reranker (A2): dla chunków w final_pool oblicz parami emb-cosine.
- Chunki z cosine > 0.98 → kolaps w jeden reprezentant: ten z najwyższym `boosted_score` (lub z taryfy z aktywnego gate gdy tie).
- Reprezentant dostaje dodatkowe pole `shared_tarifs: list[str]` = lista `tarif` wszystkich kolapsowanych chunków.
- Próg `0.98` przechowywany w `REGISTRY["dedup_threshold"]` — tunable bez zmiany kodu.

**Uwaga:** Dedup operuje na embeddingach **już obliczonych** w tej samej sesji `retrieve_multi` (embs dla kandydatów z `cand_embs`). Brak dodatkowego encode.

**shared_tarifs** dostępne w `RetrievalResult` — LLM COMPARE może wykorzystać informację, że chunk pochodzi z wielu taryf.

## Kryteria akceptacji

- [ ] `REGISTRY["dedup_threshold"]` = `0.98` (domyślny, tunable)
- [ ] Chunki w pool z emb-cosine > próg → kolaps; reprezentant = max `boosted_score`
- [ ] `RetrievalResult` ma pole `shared_tarifs: list[str]` (puste dla nie-zdeduplikowanych)
- [ ] Dedup zachodzi po force-include, przed reranker
- [ ] Test: 4 prawie-identyczne chunki (cosine 0.99) → 1 reprezentant w pool z `shared_tarifs` = 4 taryfy
- [ ] Test: chunki z cosine 0.95 (poniżej progu) → nie kolapsowane
- [ ] Test: reprezentant to ten z wyższym score (nie pierwszy w kolejności)
- [ ] `pytest tests/test_retriever.py` — zielony

## Blokowane przez

- D1 (exact-term force-include musi być na miejscu; dedup operuje na final_pool po force)
