# D5 — retriever: refaktor — wiring DocFilter + ContextSelector + dual-view

**Typ:** AFK  
**Blokowane przez:** D2, D3, D4  
**Dotyczy US:** (refaktor retriever core)  

---

## Co należy zbudować

Refaktor `src/retriever.py`: usunąć inline filtr Sparte/Tarif, wpiąć `DocFilter` (D2), `ContextPruner` / `EmbeddingPruner` (D3), `ContextSelector` (D4). Zaktualizować `RetrievalResult` o dual-view.

**Zmiany w `Retriever`:**

1. Inline filtr `sparte/tarif` w `retrieve_multi` → usunięty; zastąpiony przez `CompositeDocFilter` przekazywany przy konstruowaniu lub wywołaniu
2. Kandydaci po DocFilter → przez `ContextPruner` → `PrunedChunk` lista
3. `PrunedChunk` lista → `ContextSelector.select()` → `SelectedChunk | Abstain`
4. Abstain → `retrieve_multi` zwraca `[]` (lub specjalny `AbstainResult`); ragassistant obsługuje pusty wynik

**`RetrievalResult` po refaktorze:**
- `markdown: str` — verbatim, nienaruszony (dla generatora i usera)
- `pruned_markdown: str` — widok dla LLM (może być krótszy niż `markdown`)
- Pozostałe pola bez zmian (`section_id`, `doc_id`, `sparte`, `tarif`, `heading`, `breadcrumb`, `score`, `section_types`, `topic_tags`)

**Pool retrievalu:** `Retriever.__init__` ładuje tylko wiersze `is_retrieval_unit=True` z parquet — L1-rodzice wykluczone z indeksu wektorowego.

---

## Kryteria akceptacji

- [ ] `Retriever.__init__` filtruje parquet do `is_retrieval_unit=True`; L1-rodzice nie wchodzą do `self._sections` ani `self._sec_embs`
- [ ] `retrieve_multi` nie zawiera inline kodu filtrującego `sparte`/`tarif`; zamiast tego przyjmuje `doc_filter: DocFilter | None`
- [ ] Gdy `doc_filter` przekazany: pre-filtruje kandydatów do doc_id z `doc_filter.filter(query)`
- [ ] `RetrievalResult.markdown` == oryginalny `markdown` z parquet (verbatim, nienaruszony)
- [ ] `RetrievalResult.pruned_markdown` może być krótszy (z `ContextPruner`)
- [ ] `EMBED_THRESHOLD=None` → zachowanie jak przed refaktorem (zawsze zwraca top_k, brak abstain)
- [ ] Istniejące testy `tests/test_retriever.py` przechodzą (lub zaktualizowane do nowego interfejsu)
- [ ] Nowy test: `result.markdown == parquet_markdown` dla każdego wyniku (verbatim guarantee)
- [ ] Nowy test: retriever załadowany z parquet gdzie L1-rodzic ma `is_retrieval_unit=False` → L1-rodzic NIE pojawia się w wynikach retrievalu

## Blokowane przez

- D2 (`doc_filter` — nowy filtr zastępujący inline)
- D3 (`ContextPruner` — dual-view chunki)
- D4 (`ContextSelector` — abstain path)
