# A1 — build_parquets: dodaj `is_retrieval_unit`, usuń embedding

**Typ:** AFK  
**Blokowane przez:** Brak — można rozpocząć natychmiast  
**Dotyczy US:** 1–5  

---

## Co należy zbudować

Refaktor `src/build_parquets.py`: dodać deterministyczną kolumnę `is_retrieval_unit` i usunąć obliczanie embeddingów (przeniesione do osobnego etapu C1).

Kolumna `is_retrieval_unit` obliczana czysto na podstawie hierarchii parquet — bez LLM, zero koszt:
- L2 (subsekcje) → zawsze `True`
- L1 bez dzieci (liście) → `True`  
- L1 z dziećmi (rodzice) → `False`

L1-rodzice **pozostają** w parquet z `is_retrieval_unit=False` — potrzebne do budowania breadcrumb. Nie są usuwane.

Embeddingi: cały blok ładowania modelu i `embedder.encode()` wychodzi z tego skryptu — przeniesiony do `build_embeddings.py` (C1). Kolumna `embedding` nie jest emitowana przez ten skrypt.

---

## Kryteria akceptacji

- [ ] Parquet zawiera kolumnę `is_retrieval_unit` (dtype bool) na każdym wierszu w `sections.parquet` i `subsections.parquet`
- [ ] Reguła: L2 → `True`; L1 z subsekcjami → `False`; L1 bez subsekcji → `True`
- [ ] `sum(is_retrieval_unit)` po build ≈ 370 (152 L1-liście + 218 L2); walidacja w `_validate()` z tolerancją ±5
- [ ] Sekcja Kfz `section_code=="A"` (ma subsekcje A.1, A.2...) → `is_retrieval_unit=False`
- [ ] `sections.parquet` NIE zawiera kolumny `embedding` po refaktorze
- [ ] `build_parquets.py` nie importuje `SentenceTransformer`
- [ ] Istniejące testy w `tests/test_build_parquets.py` przechodzą
- [ ] Nowe testy TDD dla `is_retrieval_unit` w `tests/test_build_parquets.py`:
  - L1 z subsekcjami → `False`
  - L1 bez subsekcji → `True`
  - L2 → `True`
  - łączna liczba `True` w parquet ≈ 370

## Blokowane przez

Brak — można rozpocząć natychmiast.
