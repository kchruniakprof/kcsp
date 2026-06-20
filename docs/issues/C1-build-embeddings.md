# C1 — build_embeddings: nowy skład embed-text

**Typ:** AFK  
**Blokowane przez:** B1  
**Dotyczy US:** 13–16  

---

## Co należy zbudować

Nowy skrypt `src/build_embeddings.py`: ładuje wzbogacone parquet, dla każdej jednostki `is_retrieval_unit=True` komponuje tekst embeddingu z pól Core-4 + body, koduje przez BGE-M3, zapisuje kolumnę `embedding` z powrotem do parquet.

**Skład embed-text (ADR-006):**
```
{heading}
{title}
{description}
{question_1}
{question_2}
{question_3}
{question_4}
{question_5}
{markdown[:400]}
```
Komponenty łączone `\n`. Brakujące pola pomijane (nie zastępowane placeholderem). `questions[:5]` — jeśli jest mniej niż 5, użyj tyle ile jest.

Skrypt re-runnable bez re-enrichmentu: czyta gotowe Core-4 z parquet, nie wywołuje OpenRouter.

---

## Kryteria akceptacji

- [ ] Nowy skrypt `src/build_embeddings.py` działa niezależnie od `build_parquets.py` i `enrich_sections.py`
- [ ] Iteruje tylko po `is_retrieval_unit=True`; L1-rodzice nie wchodzą do indeksu wektorowego
- [ ] Embed-text złożony z: heading + title + description + questions[:5] + markdown[:400] (kolejność i separacja `\n`)
- [ ] Brakujące pola Core-4 (null/None) → pomijane, nie powodują błędu; sekcja bez `title` mimo to ma embed-text z pozostałych pól
- [ ] `markdown[:400]` — dokładnie 400 znaków (nie tokenów); dla krótszych `markdown` — cały tekst
- [ ] BGE-M3 (`BAAI/bge-m3`) z `normalize_embeddings=True`; shape `(N, 1024)`, normy ≈ 1.0 (walidacja)
- [ ] Kolumna `embedding` zapisana do `sections.parquet` (L1-liście) i `subsections.parquet` (L2)
- [ ] L1-rodzice (`is_retrieval_unit=False`) → `embedding=None/NaN` (nie blokują walidacji)
- [ ] Testy TDD dla `_embed_text(row)`:
  - pełne Core-4 → tekst zawiera heading, title, desc, 5 pytań, body[:400]
  - brak `title` → embed-text nie zawiera placeholdera, nie crashuje
  - `markdown` ≥ 400 znaków → body fragment ma dokładnie 400 znaków
- [ ] Walidacja kształtu i norm embeddingów po build (assert shape, assert allclose norms)

## Blokowane przez

- B1 (parquet musi mieć wypełnione Core-4 dla `is_retrieval_unit=True`)
