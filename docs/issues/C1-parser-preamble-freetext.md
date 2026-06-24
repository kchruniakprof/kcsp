# C1 — Parser: preambuł L1 jako RU + FreeText bloki

> PRD: `docs/PRD-retriever-repair-streams-ABCD.md` — Stream C
> Typ: AFK
> Faza: 2 (wymaga rebuild: build_parquets → enrich → build_embeddings)

## Co należy zbudować

Dwie chirurgiczne reguły parsera eliminujące utratę treści merytorycznej spoza schematu numeracji. Naprawia Q1 (Safe Drive), Q2 (EV-Wechselprämie), Q7 (Wallbox).

**Reguła 1 — Preambuł L1 jako osobny RU**
Gdy sekcja L1 ma dzieci (subsections) ORAZ jej `body` (tekst przed pierwszym dzieckiem) po `strip_noise` zawiera ≥200 zn. LUB zawiera listę (linia zaczynająca się od `-` lub cyfry) LUB zawiera pełne zdanie (`.` po ≥30 zn.) → emituj preambuł jako osobny `Section` z `is_retrieval_unit=True`.

Atrybuty preambuł RU:
- `section_code`: `{parent_code}.0` (np. `1.0`, `A.0`)
- `heading`: `Vorbemerkung` (lub pierwszy nie-pusty nagłówek preambuł jeśli istnieje)
- `breadcrumb`: `{parent_breadcrumb} > §{code} Vorbemerkung`
- `level`: 2 (traktowany jak L2, dziecko L1)
- `parent_section_id`: sekcja L1 rodzicem

**Reguła 2 — Bloki `## FreeText`**
Nagłówek `## FreeText` po normalizacji: wykryj marker załącznika regex `Anhang|Sonderbedingungen|Besondere Bedingungen` w tekście **bezpośrednio następującym** po nagłówku FreeText (pierwsze 200 zn.).

- **Z markerem** → emituj jako pseudo-L1 (level=1) z:
  - `section_code`: `ANH-{slug}` gdzie slug = pierwsze 3 słowa markera znormalizowane (lowercase, `-`)
  - `heading`: pełny tekst markera (np. „Sonderbedingungen Safe Drive")
  - Pod-`##` jako L2 dzieci tej pseudo-L1
  - `is_retrieval_unit=True` dla liści (L2) lub dla samej pseudo-L1 gdy brak L2

- **Bez markera** → emituj jako L2 dziecko ostatniej poprzedzającej L1; `is_retrieval_unit=True`

**build_parquets.py — aktualizacja zakresu walidacji**
`_validate` zaktualizować zakres `_MIN_RU` / `_MAX_RU` z ~280 na 350–420 (rebuild doda ~70–140 nowych RU z preambuł i FreeText).

**TDD regresja:**
- Q1: `ANH-sonderbedingungen-safe-drive` → `is_retrieval_unit=True`, breadcrumb zawiera „Safe Drive"
- Q2: blok EV-Wechselprämie w §E → L2 dziecko §E, `is_retrieval_unit=True`
- Q7: preambuł §1 Smart → `section_code=1.0`, `is_retrieval_unit=True`, markdown zawiera „Wallboxen"

## Kryteria akceptacji

- [ ] Preambuł L1 z body ≥200 zn. → osobny RU `{code}.0` z breadcrumb `Vorbemerkung`
- [ ] Preambuł L1 z listą lub zdaniem (bez ≥200 zn.) → osobny RU (ta sama reguła)
- [ ] Preambuł L1 bez treści merytorycznej → nie emitowany (brak False-positive)
- [ ] FreeText z markerem → pseudo-L1 `ANH-{slug}`, pod-`##` jako L2
- [ ] FreeText bez markera → L2 dziecko poprzedzającej L1
- [ ] `section_code` ANH-{slug} unikalny per dokument (collision-safe)
- [ ] `_validate` w `build_parquets.py` akceptuje zakres 350–420 RU
- [ ] Test regresyjny Q1: `section_code` zawierający „safe-drive" ma `is_retrieval_unit=True`
- [ ] Test regresyjny Q2: blok EV-Wechselprämie ma `parent_section_id` = sekcja §E
- [ ] Test regresyjny Q7: `section_code=1.0` ma `is_retrieval_unit=True` i „Wallboxen" w markdown
- [ ] `pytest tests/test_hierarchy_parser.py` — zielony (w tym 2 pre-existing failures bez zmian)

## Blokowane przez

Brak — można rozpocząć (Faza 2). Po implementacji wymagany pełny rebuild: `build_parquets → enrich_sections → build_embeddings`.
