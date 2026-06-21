# H2 — Centralizacja stałych domeny (SPARTES, SECTION_TYPES)

> Typ: AFK
> Zablokowane przez: brak — start natychmiast (niezależne od H1)

## Co należy zbudować

Przenosi stałe domeny — listę Sparten i listę typów sekcji — do jednego pliku `src/constants.py`. Obecnie:
- `{"Kfz", "Hausrat", "Glas", "Schmuck"}` zdefiniowane w 3 miejscach (`query_expansion.py`, `doc_filter.py`, `hierarchy_parser.py`)
- Typy sekcji istnieją w dwóch miejscach z rozbieżnością: 16 wartości w `hierarchy_parser.py` (ENUM_16) vs 12 w `_SECTION_TYPE Literal` w `query_expansion.py`

Test usunięcia: usuń definicję z `doc_filter.py` → błąd runtime + konieczność importu skądinąd. Złożoność pojawia się u innych wywołujących — to potwierdza że stała należy wyżej.

```python
# src/constants.py  — prototyp z przeglądu architektury
SPARTES: frozenset[str] = frozenset({"Kfz", "Hausrat", "Glas", "Schmuck"})

SECTION_TYPES: list[str] = [
    "WHAT_IS_INSURED", "EXCLUSIONS", "CLAIMS_SETTLEMENT",
    "LIMITS_COMPENSATION", "OBLIGATIONS", "PAYMENT",
    "PRICING_DISCOUNT", "TERM_CANCELLATION", "COMPLAINTS_LAW",
    "INSURED_PERSONS", "RISK_OBJECT", "WHERE_COVERED",
]
```

Rozbieżność 16 vs 12 wymaga decyzji: które 4 wartości z ENUM_16 nie mają odpowiednika w Literal? Jeśli aktywnie używane w enrichment → dodaj do `SECTION_TYPES`; jeśli tylko legacy → usuń z ENUM_16.

## Kryteria akceptacji

- [ ] `src/constants.py` istnieje z `SPARTES: frozenset[str]` i `SECTION_TYPES: list[str]`
- [ ] `query_expansion.py` importuje `SPARTES` z `constants` zamiast definiować lokalnie; `_SECTION_TYPE Literal` zastąpiony przez `Literal[tuple(SECTION_TYPES)]` lub importowany
- [ ] `doc_filter.py` importuje `SPARTES` z `constants` zamiast definiować lokalnie
- [ ] `hierarchy_parser.py` importuje `SECTION_TYPES` z `constants`; ENUM_16 usunięty lub zsynchronizowany
- [ ] Rozbieżność 12 vs 16 wyjaśniona i rozwiązana (dodanie brakujących wartości lub udokumentowanie że 4 wartości są legacy)
- [ ] `pytest --ignore=tests/test_hierarchy_parser.py -q` — zielony (247+ testów)
- [ ] Test w `test_constants.py` (nowy): `SPARTES` zawiera dokładnie `{"Kfz", "Hausrat", "Glas", "Schmuck"}`; `SECTION_TYPES` nie ma duplikatów; `len(SECTION_TYPES) >= 12`

## Blokowane przez

Brak — można rozpocząć natychmiast.
