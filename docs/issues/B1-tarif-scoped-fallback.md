# B1 — Tarif scoped do sparte + fallback no-filter

> PRD: `docs/PRD-retriever-repair-streams-ABCD.md` — Stream B
> Typ: AFK
> Faza: 1 (runtime, zero rebuild)

## Co należy zbudować

Dwie powiązane zmiany w `doc_filter.py` eliminujące kolizję tarif cross-sparte i abstain spowodowany pustym gate:

**1. `_detect_tarif` scoped do sparte_hints**
Funkcja otrzymuje listę taryf już zawężoną do dokumentów z aktywnych `sparte_hints`. Eliminuje przypadek, gdzie „Best" (taryfa Hausrat) jest wykrywana w zapytaniu Kfz, powodując `resolve_doc_set(["Kfz"], "Best") = ∅`.

Wywołujący (`resolve_doc_set`) buduje `tarif_names_for_sparte` z `documents_df`:
```python
# z prototypu — koduje decyzję scopingu
tarif_names = documents_df[documents_df["sparte"].isin(hints)]["tarif"].dropna().unique().tolist()
tarif = _detect_tarif(normalized_query, tarif_names)
```

**2. `resolve_doc_set` fallback no-filter**
Gdy wynik = `∅` (pusty frozenset — aktywny gate, zero matchów) → zamiast zwracać `frozenset()` (które retriever interpretuje jako hard-empty), zwraca `None` (no-filter semantics) i sygnalizuje `filter_fallback=True`.

`filter_fallback` przekazywany jako pole w zwracanej krotce lub przez mutowalny obiekt kontekstu — decyzja implementacyjna, byle downstream (`retriever.py`) miał dostęp do flagi.

Kontrakt: `frozenset()` **nigdy** nie wychodzi z `resolve_doc_set`. Puste = szukaj wszędzie.

## Kryteria akceptacji

- [ ] `_detect_tarif` przyjmuje `tarif_names` zawężone do sparte_hints — „Best" w query Kfz nie matchuje gdy Hausrat wyfiltrowany
- [ ] `resolve_doc_set` gdy wynik ∅ → zwraca `None` + flaga `filter_fallback=True`
- [ ] `frozenset()` (pusty aktywny gate) nie pojawia się jako return value `resolve_doc_set`
- [ ] Test: query „Smart→Best Wechsel" z `sparte_hints=["Kfz"]` → tarif=None (nie „Best")
- [ ] Test: `resolve_doc_set` z tarif+sparte bez matchów → `(None, filter_fallback=True)`, nie `frozenset()`
- [ ] Test: wszystkie oryginalne branche z G3 nadal zielone (regression)
- [ ] `pytest tests/test_doc_filter.py` — zielony

## Blokowane przez

Brak — można rozpocząć natychmiast.
