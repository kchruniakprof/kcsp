# F1 — DocFilter sentinel: `None` = no-filter, `frozenset()` = empty

**Typ:** AFK  
**Blokowane przez:** Brak — można rozpocząć natychmiast  
**Dotyczy US:** 5, 6  
**PRD:** `docs/PRD-docfilter-critic-overhaul.md`

---

## Co należy zbudować

Zmienić kontrakt `CompositeDocFilter.filter()` z `frozenset[str]` na `Optional[frozenset[str]]`, wprowadzając dwupoziomowy sentinel:

- **`None`** = żaden adapter nie zidentyfikował produktu → no-filter → retriever przeszukuje wszystkie sekcje
- **`frozenset()` (non-None, pusty)** = filtr aktywny, brak trafień → retriever zwraca `[]`

**Zmiany semantyki adapterów:**

`ProductDetectorAdapter` bez `sparte` (ani z konstruktora, ani z `query.sparte_hint`) → zwraca `None` zamiast `frozenset()`.

`RareTagMatcherAdapter` gdy brak trafień (puste `domain_terms` lub żaden tag nie pasuje) → zwraca `None` zamiast `frozenset()`.

**Logika `CompositeDocFilter`:** iteruje przez adaptery zbierając non-None wyniki. Jeśli chociaż jeden adapter zwrócił non-None → unia tych wyników. Jeśli WSZYSTKIE zwróciły `None` → zwraca `None` (no-filter).

**Retriever (Bug A fix):** obsługuje `Optional[frozenset]` z `doc_filter.filter()`:
- `None` → `positions = list(range(len(self._sections)))` (wszystkie)
- `frozenset()` (pusty, non-None) → `return []`
- non-empty frozenset → filtruj jak dotychczas

---

## Kryteria akceptacji

- [ ] `ProductDetectorAdapter` bez `sparte` (init i query) → zwraca `None`
- [ ] `RareTagMatcherAdapter` z pustymi `domain_terms` → zwraca `None`
- [ ] `RareTagMatcherAdapter` z terminami z blocklista → zwraca `None`
- [ ] `RareTagMatcherAdapter` z raretermem bez trafień w tagach → zwraca `None`
- [ ] `CompositeDocFilter` gdy wszystkie adaptery `None` → zwraca `None`
- [ ] `CompositeDocFilter` gdy jeden adapter `frozenset({doc1})` + drugi `None` → zwraca `frozenset({doc1})`
- [ ] `CompositeDocFilter` gdy jeden adapter `frozenset()` (pusty, aktywny) + drugi `None` → zwraca `frozenset()` (pusty)
- [ ] Retriever z `doc_filter=None` → wszystkie sekcje (bez zmian)
- [ ] Retriever z `doc_filter.filter()` = `None` → wszystkie sekcje (no-filter path)
- [ ] Retriever z `doc_filter.filter()` = `frozenset()` (pusty) → `[]`
- [ ] Istniejący test `test_doc_filter_empty_set_returns_empty` zastąpiony dwoma: `test_none_means_no_filter` + `test_empty_frozenset_returns_empty`
- [ ] Wszystkie istniejące testy nadal PASS

## Blokowane przez

Brak — można rozpocząć natychmiast.
