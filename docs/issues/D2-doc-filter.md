# D2 — doc_filter: Protocol + adaptery + CompositeDocFilter

**Typ:** AFK  
**Blokowane przez:** D1  
**Dotyczy US:** 17–20  

---

## Co należy zbudować

Nowy moduł `src/doc_filter.py` portowany z wzorca DKV. Zastępuje inline filtr Sparte/Tarif w `src/retriever.py`.

**`DocFilter` Protocol** — interfejs adapterów:
```python
class DocFilter(Protocol):
    def filter(self, query: ExpandedQuery) -> frozenset[str]:  # frozenset of doc_id
        ...
```

**`ProductDetectorAdapter`** — tłumaczy `sparte` i `tarif` z `ExpandedQuery` na `frozenset[doc_id]` przez lookup w `documents.parquet`. Jeśli `tarif=None` → zwraca wszystkie doc_id danej Sparte. Nieznany tarif → pusta frozenset (nie rzuca wyjątku).

**`RareTagMatcherAdapter`** — mapuje `domain_terms` z `ExpandedQuery` na `frozenset[doc_id]`: dla każdego termu sprawdza które sekcje mają go w `topic_tags`, zbiera `doc_id` tych sekcji. Termy z blocklista generyków (lista z `plan.md §5`) → ignorowane. Pusta lista `domain_terms` → pusta frozenset.

**`CompositeDocFilter`** — przyjmuje listę adapterów; zwraca **unię** (nie iloczyn) ich frozenset. Pusta unia (wszystkie adaptery zwróciły `∅`) → brak filtra (retriever odpada do pełnego korpusu).

Blocklista generyków dla `RareTagMatcherAdapter`:
```python
GENERIC_BLOCKLIST = frozenset({
    "Versicherung", "Schaden", "Vertrag", "Versicherer",
    "Versicherungsnehmer", "Prämie", "Leistung",
})
```

---

## Kryteria akceptacji

- [ ] `DocFilter` Protocol zdefiniowany; `ProductDetectorAdapter`, `RareTagMatcherAdapter`, `CompositeDocFilter` go implementują
- [ ] `ProductDetectorAdapter(sparte="Hausrat", tarif="Smart")` → zwraca tylko `doc_id` Hausrat-Smart, nie Kfz ani inne Hausrat
- [ ] `ProductDetectorAdapter(sparte="Kfz", tarif=None)` → zwraca oba Kfz doc_id
- [ ] `ProductDetectorAdapter` z nieznanym tarif → pusta frozenset, brak wyjątku
- [ ] `RareTagMatcherAdapter(domain_terms=["Glasbruch"])` → doc_id sekcji z `topic_tags` zawierającym "Glasbruch"
- [ ] `RareTagMatcherAdapter(domain_terms=["Schaden"])` → pusta frozenset (generyk)
- [ ] `RareTagMatcherAdapter(domain_terms=[])` → pusta frozenset
- [ ] `CompositeDocFilter` → unia adapterów; jeśli obie frozenset puste → zwraca `frozenset()` (sygnał: brak filtra)
- [ ] Testy TDD w `tests/test_doc_filter.py`: wszystkie powyższe przypadki + test `CompositeDocFilter` z fixture `documents.parquet`
- [ ] `src/retriever.py` NIC nie zmienione w tym issue (refaktor retriever = D5)

## Blokowane przez

- D1 (`llm_providers` / `model_registry` — infrastruktura wspólna)
