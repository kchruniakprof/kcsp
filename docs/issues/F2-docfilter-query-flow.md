# F2 — DocFilter query flow: `ExpandedQuery` passthrough przez `retrieve_multi`

**Typ:** AFK  
**Blokowane przez:** F1  
**Dotyczy US:** 1–4, 7, 8  
**PRD:** `docs/PRD-docfilter-critic-overhaul.md`

---

## Co należy zbudować

Przepuścić `ExpandedQuery` przez cały pipeline retrieval aż do adapterów DocFilter, eliminując fake query i umożliwiając kaskadę B+A.

**Zmiana sygnatury `retrieve_multi`:** dodać opcjonalny parametr `query_obj` (typ `Any`, faktycznie `ExpandedQuery`). Gdy `doc_filter` jest aktywny, wywołać `doc_filter.filter(query_obj)` zamiast `doc_filter.filter(fake_query)`. Usunąć fake query `type("_Q", (), {...})()`.

**Zmiana `RareTagMatcherAdapter`:** adapter jest już stateless (nie zmienia konstruktora). Jednak musi czytać `domain_terms` z przekazanego `query_obj` przez `getattr(query_obj, "domain_terms", [])` — co już robi. Fix polega na tym, że teraz `query_obj` to realny `ExpandedQuery`, nie fake query z pustym `domain_terms`. Kaskada B+A działa automatycznie po F1 (brak trafień → `None` → fallback no-filter).

**Zmiana `RAGAssistant`:** przekazać `expanded` (wynik `QueryExpansion.expand()`) jako `query_obj` do `retrieve_multi`. Adaptery budowane raz per-query (stateless — bez zmiany).

**Kaskada B+A (weryfikacja end-to-end):** dla pytania generycznego (`sparte_hint=None`, `domain_terms=[]`):
1. `ProductDetectorAdapter` → `None` (brak sparte)
2. `RareTagMatcherAdapter` → `None` (puste domain_terms)
3. `CompositeDocFilter` → `None` (wszystkie None)
4. Retriever → wszystkie sekcje (no-filter)

Dla pytania z raretermem (`domain_terms=["grobe Fahrlässigkeit"]`):
1. `ProductDetectorAdapter` → `None` (brak sparte)
2. `RareTagMatcherAdapter` → `frozenset({doc_id...})` jeśli tagi pasują; `None` jeśli nie
3. `CompositeDocFilter` → union lub `None`

---

## Kryteria akceptacji

- [ ] `retrieve_multi` przyjmuje opcjonalny `query_obj` (backward-compatible — domyślnie `None`)
- [ ] Fake query `type("_Q", (), ...)()` usunięty z retriever.py
- [ ] `RAGAssistant` przekazuje `expanded` jako `query_obj` do `retrieve_multi`
- [ ] Query z `sparte_hint=None` i `domain_terms=[]` → retriever zwraca wyniki ze wszystkich sparte (no-filter path)
- [ ] Query z `sparte_hint=None` i `domain_terms=["grobe Fahrlässigkeit"]` → retriever zawęża do sekcji z tym tagiem (jeśli istnieje) LUB szuka wszędzie (no-filter fallback gdy brak trafień)
- [ ] Query z `sparte_hint="Kfz"` → retriever zwraca tylko sekcje Kfz (jak dotychczas)
- [ ] Istniejące testy retriever + doc_filter + ragassistant PASS
- [ ] Nowy test: cross-branch query (`sparte_hint=None`, `domain_terms=[]`) → wyniki z ≥2 sparte (in-memory fixture, bez LLM)
- [ ] Nowy test: generic query z raretermem → `RareTagMatcher` dostaje realny `domain_terms` (nie pusty)

## Blokowane przez

- F1 (sentinel `Optional[frozenset]` musi być zaimplementowany przed zmianą logiki composite)
