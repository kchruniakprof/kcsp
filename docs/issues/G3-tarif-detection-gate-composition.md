# G3 — Tarif detection + resolve_doc_set + gate composition

> PRD: `docs/PRD-product-scope-retrieval.md` — sekcja A+B.2, A+B.3, A+B.4
> Typ: AFK
> Zablokowane przez: G2 (wymaga `sparte_hints: list`)

## Co należy zbudować

Trzy powiązane zmiany w `doc_filter.py`:

**1. `_detect_tarif(normalized_query, tarif_names) -> Optional[str]`**
Deterministyczna funkcja word-boundary match (nie LLM). Longest-first. Rider aliasy z `split("+")` — `"Best+Fahrraddiebstahl"` → alias `Fahrraddiebstahl` mapuje do compound. Base token `"Best"` NIE jest samodzielnym aliasem (longest-match wins). Używa `normalized_query` (nie raw), więc działa dla pytań PL/EN.

```python
# z prototypu — koduje decyzję aliasowania
candidates = sorted(tarif_names, key=len, reverse=True)
for t in candidates:
    tokens = [t] + [part for part in t.split("+") if part != t.split("+")[0]]
    for tok in tokens:
        if re.search(rf"\b{re.escape(tok)}\b", normalized_query, re.IGNORECASE):
            return t
return None
```

**2. `resolve_doc_set(sparte_hints, tarif, documents_df) -> Optional[frozenset[str]]`**
Izolowalna funkcja z logiką 5-gałęziową:
- `sparte_hints=[]` → `None` (no filter)
- single sparte, tarif=None → wszystkie docs tej sparte
- single sparte + tarif → docs z `sparte AND tarif`
- multi sparte → union docs wszystkich sparte
- related_sparte safety-net: dodaje Hausrat docs gdy (Glas lub Schmuck ∈ sparte_hints) ∧ query zawiera `Hausrat|Haushalt|Wohnung` ∧ "Hausrat" ∉ sparte_hints

**3. Gate composition w `CompositeDocFilter`**
Zastąpić union-semantykę (current) gate-semantyką:
```
gate_result = resolve_doc_set(sparte_hints, tarif, documents_df)
rare_result = RareTagMatcherAdapter.filter(query)
if gate_result is None → None (no filter)
elif rare_result is None → gate_result
else: narrowed = gate ∩ rare; final = narrowed if narrowed else gate_result
```

## Kryteria akceptacji

- [ ] `_detect_tarif` publiczna (lub testowalna) funkcja: word-boundary, longest-first, rider alias, base NIE solo-alias
- [ ] `resolve_doc_set` publiczna funkcja: wszystkie 5 gałęzi logiki pokryte
- [ ] related_sparte safety-net: trigger wąski (wymaga hasła Hausrat w query)
- [ ] `CompositeDocFilter` używa gate-semantyki (intersection nie union)
- [ ] Testy `test_doc_filter.py`: każda gałąź `resolve_doc_set`; gate composition (rare cross-sparte → trzyma gate; rare ∩ gate → zawęża; gate=None → None); related_sparte trigger/brak-triggera
- [ ] `pytest tests/test_doc_filter.py` — zielony

## Blokowane przez

- G2 (sparte_hints: list — typ wymagany przez resolve_doc_set)
