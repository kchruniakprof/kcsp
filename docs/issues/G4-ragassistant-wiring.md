# G4 — RAGAssistant wiring (sparte_hints + tarif)

> PRD: `docs/PRD-product-scope-retrieval.md` — sekcja RAGAssistant
> Typ: AFK
> Zablokowane przez: G2 + G3

## Co należy zbudować

`RAGAssistant` musi przekazać nowe typy do pipeline po zmianach G2+G3:

- Wywołać `_detect_tarif(expanded.normalized_query, tarif_names)` po `query_expansion.expand()`
- Przekazać `sparte_hints` (lista) + wykryty `tarif` do `ProductDetectorAdapter` / `resolve_doc_set`
- Cross-sell logic: zmienić `expanded.sparte_hint` → `expanded.primary_sparte` (compat property z G2)
- `_CROSS_SELL_MAP` lookup przez `primary_sparte`
- Usunąć stare odwołania do `sparte_hint` (singular)

`tarif_names` = lista unikalnych wartości kolumny `tarif` z `documents_df` (bez None/NaN).

## Kryteria akceptacji

- [ ] `RAGAssistant.ask()` wywołuje tarif detection po expansion (używa `normalized_query`)
- [ ] `ProductDetectorAdapter` otrzymuje `sparte_hints: list` zamiast `sparte: Optional[str]`
- [ ] Cross-sell używa `primary_sparte` (nie `sparte_hint`)
- [ ] Brak odwołań do nieistniejącego `expanded.sparte_hint` w pliku
- [ ] Testy `test_ragassistant.py`: RAGAssistant z multi-sparte query → doc_filter obejmuje oba produkty; single-sparte z tarif → zawężone docs; cross-sell nadal działa przez primary_sparte
- [ ] `pytest tests/test_ragassistant.py` — zielony

## Blokowane przez

- G2 (sparte_hints schema)
- G3 (resolve_doc_set + gate composition)
