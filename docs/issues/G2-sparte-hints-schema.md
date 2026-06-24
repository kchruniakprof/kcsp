# G2 — sparte_hints schema w ExpandedQuery

> PRD: `docs/PRD-product-scope-retrieval.md` — sekcja A+B.1, A+B.5
> Typ: AFK
> Zablokowane przez: brak — start natychmiast (równolegle z G1)

## Co należy zbudować

`ExpandedQuery` ma `sparte_hint: Optional[str]` — single value, nie może reprezentować comparison/cross-sparte queries. Zmienić na `sparte_hints: list[Literal["Kfz","Hausrat","Glas","Schmuck"]]`.

Dodać compat property `primary_sparte` (first element / None) żeby `RAGAssistant` i cross-sell logic nie wymagały refaktoru w tym issue.

Zaktualizować `_SYSTEM_PROMPT` i few-shot w `QueryExpansion`:
- Usunąć regułę `"null: cross-branch or unclear"` (instruowała LLM zwracać null dla comparison — błąd)
- Dodać: "return ALL detected Spartes; for comparison queries return ≥2 Spartes; OOS → empty list []"
- Dodać 2 few-shot przykłady multi-sparte w messages (user→assistant)

## Kryteria akceptacji

- [ ] `sparte_hints: list[Literal["Kfz","Hausrat","Glas","Schmuck"]]` w `ExpandedQuery` (zastępuje `sparte_hint`)
- [ ] Walidator: dedup z zachowaniem kolejności, cap ≤4, OOS wartości filtrowane → []
- [ ] `primary_sparte` property: `sparte_hints[0]` / `None` gdy pusta lista
- [ ] Stare `sparte_hint` usunięte z modelu (breaking change — G4 obsłuży RAGAssistant)
- [ ] Prompt: brak frazy "null: cross-branch or unclear"; obecna reguła multi-sparte + ≥2 dla comparison
- [ ] Few-shot: ≥2 przykłady multi-sparte w `_call_llm` messages list
- [ ] Testy w `test_query_expansion.py`: validator (dedup/cap/OOS), property (first/None), prompt nie zawiera zakazanej frazy
- [ ] `pytest tests/test_query_expansion.py` — zielony

## Blokowane przez

Brak — można rozpocząć natychmiast.
