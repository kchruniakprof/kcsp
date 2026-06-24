# G1 — Index Section Filter (build_parquets.py)

> PRD: `docs/PRD-product-scope-retrieval.md` — sekcja D
> Typ: AFK
> Zablokowane przez: brak — start natychmiast

## Co należy zbudować

Sekcje alfabetycznego indeksu produktu (np. §A, §B, §C — strony ze strukturą "Fahrraddiebstahl (G.1) 14") muszą być wykluczone z puli retrieval. Są one oznaczone `is_retrieval_unit=True` przez obecną logikę L1/L2, ale nie zawierają treści merytorycznej — zaśmiecają wyniki.

Dodać deterministyczną regułę override w `build_parquets.py` jako dodatkowy krok po obliczeniu `is_retrieval_unit` z hierarchii. Wyeksponować regułę jako testowalną funkcję `is_index_section(heading, body) -> bool`.

Reguła: sekcja jest indeksem gdy **którekolwiek**:
- heading (po strippowaniu) = dokładnie jedna litera A–Z (`^[A-Z]$`)
- body zawiera ≥50% linii pasujących do page-ref: `\([A-Z]+\.?\d*\)\s*\d+`

## Kryteria akceptacji

- [ ] `is_index_section(heading, body)` funkcja publiczna w `build_parquets.py` (lub osobny moduł)
- [ ] `build()` aplikuje override: gdy `is_index_section` → `is_retrieval_unit=False`, niezależnie od logiki L1/L2
- [ ] Testy w `test_build_parquets.py`: True dla heading="A" + body indeksowe; False dla normalnej sekcji ubezpieczeniowej; True dla body ≥50% page-ref nawet gdy heading normalny
- [ ] Walidacja count w `_validate()` pozostaje poprawna (zakres 350–420 może wymagać korekty po wykluczeniu ~4 sekcji)
- [ ] `pytest tests/test_build_parquets.py` — zielony (poza pre-existing failures)

## Blokowane przez

Brak — można rozpocząć natychmiast.
