# H1 — SectionRecord TypedDict jako granica sekcji

> Typ: AFK
> Zablokowane przez: brak — start natychmiast

## Co należy zbudować

Wprowadza `SectionRecord(TypedDict)` jako wymuszony przez typy kontrakt między RAGAssistant a trzema modułami (Retriever, Generator, Critic). Obecnie sekcje przepływają jako `dict[str, Any]` — brak pola `markdown` powoduje cichą awarię (Critic ocenia pusty kontekst → ABSTAIN).

```python
# src/schemas.py  — prototyp z przeglądu architektury
from typing import TypedDict

class SectionRecord(TypedDict):
    section_id: int
    heading: str
    markdown: str
    breadcrumb: str
```

Sygnatury do aktualizacji:
- `Retriever._build_result()` → zwraca `SectionRecord`
- `Generator.generate(sections: list[SectionRecord])`
- `Critic.evaluate(sections: list[SectionRecord])`
- `run_critic(sections: list[SectionRecord])`
- `RAGAssistant.ask()` — buduje `list[SectionRecord]` z `RetrievalResult` i przekazuje do Generator + Critic

Konwencja nazewnictwa pochodzi z domeny projektu — "sekcja" jest pierwotną jednostką retrieval per ADR-005.

## Kryteria akceptacji

- [ ] `src/schemas.py` zawiera `SectionRecord(TypedDict)` z polami: `section_id: int`, `heading: str`, `markdown: str`, `breadcrumb: str`
- [ ] Retriever, Generator, Critic mają zaktualizowane sygnatury przyjmujące `list[SectionRecord]` zamiast `list[dict]`
- [ ] RAGAssistant buduje `list[SectionRecord]` z `r.markdown` (nie `r.pruned_markdown`) przed wywołaniem Generator i Critic
- [ ] mypy (lub pyright) nie zgłasza błędów typów na nowych sygnaturach
- [ ] Żaden istniejący test nie psuje się po zmianie
- [ ] Nowy test w `test_critic.py`: przekazanie `SectionRecord` bez pola `markdown` → `TypeError` (nie cicha awaria)
- [ ] Nowy test w `test_generator.py`: `SectionRecord` bez pola `heading` → `TypeError`
- [ ] `pytest --ignore=tests/test_hierarchy_parser.py -q` — zielony (247+ testów)

## Blokowane przez

Brak — można rozpocząć natychmiast.
