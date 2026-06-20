# B1 — enrich_sections: batch Core-4 z checkpoint/resume

**Typ:** AFK  
**Blokowane przez:** A1  
**Dotyczy US:** 6–12  

---

## Co należy zbudować

Nowy skrypt `src/enrich_sections.py`: iteruje po wierszach parquet z `is_retrieval_unit=True`, wywołuje `enrich_section()` z `src/enrichment.py`, zapisuje Core-4 (`title`, `description`, `questions`, `topic_tags`) z powrotem do parquet, obsługuje checkpoint/resume i cost-gate.

**Ważne:** ten skrypt uruchamia się TYLKO po explicit go od użytkownika (realny koszt LLM ~370 wywołań). Cost-gate musi być w skrypcie.

Checkpoint: plik `parquet/enrichment_checkpoint.json` — klucz `section_id` → `true`. Po każdym sukcesie dopisuje wpis natychmiast (nie batch na końcu). Przy re-starcie: load checkpoint → skip już wzbogaconych → kontynuuj od pierwszej niewzbogaconej.

Pola Core-4 zapisywane z powrotem do `sections.parquet` i `subsections.parquet` (update wierszy `is_retrieval_unit=True`; L1-rodzice pomijane, ich pola Core-4 pozostają null).

---

## Kryteria akceptacji

- [ ] Skrypt iteruje wyłącznie po `is_retrieval_unit=True` (nie płaci za L1-rodziców)
- [ ] Przed startem: wypisuje szacowany koszt (liczba sekcji × est. tokenów) i pyta `Proceed? [y/N]`; flaga `--yes` pomija pytanie
- [ ] Checkpoint zapisywany po każdym sukcesie (nie na końcu); plik `parquet/enrichment_checkpoint.json`
- [ ] Skip-done: sekcje już w checkpoint są pomijane bez nowego wywołania API
- [ ] Po przerwaniu i restarcie: nie duplikuje wywołań; zaczyna od pierwszej nieskończonej
- [ ] Pola `title`, `description`, `questions`, `topic_tags` zapisane w parquet dla wszystkich `is_retrieval_unit=True`
- [ ] Brak null w Core-4 dla `is_retrieval_unit=True` po kompletnym batchie (walidacja na końcu skryptu)
- [ ] Pydantic validation przez instructor: malformed output → auto-retry (max_retries=3); po wyczerpaniu retries → log error + skip z wpisem do osobnego `enrichment_errors.json`
- [ ] Prompt po angielsku; wszystkie wyjścia w niemiecku (sprawdzić: `title` nie zawiera angielskich słów poza terminami prawnymi)
- [ ] `questions` per sekcja: 5–10 elementów (instructor waliduje zakres)
- [ ] Testy: deterministyczny mock `enrich_section()` zwracający fixture `SectionDetails`; test checkpoint skip-done; test cost-gate prompt

## Blokowane przez

- A1 (parquet musi zawierać `is_retrieval_unit`)
