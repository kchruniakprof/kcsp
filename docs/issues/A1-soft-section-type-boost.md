# A1 — Soft section_type boost (zamiana hard-drop)

> PRD: `docs/PRD-retriever-repair-streams-ABCD.md` — Stream A (runtime)
> Typ: AFK
> Faza: 1 (runtime, zero rebuild)

## Co należy zbudować

Zmiana semantyki filtra `section_type` w `retrieve_multi` z hard-drop na addytywny boost scoringowy. Eliminuje przypadek Q4/Q6 Naturgefahren: sekcja 292 (rank #0 dense, score 0.56–0.59) jest wyrzucana przed scoringiem bo ma typ `SPECIAL_PROVISIONS` zamiast `WHAT_IS_INSURED`.

**Mechanika:**
- Zamiast wycinać kandydatów niepassujących do `section_types`, zachowaj wszystkich.
- Po obliczeniu score dense (dot-product): jeśli chunk zawiera co najmniej jeden z `section_types` → dodaj `+0.04` do score (raz, nie per-typ, nie stackuje).
- Sortuj po boosted_score; do reranker/top_k trafia lista posortowana ze scorami po boostcie.

Boost `+0.04` jest addytywny, nie multiplikatywny. Wartość calibrowana do typowego zakresu cosine (0.4–0.7) — daje wyraźny sygnał nie dominując.

**Testy regresyjne TDD:**
- Test czerwony przed fixem: sekcja 292 NIE jest w top-5 dla Q4/Q6 query przy starym hard-drop.
- Test zielony po fixie: sekcja 292 jest w top-5 przy soft-boost.

## Kryteria akceptacji

- [ ] `retrieve_multi`: brak hard-drop po `section_types` — wszystkie kandydaci z DocFilter wchodzą do scoringu
- [ ] Boost `+0.04` aplikowany dokładnie raz dla chunku matchującego ≥1 typ z `section_types`
- [ ] Chunk z wieloma pasującymi typami dostaje boost tylko raz (nie `+0.04 × n`)
- [ ] Chunk bez pasującego typu → score bez zmiany (nie kara)
- [ ] Test regresyjny Q4 (welche Gefahren Naturgefahren): section_id 292 w top-5
- [ ] Test regresyjny Q6 (ausgeschlossen Naturgefahren): section_id 292 w top-5
- [ ] Testy istniejące `test_retriever.py` — zielone (backward-compat)
- [ ] `pytest tests/test_retriever.py` — zielony

## Blokowane przez

Brak — można rozpocząć natychmiast. Równolegle z B1.
