# D1 — Exact-term force include

> PRD: `docs/PRD-retriever-repair-streams-ABCD.md` — Stream D (runtime)
> Typ: AFK
> Faza: 1 (runtime, zero rebuild)

## Co należy zbudować

Deterministyczna warstwa leksykalna: chunk zawierający termin domenowy z zapytania jest **wymuszony do pool** niezależnie od score dense. Zapewnia niezawodność tam gdzie dense score może być niewystarczający.

**Mechanika:**
- W `Retriever.__init__`: zbuduj odwrócony indeks `{normalized_term: [section_idx]}` z pola `topic_tags` wszystkich RU w indeksie.
- W `retrieve_multi` po DocFilter (kandydaci z pozycji `positions`): wyciągnij `domain_terms` i `topic_tags` z `query_obj`; znormalizuj (lowercase, strip); dla każdego termu spoza `GENERIC_BLOCKLIST` sprawdź substring match w `markdown` (znorm.) każdego kandydata → dodaj do `forced_set`.
- Merge: `final_pool = positions_set ∪ forced_set` (forced nie duplikują, nie przebijają DocFilter — forced nadal ze zbioru kandydatów po DocFilter, nie z całego indeksu).

**Uwaga dot. fuzji z DocFilter:** forced-include działa **wewnątrz** zbioru dopuszczonego przez DocFilter — nie omija filtra dokumentów. Chunk z wymaganym terminem ale spoza gate'a nie jest forcowany.

**GENERIC_BLOCKLIST** już istnieje w `src/doc_filter.py` — reużyj.

**Normalized substring match:** `term.lower() in chunk_markdown.lower()` — prosto i łapie kompozycje niemieckie (np. „Naturgefahren" w „Elementargefahren/Naturgefahren").

## Kryteria akceptacji

- [ ] Odwrócony indeks budowany w `__init__` z `topic_tags` RU
- [ ] `retrieve_multi`: forced_set = chunki z DocFilter-pool zawierające `domain_terms ∪ topic_tags` \ `GENERIC_BLOCKLIST`
- [ ] Forced-include nie omija DocFilter (operuje na `positions`, nie na całym `self._sections`)
- [ ] Znormalizowany substring match — kompozycje DE łapane
- [ ] Termy z `GENERIC_BLOCKLIST` nie wyzwalają force-include
- [ ] Test: query z `domain_term="Naturgefahren"` → chunk 292 w pool nawet gdy dense score niski
- [ ] Test: termin ogólny (np. „Versicherung") → nie wyzwala force-include (blocklist)
- [ ] `pytest tests/test_retriever.py` — zielony

## Blokowane przez

- A1 + A2 (ten sam `retrieve_multi`; kolejność operacji: boost → pool_k → force-include → dedup → rerank)
