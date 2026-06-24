# PRD — Naprawa retrievera: Streamy A+B+C+D

> Wersja: 1.0 | Data: 2026-06-21
> Poprzedni PRD: `docs/PRD-product-scope-retrieval.md` (G1–G5)
> Cel: 50/50 na broker50 bez regresji na eval 99 (baseline: 43/50)
> Plan fazowany: Faza 1 (runtime-only, 0 rebuildu) → Faza 2 (rebuild) → Faza 3 (strojenie)

---

## Opis problemu

System RAG ERGO P&C (Kfz/Hausrat/Glas/Schmuck) nie udziela odpowiedzi na 7 z 50 pytań
zestawu broker50. Diagnoza grounded (offline probe BGE-M3 + inspekcja parquet) wykazała, że
bi-encoder rankuje właściwe chunki wysoko — głównym wrogiem jest pipeline downstream:

1. **Twardy filtr `section_type` wyrzuca #0 przed scoringiem** (Q4, Q6 Naturgefahren):
   sekcja 292 (§4.9 „was versichern wir damit?") dostaje typ `SPECIAL_PROVISIONS` zamiast
   `WHAT_IS_INSURED` bo keyword `"versichert"` nie matchuje nagłówka z `"versichern"`.
   Filtr w `retriever.py:178–186` to hard-drop, nie boost — chunk odpada mimo dense rank #0.

2. **Kolizja tarif cross-sparte** (Q3 Smart→Best Wechsel, Q5):
   `_detect_tarif` łapie słowo „Best" z Hausrat w zapytaniu Kfz → `resolve_doc_set(["Kfz"],
   "Best")` = ∅ → `retriever` zwraca `[]` (abstain zamiast fallback). Dodatkowo brak
   zawężenia taryfy do sparte powoduje zalew sekcji §25 „Überversichert" z 4 prawie-duplikatów.

3. **Treść poza schematem numeracji — nieindeksowalna lub pogrzebana** (Q1, Q2, Q7):
   - Preambuł L1 z listą elementów (§1 Smart Wallbox) jest osieroconym rodzicem —
     nie-RU, body grzęźnie w ogonek sąsiedniego chunku lub jest pomijane.
   - Bloki `## FreeText` (Anhang „Sonderbedingungen Safe Drive", „EV-Wechselprämie")
     są wessane w ogon chunku o innym kodzie sekcji → dense nie rankuje ich po słowach kluczowych.

4. **Brak warstwy leksykalnej i dedup** (Q5 zalew, defense-in-depth):
   Dokładne terminy domenowe (np. „Naturgefahren", „Wallboxen") nie są gwarantowane w pool
   nawet gdy dense score jest wysoki. 4 prawie-identyczne dokumenty Hausrat zalewają top-5.

---

## Rozwiązanie

### Stream A — Miękki filtr section_type + reranker zawsze-on + LLM retyping

Filtr `section_type` zmieniony z hard-drop na addytywny boost `+0.04` (raz, bez stackowania).
`CrossEncoderReranker` (komponent istnieje, `src/retriever.py`) wpinany zawsze (`reranker=None`
w `promptfoo_provider.py` → inject). `pool_k`: cały przefiltrowany zbiór gdy ≤50, inaczej 30.

Przy enrichmencie: `section_types` = union(reguła LLM multi-label, reguły keyword) — over-labeling
nieszkodliwy przy miękkim filtrze, under-labeling kosztowny.

### Stream B — Tarif scoped do sparte + fallback no-filter

`_detect_tarif` dostaje listę taryf już zawężoną do `sparte_hints` → „Best" z Hausrat nie
matchuje w zapytaniu Kfz. `resolve_doc_set` gdy wynik = ∅ → **fallback no-filter** +
flaga `filter_fallback=True` zamiast pustego returna (nigdy abstain z powodu pustego gate).

### Stream C — Chirurgiczna łatka parsera sub-dokumentów

Dwie nowe reguły w `hierarchy_parser.py`:
- **Preambuł L1**: gdy L1 ma dzieci I body po `strip_noise` ≥ ~200 zn. lub zawiera listę/zdanie
  → emituj jako osobny RU (kod `N.0`, breadcrumb `…> Vorbemerkung`).
- **Bloki `## FreeText`**: marker załącznika (`Anhang|Sonderbedingungen|Besondere Bedingungen`)
  → nowa pseudo-L1 (np. `ANH-SafeDrive`), pod-`##` jako jej L2. Brak markera → L2 dziecko
  najbliższej poprzedzającej L1.

### Stream D — BGE-M3 sparse+dense (RRF) + exact-term include + dedup

Swap `sentence-transformers` → `FlagEmbedding` (`BGEM3FlagModel`, `return_dense+return_sparse`
w 1 przebiegu), fuzja RRF (nie BM25 — kruchy na kompozycje). Deterministyczny exact-term
include: chunk wymuszony do pool gdy zawiera `domain_term` jako znorm. substring (łapie kompozycje
DE). Dedup near-duplicate przed rerankingiem (emb-cosine >0.98 lub MinHash), reprezentant =
najwyższy score + pole `shared_tarifs`.

---

## Historie użytkownika

### Stream A — Miękki filtr + reranker + retyping

1. Jako agent ERGO pytający o Naturgefahren chcę mieć pewność, że sekcja §4.9 trafia do
   top-5, nawet gdy jej typ jest omyłkowo sklasyfikowany jako `SPECIAL_PROVISIONS`, aby
   odpowiedź nie była pusta (abstain) dla pytań pokrytych w dokumencie.

2. Jako deweloper chcę, żeby filtr `section_type` był addytywnym boostem (`+0.04`), a nie
   hard-dropem, aby wysoko zrankowany przez bi-encoder chunk nigdy nie był wyrzucany przed
   scoringiem.

3. Jako deweloper chcę mieć pewność, że boost nie stackuje się per-typ (cap raz), aby
   chunk z wieloma typami nie uzyskał nieproporcjonalnie wysokiego score.

4. Jako deweloper chcę `CrossEncoderReranker` wstrzykiwany zawsze (nie tylko gdy jest podany
   explicite), aby reranking był aktywny w każdym wywołaniu produkcyjnym.

5. Jako deweloper chcę `pool_k` = min(len(filtered_pool), 30) (lub cały pool gdy ≤50), aby
   reranker oceniał sensowną liczbę kandydatów bez kosztownego scorowania całego indeksu.

6. Jako deweloper chcę, żeby `section_types` w enrichmencie był union(reguły keyword, LLM
   multi-label), aby over-labeling był bezpieczny (miękki filtr), a under-labeling niemożliwy
   gdy LLM poprawnie klasyfikuje.

7. Jako deweloper chcę testy regresyjne asertujące, że sekcja 292 (§4.9 Naturgefahren)
   trafia do top-5 dla zapytań Q4 i Q6 broker50, aby przyszłe zmiany filtra nie łamały tych przypadków.

### Stream B — Tarif scoped do sparte + fallback

8. Jako agent ERGO pytający o „Smart vs Best Wechsel" w Kfz chcę, żeby detekcja taryfy
   nie mylnie łapała słowa taryf Hausrat obecnych w treści zapytania, aby retrieval zwracał
   wyniki z właściwej sparte.

9. Jako deweloper chcę, żeby `_detect_tarif` otrzymywał tylko taryfy z katalogu przefiltrowanego
   do aktywnych `sparte_hints`, aby kolizja cross-sparte była niemożliwa.

10. Jako deweloper chcę, żeby pusty wynik `resolve_doc_set` (∅) powodował fallback do
    no-filter z flagą `filter_fallback=True`, a nie abstain, aby system zawsze próbował
    zwrócić jakiś wynik.

11. Jako deweloper chcę flagę `filter_fallback=True` dostępną dla downstream warstw
    (adaptacyjny floor-abstain w Fazie 3), aby niezawodność filtra była mierzalna.

12. Jako deweloper chcę testy jednostkowe dla nowego branch `_detect_tarif(scoped)` i
    `resolve_doc_set(fallback)`, aby każda gałąź logiki była pokryta.

### Stream C — Parser sub-dokumentów

13. Jako agent ERGO pytający o „Safe Drive bonus 10%/30%/5000km" chcę, żeby treść
    „Sonderbedingungen Safe Drive" z Anhang była osobnym retrievalowalnym chunkiem, a nie
    ukryta w ogonek sekcji N.3, aby odpowiedź była precyzyjna.

14. Jako agent ERGO pytający o „EV-Wechselprämie 2500€" chcę, żeby blok `## FreeText` w §E
    był indeksowany jako L2 dziecko §E, a nie pominięty, aby retrieval mógł go zwrócić.

15. Jako agent ERGO pytający o „Wallbox 3000€-Begrenzung" chcę, żeby preambuł §1 Smart
    z listą „Wallboxen, Ladesäulen" był osobnym RU (L1.0 Vorbemerkung), a nie osieroconym
    nie-RU, aby treść była dostępna w retrieval.

16. Jako deweloper chcę regułę preambuł L1 (body ≥200 zn. lub zawiera listę → osobny RU,
    kod `N.0`) w `hierarchy_parser.py`, aby preambuły z treścią merytoryczną nie były tracone.

17. Jako deweloper chcę regułę `## FreeText` z detekcją markera załącznika (regex
    `Anhang|Sonderbedingungen|Besondere Bedingungen`) → pseudo-L1, brak markera → L2 §poprzedniej,
    aby bloki FreeText były indeksowane hierarchicznie.

18. Jako deweloper chcę, żeby `_validate` w `build_parquets.py` akceptował zakres 350–420
    RU po rebuild (wzrósł z ~280), aby walidacja nie failowała po zwiększeniu liczby chunków.

19. Jako deweloper chcę testy regresyjne parsera asertujące, że sekcje `ANH-SafeDrive`
    (Q1), blok `EV-Wechselprämie` w §E (Q2) i preambuł §1 Smart (Q7) mają
    `is_retrieval_unit=True` i poprawne breadcrumby po rebuild.

### Stream D — Hybrid + exact-term + dedup

20. Jako deweloper chcę BGE-M3 sparse+dense w jednym przebiegu (`BGEM3FlagModel`,
    `return_dense=True, return_sparse=True`), aby retrieval łączył sygnał semantyczny z
    leksykalnym bez osobnego BM25.

21. Jako deweloper chcę fuzję RRF (Reciprocal Rank Fusion) wyników dense i sparse, a nie
    prostego sumowania, aby model nie wymagał kalibracji wag per-korpus.

22. Jako agent ERGO chcę, żeby chunk zawierający dokładny termin domenowy (`domain_term`
    lub `topic_tag` ze słownika enrichmentu) był zawsze wymuszony do puli kandydatów, aby
    exact-match był niezawodny niezależnie od score dense/sparse.

23. Jako deweloper chcę znormalizowane substring-match (lowercase, bez znaków szczególnych)
    dla exact-term include, aby „Naturgefahren" łapało kompozycje DE i warianty gramatyczne.

24. Jako deweloper chcę `GENERIC_BLOCKLIST` (`src/doc_filter.py`) odfiltrowującą ogólne
    terminy (Versicherung, Schaden, …) z exact-term include, aby precyzja forced-pool była
    zachowana.

25. Jako deweloper chcę dedup near-duplicate przed rerankingiem (emb-cosine > 0.98 lub
    MinHash znorm. tekstu), aby 4 prawie-identyczne dokumenty Hausrat nie zalewały top-5.

26. Jako deweloper chcę pole `shared_tarifs` w repr. po dedup, aby COMPARE w LLM wiedział
    że chunk pochodzi z wielu dokumentów taryf.

27. Jako deweloper chcę testy jednostkowe dla `RRF`, `exact_term_force_pool` i `dedup_near_dup`,
    każdą funkcję testowalną w izolacji.

---

## Decyzje dotyczące wdrożenia

### Stream A

- **Moduł `src/retriever.py`**: zmiana `retrieve_multi` — filtr `section_type` zamieniony
  na boost addytywny `+0.04` po scorowaniu; `pool_k = min(len(filtered), 30)` (cały gdy ≤50).
- **`reranker` zawsze-on**: `promptfoo_provider.py` (i wszystkie instantiacje `Retriever` w prod)
  inject `CrossEncoderReranker()` zamiast `None`.
- **`src/enrich_sections.py`**: typ enrichmentu = union(wynik LLM multi-label, `_assign_types`
  keyword). Asymetria: miękki filtr faworyzuje over-labeling.
- **Faza 3**: adaptacyjny floor-abstain (`margin_top1-top2`, `filter_fallback`, `query_conf`)
  — poza zakresem Fazy 1 i 2.

### Stream B

- **`src/doc_filter.py`**: `_detect_tarif(normalized_query, tarif_names_for_sparte)` —
  `tarif_names_for_sparte` = `documents_df[documents_df['sparte'].isin(sparte_hints)]['tarif'].unique()`.
- **`src/doc_filter.py`**: `resolve_doc_set` — gdy wynik = ∅ → return `None` (no-filter)
  + ustawiaj flagę `filter_fallback=True` w obiekcie ExpandedQuery lub jako atrybut gate.
- **Kontrakt gate**: `frozenset()` (pusty, aktywny gate) → nigdy nie wychodzi z `resolve_doc_set`;
  to wbudowany błąd bezpieczeństwa. Fallback None = „szukaj wszędzie".

### Stream C

- **`src/hierarchy_parser.py`**: nowe funkcje `_emit_preamble_ru` i `_emit_freetext_block`;
  wywoływane w `parse_document` w odpowiednich miejscach iteracji po nagłówkach.
- **Kod preambuł**: `{parent_code}.0` (np. `1.0`, `A.0`); breadcrumb `…> §{code} Vorbemerkung`.
- **Kod załącznika**: `ANH-{slug}` (slug = pierwsze słowa markera po normalizacji), breadcrumb
  `{doc_id} > Anhang > {slug}`.
- **`src/build_parquets.py`**: zaktualizować stałe `_MIN_RU` / `_MAX_RU` (zakres 350–420).
- **Rebuild po Stream C**: `build_parquets → enrich_sections → build_embeddings` (ADR-007).

### Stream D

- **`src/build_embeddings.py`**: swap na `BGEM3FlagModel` z `FlagEmbedding`; przechowywać
  osobno dense i sparse (sparse jako dict `{token_id: weight}`); fuzja RRF w retrieverze.
- **`src/retriever.py`**: `retrieve_multi` — po filtrze doc/type: (1) dense score, (2) sparse
  score → RRF rank → forced-pool exact-term → dedup → reranker → top_k.
- **Exact-term include**: indeks odwrócony `{normalized_term: [section_idx]}` budowany w
  `__init__`; przy każdym query union wyników indeksu dla `domain_terms ∪ topic_tags` \ blocklist.
- **Dedup**: po przebudowaniu pool (przed rerankerem); próg emb-cosine tunable w `REGISTRY`.
- **`BGEM3FlagModel`** ładowany lazy (jak `CrossEncoderReranker`); install `FlagEmbedding`.

---

## Decyzje dotyczące testowania

**Dobry test**: weryfikuje zewnętrzne zachowanie (wynik `retrieve_multi` dla danego query),
nie bada prywatnych zmiennych ani kolejności wewnętrznych kroków. Używa małych in-memory
DataFrame (jak w `tests/test_ragassistant.py`) zamiast prawdziwego parquet.

**Testy regresyjne per fail** (TDD: czerwony → fix):
- 7 testów asertuje `section_id` w top-5 lub oczekiwany termin w odpowiedzi dla Q1–Q7 broker50.
- Baseline: test czerwony przed fixem, zielony po.

**Moduły testowane:**

| Moduł | Plik testowy | Co testować |
|---|---|---|
| `src/doc_filter.py` | `tests/test_doc_filter.py` | `_detect_tarif` scoped; `resolve_doc_set` fallback; gate composition nowe branche |
| `src/retriever.py` | `tests/test_retriever.py` | boost soft-type vs. hard-drop; pool_k policy; reranker always-on; exact-term force; dedup; RRF |
| `src/hierarchy_parser.py` | `tests/test_hierarchy_parser.py` | preambuł L1 → RU; FreeText marker → pseudo-L1; FreeText brak markera → L2 |
| `src/build_parquets.py` | `tests/test_build_parquets.py` | zakres RU po rebuild (350–420) |

**Wzorzec testów istniejących** do naśladowania: `tests/test_doc_filter.py`,
`tests/test_retriever.py` — oba używają `pd.DataFrame` w pamięci, mockują embedder.

---

## Poza zakresem

- **Adaptacyjny floor-abstain** (Faza 3) — poza tym PRD; wymaga zmierzonego baseline po Fazie 1+2.
- **Wallbox assert leksykalny** — treść ma l.mn. „Wallboxen, Ladesäulen"; fix = parser (Stream C),
  nie zmiana asercji w eval.
- **Zamiana modelu BGE-M3 dense** — tylko rozszerzenie o sparse (`return_sparse`); model dense
  pozostaje `BAAI/bge-m3`.
- **BM25** — explicite odrzucony (wymaga dekompozytora kompozycji DE, kruchy na novel compoundy).
- **Triple hybrid** (dense+sparse+BM25) — odrzucony: dodatkowa złożoność bez przewagi.
- **Flat-split parsera** — odrzucony jako zbyt szeroki blast-radius; Stream C = chirurgiczna łatka.
- **Zmiany w Critic** — nie dotykamy `src/critic.py` w tym PRD.
- **Zmiany w `query_expansion.py`** — G2b już zaaplikowany; poza zakresem.
- **Issues EV (eval_set.yaml bugfix)** — osobne zadanie, niezależne od retrivalu.

---

## Dodatkowe uwagi

**Kolejność wdrożenia (wg zależności):**

```
Faza 1 (runtime, 0 rebuild):
  Stream B całość → Stream A runtime (boost, reranker, pool_k) → Stream D runtime (exact-term, dedup)
  Mierzyć po Fazie 1: broker50 + eval99. Cel: Q3,Q4,Q5,Q6 naprawione.

Faza 2 (jeden rebuild):
  Stream C (parser) → rebuild parquets → enrich (retyping LLM) → build_embeddings (BGEM3 sparse+dense)
  Mierzyć po Fazie 2: broker50 + eval99. Cel: Q1,Q2,Q7 naprawione → 50/50.

Faza 3 (strojenie, poza tym PRD):
  Adaptacyjny floor-abstain na zmierzonym baseline.
```

**Dowód poprawności diagnozy (walidacja przed startem):**
- `subsections.parquet` section 292 → `section_types=['SPECIAL_PROVISIONS']` (potwierdzony).
- Offline probe BGE-M3: 292 rank #0 dla Q4/Q6 (score 0.56–0.59) — bi-encoder nie jest winny.
- Winny = filtr w `retriever.py:178–186`.

**Update istniejących issues:**
- `G3` (`doc_filter.py`) → dopisać scoped `_detect_tarif` + `filter_fallback`.
- `G5` (`retriever.py`) → zmienić pool_k=20 na policy ≤50/30; reranker zawsze-on (inject w provider).
- Nowe issues: `A2-section-type-soft-boost.md`, `A3-retyping-union.md`, `C1-parser-preamble-freetext.md`,
  `D1-bgem3-sparse-rrf.md`, `D2-exact-term-include.md`, `D3-near-dedup.md`.

**Priorytety kryterium akceptacji:**
Kryterium nadrzędne: **dokładność + niezawodność > latencja/koszt**. Chirurgiczność > szeroki
refactor tam, gdzie blast-radius wysoki (Stream C).
