# Handoff — naprawy DocFilter + Critic (po grillu eval)

> Data: 2026-06-20  
> Stan repo: branch `master`, ostatni commit `f28178f`  
> Poprzedni dokument problemów: `problemy.md` (ten plik = ustalenia z sesji grill, gotowe do implementacji)  
> **Status: NIC NIE ZAIMPLEMENTOWANE — same ustalenia projektowe.**

---

## Kontekst

Po E1 (`results_full_enriched.json`, 99 pytań, pass-rate 75.8%) przeprowadzono grill 24 failów. Rozbito na 3 grupy wg prawdziwej przyczyny w danych. Sesja = uzgodnienie planu napraw, bez kodu.

Pliki referencyjne (nie powielać tu treści):
- Wyniki eval: `results_full_enriched.json`, baseline `results_full.json`
- Wzorzec Critic DKV: `d:\_FUN\DKV_Belgium\calude\accuracy\src\critic.py` + `model_registry.py`
- Nasze pliki: `src/critic.py`, `src/doc_filter.py`, `src/retriever.py`, `src/ragassistant.py`, `src/generator.py`, `src/llm_providers.py`, `src/model_registry.py`
- E1 issue: `docs/issues/E1-eval-calibration.md`

---

## Diagnoza — 24 faile, 3 grupy

| Grupa | N | Przyczyna | Case # |
|-------|---|-----------|--------|
| 1. Empty retrieval (`src=[]`) | 6 | DocFilter pusty → retriever abstain | 1,2,16,18,19,23 |
| 2. Critic over-abstain (`src≠[]`) | 15 | sekcje OK, Critic mówi "nie wiem" | 4,5,7,8,9,10,11,12,13,14,15,17,20,21,22 |
| 3. Generator (`abstained=False`) | 3 | 2× artefakt assertu, 1× realny bug | 3,6,24 |

**Kluczowy insight:** wszystkie 24 failują na asercie `icontains-any` (keyword-match). 21/24 to abstain → stały komunikat → brak keyworda → fail trywialny. Metryka 75.8% **zaniża** prawdziwą jakość (niemiecka fleksja/synonimy łamią keyword-match).

---

## Problem 1 — DocFilter cross-branch (grupa 1: 6 case)

**Root cause (ustalony w danych):** wszystkie 6 mają `sparte_hint=None`.
- 1a (#1,#2): cross-branch COMPARISON "Glas vs Hausrat" — z natury dwie sparte.
- 1b (#16,18,19,23): pytania **generyczne** (grobe Fahrlässigkeit, Obliegenheiten, Ratenzahlung) — nie wymieniają produktu. `sparte=null` jest POPRAWNE; eval `expected_sparte` jest błędny.

Łańcuch awarii: `sparte=null` → `ProductDetectorAdapter`=∅ → **Bug B** (`RareTagMatcherAdapter` czyta `domain_terms` z fake query = `[]`) → composite=∅ → **Bug A** (retriever traktuje `frozenset()` jako "brak wyników" → `return []`).

### Decyzje
- **Opcja B (query flow)** — adaptery stateless, budowane raz; `domain_terms`/`sparte_hint` płyną przez `query_obj` do `filter()`. Fake query znika. (Uwaga: B nie jest "dokładniejszy" niż konstruktor-injection — wybrany za czystość: stateless, jedno źródło prawdy.)
- **Opcja X (sentinel)** — `None` = no-filter → wszystkie sekcje; `frozenset()` pusty = filtr aktywny bez trafień → `[]`. `CompositeDocFilter` zwraca `None` gdy union pusty.
- **Kaskada B+A** — `RareTagMatcher` próbuje zawęzić po `domain_terms` (precyzja, np. tag `grobe Fahrlässigkeit` istnieje w enrichmencie); gdy ∅ → no-filter fallback szuka wszędzie (recall). Dla pytań generycznych "szukaj wszędzie" jest pożądane.

### Use-case domenowy (potwierdzony)
"Glas vs Hausrat" = realny: agent pyta o granicę pokrycia/upsell (Glas to często Ergänzung do Hausrat — patrz `_CROSS_SELL_MAP`). Obsługa: **graceful** — retrieval musi dosięgnąć obu sparte, generator składa granicę. NIE pełny COMPARE dwóch taryf.

### Pliki: `src/doc_filter.py`, `src/retriever.py`, `src/ragassistant.py`

---

## Problem 2 — Critic over-abstain (grupa 2: 15 case)

Przepisać `src/critic.py` na konstrukcję DKV. **Zakres: Full minus network fallback (bez Cerebras).**

### Co przenieść z DKV (`d:\_FUN\DKV_Belgium\calude\accuracy\src\critic.py`)
1. **Structured output** — `instructor` + `response_model=CriticOutput` (zamiast surowy `json_object`+`json.loads`). Wymaga instructor-client (`groq_client()`), nie surowy `Groq()`.
2. **CoT** — `chain_of_thought: List[str]` (max 5), `reasoning: List[str]` (max 2), `confidence_score` (ge/le).
3. **`_coerce_str_list` validator** — flattenuje list-of-dict → list-of-str (anti-crash retry instructora).
4. **Prompt anti-over-abstain** (TO leczy 15 failów) — "DEFAULT TO PASS"; hedging ≠ halucynacja; niepełna odpowiedź → PASS; ABSTAIN tylko gdy wymyślone kwoty/daty/nazwane warunki nieobecne w kontekście. Treść z DKV, dostosowana do ERGO (zostaje niemiecki).
5. **REGEN loop** (`run_critic`) — REGEN → `generate_fn()` raz → recheck; `recheck==ABSTAIN` → ABSTAIN, inaczej PASS. Bez nieskończonej pętli.
6. **Ensemble** — PASS od primary → drugi model weryfikuje; `ensemble==ABSTAIN` → ABSTAIN.

### Decyzje (3)
1. **Ensemble = drugi Groq** — primary `qwen/qwen3-32b`, ensemble `llama-3.3-70b-versatile` (inna rodzina). Nowy klucz `REGISTRY["critic_ensemble"] = "llama-3.3-70b-versatile"`.
2. **Graceful PASS** — wyjątek primary LUB ensemble → log warning + PASS z oryginalną odpowiedzią (awaria bramki ≠ blokada odpowiedzi). **Zastępuje** network fallback DKV.
3. **Flaga w eval** — `enable_ensemble: bool`, domyślnie **OFF** (baseline tani: 1 call/query). Włączyć świadomie w osobnym run by zmierzyć Δ.

### Adaptacje vs DKV 1:1
- `model_registry` zostaje prosty `dict[str,str]` (NIE `ModelBinding`/`routing_kwargs`).
- `generate_fn` callback dostarcza `ragassistant`: `lambda: generator.generate(query, sections, mode)`.
- REGEN tylko dla VERBATIM (COMPARE i tak pomija critica — `ragassistant.py` ~linia 135).
- `CriticResult` musi oddać ewentualnie zregenerowaną odpowiedź do ragassistant.

### Pliki: `src/critic.py`, `src/model_registry.py`, `src/ragassistant.py`, `src/promptfoo_provider.py`

---

## Grupa 3 — generator (3 case)

| Case | Werdykt |
|------|---------|
| #3 Autotelefon/Navi | odpowiedź OK; miss przez synonim (`Navigationssystem` vs `Navigations-/Multifunktionsgerät`); drobny retrieval (Spezial vs Standard) |
| #6 Garaż 1200m | odpowiedź DOSKONAŁA; miss przez fleksję (`Garage` vs `Garagen`, `entfernt` vs `Entfernung`) — artefakt assertu |
| #24 wymiana pojazdu | **REALNY BUG** — retrieval trafia `Veräußerung` (sprzedaż) zamiast `Fahrzeugwechsel` (wymiana); `Mahnung` nietknięte |

### Decyzje
- **Assert → `llm-rubric` (promptfoo) dla in-scope** — ocena semantyczna, odporna na fleksję/synonim. OOS zostają na prostym asercie (abstain detection). Koszt: +1 LLM/pytanie, niedeterministyczne.
- **#24 = realny retrieval bug** — zachować do naprawy. Najpierw zbadać: czy sekcja o `Fahrzeugwechsel` w ogóle istnieje w parquet (`sections`/`subsections`).

---

## Drobiazgi domknięte (C/D/E/F)

- **C — eval ground-truth:** znika. Po `llm-rubric` `expected_sparte` staje się informacyjne; `eval_set.yaml` nieruszany.
- **D — `test_doc_filter_empty_set_returns_empty`:** rozbić na `test_none_means_no_filter` (→ wszystkie) + `test_empty_frozenset_returns_empty` (→ `[]`).
- **E — sygnatury:** `retrieve_multi` dostaje opcjonalny `query_obj` przekazywany do `doc_filter.filter()`; adaptery czytają z query; fake query znika. Caller: tylko `ragassistant`.
- **F — re-eval:** etapami. Subset 21 (6 empty + 15 critic) by tanio zmierzyć Δ → potem pełny 99 jako walidacja. Z `llm-rubric`.

---

## Kolejność wykonania (sugerowana)

1. **Problem 1** (deterministyczny, najłatwiej zweryfikować bez LLM):
   - doc_filter: sentinel `None`/`frozenset()`, query_obj flow, RareTagMatcher czyta domain_terms
   - retriever: `None`→wszystkie, `frozenset()`pusty→`[]`, dodać `query_obj`
   - ragassistant: zbudować adaptery raz, przekazać `query_obj`
   - testy D (rozbić), nowe testy cross-branch + generyczne (1b)
2. **Problem 2** (Critic Full −Cerebras) — TDD, instructor structured output, REGEN, ensemble-flag, graceful PASS
3. **Assert → llm-rubric** w `promptfooconfig.full.yaml` / `eval_set.yaml` (in-scope)
4. **#24** — zbadać dane, naprawić retrieval Fahrzeugwechsel
5. **Re-eval** subset 21 → pełny 99

---

## Lista realnych bugów (nie-artefaktów)

1. **DocFilter cross-branch + generyczne** (6 case) — Problem 1
2. **Critic over-abstain** (15 case) — Problem 2
3. **#24 retrieval Fahrzeugwechsel≠Veräußerung** (1 case)
4. *(carry-over z `problemy.md`)* Shadow ContextSelector O(n) w `promptfoo_provider.py` — przed produkcją
5. *(carry-over)* Windows Credential Manager blokuje git push; **PAT widoczny w historii czatu — zrotować**

---

## Sugerowane umiejętności

- `/tdd` — Problem 1 (doc_filter/retriever/ragassistant): RED→GREEN pionowymi wycinkami. Testy deterministyczne (mock embedder), bez kosztu LLM.
- `/tdd` — Problem 2 (Critic): structured output + REGEN + ensemble. Mock instructor-client w testach.
- `/grill-me` — jeśli pojawią się nowe decyzje przy implementacji llm-rubric (kształt rubryki, próg oceny).

---

## Stan E1 (kontekst, nie do zmiany)

Wszystkie kryteria akceptacji E1 spełnione (patrz `problemy.md` sekcja "Stan E1"). `EMBED_THRESHOLD = None` (dystrybucja bimodalna). Ten handoff to praca PO E1 — poprawa jakości ponad baseline.
