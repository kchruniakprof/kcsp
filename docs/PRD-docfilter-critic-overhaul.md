# PRD — DocFilter Cross-Branch Fix + Critic Overhaul

> Źródło: sesja grill po E1 (`results_full_enriched.json`, 99 pytań, pass-rate 75.8%)  
> Dokument decyzji: `out_problemy.md`  
> Status: ready-for-agent

---

## Opis problemu

Agent ubezpieczeniowy ERGO abstainuje w 21/99 przypadkach, gdzie odpowiedź jest możliwa do udzielenia.

**Problem 1:** Pytania generyczne (bez wymienionego produktu) i cross-branch COMPARISON (np. "Glas vs Hausrat") trafiają w łańcuch błędów DocFilter → pusty retrieval → abstain. Użytkownik dostaje "Ich kann keine Antwort geben" mimo że sekcje w bazie istnieją.

**Problem 2:** Nawet gdy retrieval dostarcza trafne sekcje, Critic zbyt agresywnie werdykuje ABSTAIN dla pytań o wykluczenia i procedury szkodowe. System blokuje poprawne odpowiedzi.

Oba problemy powodują, że metryka pass-rate **zaniża** rzeczywistą jakość systemu: 21/24 failów to artefakt abstain → stały komunikat → brak słowa kluczowego w assert.

---

## Rozwiązanie

**Problem 1 — DocFilter sentinel + query flow:**  
Zastąpić aktualny błędny sentinel (pusty `frozenset()` = "brak wyników") logiką dwupoziomową: `None` = brak filtru → szukaj wszędzie; `frozenset()` = filtr aktywny bez trafień → pusta lista. `RareTagMatcherAdapter` ma otrzymywać `domain_terms` z `ExpandedQuery` (nie fake query), co umożliwia fallback na "szukaj wszędzie" gdy enriched tags pasują do genericquery.

**Problem 2 — Critic Full DKV −Cerebras:**  
Przepisać Critica na strukturę z projektu DKV (instructor + Pydantic, CoT, REGEN loop, ensemble) z promptem anti-over-abstain ("DEFAULT TO PASS"). Graceful PASS zastępuje network fallback.

---

## Historie użytkownika

### Problem 1 — DocFilter

1. Jako agent RAG, chcę retrievować sekcje ze wszystkich sparte gdy pytanie nie wymienia konkretnego produktu, aby nie abstainować na pytaniach generycznych.
2. Jako agent RAG, chcę retrievować sekcje z Glas I Hausrat jednocześnie gdy pytanie pyta o porównanie cross-branch, aby udzielić odpowiedzi graceful (granica pokrycia, upsell).
3. Jako agent RAG, chcę aby `RareTagMatcher` dopasowywał `domain_terms` z realnego `ExpandedQuery`, a nie z fake query, aby pytania z rareterms (np. "grobe Fahrlässigkeit") trafiały w tagi enrichmentu.
4. Jako agent RAG, chcę aby pusty wynik `RareTagMatcher` skutkował fallbackiem "szukaj wszędzie" (no-filter), a nie pustą listą, aby pytania generyczne bez pasujących tagów nadal dostawały kandydatów.
5. Jako agent RAG, chcę rozróżniać "DocFilter nie zidentyfikował produktu (no-filter)" od "DocFilter zidentyfikował produkt ale bez trafień (empty)", aby logika retrieval była poprawna.
6. Jako tester, chcę deterministycznych testów DocFilter (bez LLM) weryfikujących oba stany sentinela (`None` vs `frozenset()`), aby regresje były wykrywalne w CI.
7. Jako tester, chcę testu potwierdzającego że cross-branch query z `sparte_hint=None` zwraca wyniki ze wszystkich sparte, aby fix był weryfikowalny bez eval.
8. Jako tester, chcę testu potwierdzającego że pytanie generyczne z non-empty `domain_terms` (rareterms) retrievuje sekcje z pasującymi tagami, aby kaskada B+A była przetestowana end-to-end.

### Problem 2 — Critic

9. Jako agent RAG, chcę aby Critic PASS-ował odpowiedzi zawierające hedging ("nicht explizit erwähnt", "Quellen geben nicht an"), bo hedging ≠ halucynacja.
10. Jako agent RAG, chcę aby Critic PASS-ował niekompletne odpowiedzi (część pytania udzielona z kontekstu), bo niepełność ≠ wymyślone fakty.
11. Jako agent RAG, chcę aby Critic ABSTAIN-ował TYLKO gdy odpowiedź zawiera wymyślone kwoty, daty lub nazwane warunki nieobecne w kontekście.
12. Jako agent RAG, chcę aby REGEN powodował jednorazową regenerację + recheck, po czym recheck=ABSTAIN → ABSTAIN, recheck=REGEN → PASS (akceptuj regenerowaną), aby uniknąć nieskończonej pętli.
13. Jako agent RAG, chcę aby awaria primary Critica (wyjątek) skutkowała graceful PASS z oryginalną odpowiedzią i logiem warning, aby awaria bramki nie blokowała użytkownika.
14. Jako agent RAG, chcę opcjonalnego ensemble (drugi model, inna rodzina) weryfikującego odpowiedź po PASS primary, aby zwiększyć precision bez kosztu przy domyślnym wyłączeniu.
15. Jako inżynier eval, chcę flagi `enable_ensemble: bool` (domyślnie OFF) przekazywanej przez promptfoo provider, aby baseline eval był tani (1 call/query) a ensemble mierzony świadomie.
16. Jako tester, chcę testów Critica mockujących instructor-client (bez realnych wywołań LLM), aby TDD cykl był deterministyczny i bez kosztu API.
17. Jako tester, chcę testu potwierdzającego że REGEN → regeneracja → recheck=REGEN → PASS (akceptacja), aby logika "nie-blokuj przy REGEN recheck" była pokryta.
18. Jako tester, chcę testu potwierdzającego że awaria primary → graceful PASS (nie raise), aby behavior awarii był specyfikowany i niezłamany przez refaktor.

---

## Decyzje dotyczące wdrożenia

### DocFilter — sentinel i query flow

**Decyzja B (query flow):** Adaptery budowane raz (stateless). `ExpandedQuery` przepływa przez `retrieve_multi` → `doc_filter.filter(query_obj)`. Fake query znika. Adaptery czytają `sparte_hint` i `domain_terms` z przekazanego `query_obj`.

**Decyzja X (sentinel):** `CompositeDocFilter.filter()` zwraca `Optional[frozenset[str]]`:
- `None` = no-filter → retriever przeszukuje wszystkie sekcje
- `frozenset()` (non-None, pusty) = filtr aktywny, brak trafień → retriever zwraca `[]`

`ProductDetectorAdapter` bez `sparte` → `None` (no-filter). `RareTagMatcher` bez trafień → `None` (no-filter fallback). Composite: jeśli chociaż jeden adapter zwraca non-None → union; jeśli wszystkie `None` → `None` (no-filter).

**Decyzja kaskada B+A:** `RareTagMatcher` próbuje zawęzić po `domain_terms` z realnego `ExpandedQuery`. Gdy `domain_terms` puste LUB żaden tag nie pasuje → zwraca `None` (szukaj wszędzie). Dla pytań generycznych = poprawne zachowanie.

**Sygnatura `retrieve_multi`:** dostaje opcjonalny `query_obj: Optional[ExpandedQuery]` przekazywany do `doc_filter.filter()`. Jeśli `doc_filter is None` → positions = all. Jeśli `doc_filter.filter(query_obj) is None` → positions = all. Jeśli `frozenset()` → return [].

**Istniejący test do podziału:** `test_doc_filter_empty_set_returns_empty` (testuje `_AllowedDocFilter(set())`) → rozbić na dwa testy: `test_none_returns_all_sections` + `test_empty_frozenset_returns_empty`.

### Critic — DKV Full −Cerebras

**Structured output:** `instructor.from_groq(groq_client)` zamiast surowego `json_object` + `json.loads`. Model odpowiada przez Pydantic `CriticOutput` z auto-retry instructora.

**`CriticOutput` (Pydantic):**
```python
# Z prototypu DKV — koduje kształt odpowiedzi LLM
class CriticOutput(BaseModel):
    chain_of_thought: List[str]   # max 5 pozycji, short bullets
    reasoning: List[str]           # max 2 pozycje; 1 dla PASS
    verdict: Literal["PASS", "REGEN", "ABSTAIN"]
    confidence_score: float        # 0.0–1.0

    @field_validator("chain_of_thought", "reasoning", mode="before")
    @classmethod
    def _coerce_str_list(cls, v): ...  # flattenuje list-of-dict → list-of-str
```

**`CriticResult` (wyjście run_critic):**
```python
@dataclass
class CriticResult:
    verdict: CriticVerdict       # PASS | ABSTAIN (REGEN nie wychodzi na zewnątrz)
    reason: str
    confidence: float
    answer: str | None           # zregenerowana odpowiedź jeśli REGEN→PASS; None jeśli ABSTAIN
    retried: bool
    used_ensemble: bool
```

**Prompt anti-over-abstain:** port z DKV, dostosowany do ERGO (niemiecki produkt). Kluczowe reguły: "DEFAULT TO PASS"; hedging ≠ halucynacja; niepełna odpowiedź → PASS; ABSTAIN TYLKO gdy wymyślone kwoty/daty/warunki nieobecne w kontekście.

**REGEN loop:** `run_critic` — po REGEN wywołuje `generate_fn()` raz → recheck. recheck=ABSTAIN → ABSTAIN. recheck=REGEN lub PASS → PASS z nową odpowiedzią.

**Ensemble:** opcjonalny `ensemble_client`. PASS od primary → `evaluate(ensemble_client)`. ensemble=ABSTAIN → ABSTAIN. Awaria ensemble → graceful PASS (log warning).

**Graceful PASS:** wyjątek primary LUB ensemble → log warning + PASS z oryginalną odpowiedzią. Zastępuje network fallback Cerebras z DKV.

**Flaga ensemble:** `enable_ensemble: bool = False` w konstruktorze Critica LUB w `run_critic`. Domyślnie OFF.

**Ensemble model:** `REGISTRY["critic_ensemble"] = "llama-3.3-70b-versatile"` (Groq; inna rodzina niż primary `qwen/qwen3-32b`).

**`generate_fn` callback:** dostarczany przez RAGAssistant: `lambda: generator.generate(query, sections, mode)`. Używany tylko dla VERBATIM (COMPARE pomija critica).

**RAGAssistant:** po zmianie Critica — jeśli `critic_result.answer` jest non-None (REGEN→PASS) → użyj `critic_result.answer` zamiast `generated.answer`.

---

## Decyzje dotyczące testowania

**Zasada:** Testy weryfikują zachowanie przez publiczne interfejsy (nie internals). Test opisuje specyfikację — "system z `sparte_hint=None` zwraca wyniki". Przeżyje refaktor internalsów.

**DocFilter — testy deterministyczne (bez LLM, bez parquet):**
- Fixture: in-memory DataFrames z fiksowanymi doc_id/sparte/tarif/topic_tags.
- Weryfikacja `None` sentinela: `CompositeDocFilter` z adapターami zwracającymi `None` → wynik `None`.
- Weryfikacja `frozenset()` sentinela: adapter z aktywnym filtrem bez trafień → `frozenset()`.
- Retriever z `doc_filter=None` → wszystkie sekcje.
- Retriever z `doc_filter` zwracającym `None` → wszystkie sekcje.
- Retriever z `doc_filter` zwracającym `frozenset()` → `[]`.
- Kaskada: query z `domain_terms=["grobe Fahrlässigkeit"]` → sekcje z tym tagiem.
- Cross-branch: query z `sparte_hint=None` i `domain_terms=[]` → wszystkie sekcje (no-filter).

**Critic — testy mock instructor-client:**
- Mock `primary_client` zwracający `CriticOutput(verdict="PASS", ...)` → `CriticResult.verdict == PASS`.
- Mock primary=REGEN → `generate_fn()` wywołana → mock recheck=PASS → `CriticResult.verdict == PASS, retried=True`.
- Mock primary=REGEN → recheck=ABSTAIN → `CriticResult.verdict == ABSTAIN`.
- Mock primary raises Exception → `CriticResult.verdict == PASS, used_ensemble=False` (graceful PASS).
- enable_ensemble=True, primary=PASS, ensemble=ABSTAIN → `CriticResult.verdict == ABSTAIN, used_ensemble=True`.
- enable_ensemble=True, ensemble raises → graceful PASS.

**Istniejące wzorce testów:** `tests/test_retriever.py` (mock embedder + `_AllowedDocFilter`), `tests/test_doc_filter.py` (fixtures parquet). Nowe testy mogą używać in-memory fixtures zamiast parquetów dla deterministyczności.

---

## Poza zakresem

- Assert → llm-rubric: migracja `promptfooconfig.full.yaml` (osobna inicjatywa, issue F).
- Bug #24 (retrieval Fahrzeugwechsel): osobny issue po zbadaniu danych.
- Shadow ContextSelector O(n): optymalizacja pre-production.
- Windows Credential Manager / rotacja PAT: operacja manualna.
- Nowe Sparte lub Tarife: nie dodajemy.
- Per-intent konfiguracja Critica (DKV `_INTENT_CONFIGS`): nie przenosimy — ERGO ma jeden prompt.

---

## Dodatkowe uwagi

**Kontekst domenowy:** Glas to często Ergänzung do Hausrat (patrz `_CROSS_SELL_MAP`). Cross-branch query "Glas vs Hausrat" jest realnym use-caseM agenta — granica pokrycia / upsell suggestion. Retrieval musi dosięgnąć obu sparte; generator składa odpowiedź graceful.

**Dwie warstwy struktury:** Sparte (produkt: Kfz/Hausrat/Glas/Schmuck) vs Tarif (wariant w produkcie: Smart/Best/Spezial/Standard). `ProductDetectorAdapter` filtruje po Sparte (i opcjonalnie Tarif). Cross-branch = dwie Sparte → `sparte_hint=None`.

**Metryka po naprawie:** Oczekiwany wzrost pass-rate o ~15–20pp (6 empty retrieval + 15 critic over-abstain = 21 przypadków do naprawy). Re-eval: subset 21 (tani) → pełny 99 (z llm-rubric).

**Kolejność implementacji:** Problem 1 (deterministyczny, zero LLM cost) → Problem 2 (TDD z mock LLM). Problem 1 nie blokuje Problem 2 — można równolegle.
