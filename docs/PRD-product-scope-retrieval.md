# PRD — Product-Scope Extractor + Retrieval Improvements

> Wersja: 1.0 | Data: 2026-06-20
> Poprzedni PRD: `docs/PRD-docfilter-critic-overhaul.md` (F1–F5)
> Cel: implementacja A+B (product-scope extractor), C (cross-encoder reranker), D (index sections)

---

## Opis problemu

System RAG dla ERGO P&C (Kfz/Hausrat/Glas/Schmuck) zwraca błędne wyniki w czterech klasach przypadków:

1. **Multi-product queries** (comparison, cross-sell): `CompositeDocFilter` robi UNION adapterów — `ProductDetectorAdapter` (restrykcyjny) miesza się z `RareTagMatcherAdapter` (addytywny), przez co dokumenty Schmuck/Glas trafiają do pytań stricte Kfz.
2. **Tarif-level confusion**: brak poziomiu tarif w wykrywaniu produktu → pytania o `Best+Fahrraddiebstahl` retrieują z `Best` i `Best+Naturgefahren` jednocześnie.
3. **Missing sections in top-5**: sekcje odpowiadające pytaniu są w indeksie, ale nie trafiają do top-5 bo `top_k=5` jest zbyt wąskie — cross-encoder mógłby je awansować z puli 20.
4. **Index pollution**: sekcje będące stronicowanym indeksem alfabetycznym (165–168) są oznaczone jako `is_retrieval_unit=True`, co obniża jakość puli kandydatów.

---

## Rozwiązanie

### D — Index section filter (build_parquets.py)
Sekcje-indeksy (alfabetyczne, bez treści ubezpieczeniowej) są wykluczone z retrieval units na etapie budowania parquet. Reguła deterministyczna: heading po `##` = pojedyncza litera A–Z **lub** body >50% matchuje page-ref `\([A-Z]+\.?\d*\)\s*\d+`.

### A+B — Product-Scope Extractor
Zastąpienie union-semantyki `CompositeDocFilter` gate-semantyką:
- `ProductDetectorAdapter` staje się **hard gate** (restryktor — zawęża zbiór kandydatów do jednej sparte/tarif).
- `RareTagMatcherAdapter` staje się **narrow-within** (zawęża *wewnątrz* gate, nie rozszerza poza nim).
- Composition: `narrowed = gate ∩ rare; final = narrowed if narrowed else gate`.

`ExpandedQuery` zyskuje pole `sparte_hints: list[Literal["Kfz","Hausrat","Glas","Schmuck"]]` (lista, nie single-value) + tarif wykrywany deterministycznie word-boundary matchem na `normalized_query`. Prompt `QueryExpansion` dostaje few-shot dla multi-sparte + usunięcie błędnej reguły `"null=cross-branch"`.

### C — Cross-Encoder Reranker
Dwa etapy retrieval: pool top-k=20 (bi-encoder) → reranker → top-5 (cross-encoder). Model: `BAAI/bge-reranker-v2-m3`. Nowy komponent `CrossEncoderReranker` z interfejsem zgodnym z `ContextPruner` (inject pattern).

---

## Historie użytkownika

### D — Index sections

1. Jako agent ERGO chcę, żeby wyniki retrieval nie zawierały wpisów z alfabetycznego indeksu produktu (§ A, B, C...), aby odpowiedzi były oparte wyłącznie na treści merytorycznej.
2. Jako deweloper chcę regułę wykluczania sekcji-indeksów w `build_parquets.py`, aby była deterministyczna i nie wymagała ręcznego utrzymania listy ID.
3. Jako deweloper chcę, żeby parquet rebuild automatycznie walidował zakres `is_retrieval_unit` count po zmianie reguły, aby wykryć nieoczekiwane regresje.
4. Jako deweloper chcę test jednostkowy dla reguły index-section, który weryfikuje True/False dla konkretnych headingów i body, aby reguła nie regresowała po przyszłych zmianach parsera.

### A+B — Product-Scope Extractor

5. Jako agent ERGO zadający pytanie wyłącznie o Kfz chcę, żeby retrieval nie zwracał sekcji z dokumentów Hausrat/Glas/Schmuck, aby nie otrzymywać mylących informacji o innych produktach.
6. Jako agent ERGO zadający pytanie o konkretny tarif Kfz (np. "Spezial") chcę, żeby retrieval był zawężony do dokumentów tego tarifu, a nie wszystkich Kfz, aby odpowiedź była precyzyjna.
7. Jako agent ERGO porównujący dwa produkty (np. "jaka różnica między Hausrat Best a Glas?") chcę, żeby retrieval obejmował sekcje z obu produktów jednocześnie, aby odpowiedź zawierała materiał do porównania.
8. Jako agent ERGO pytający o compound tarif (np. "Best+Fahrraddiebstahl") chcę, żeby retrieval był zawężony do dokumentów tego compound tarifu (nie do bazowego Best), aby nie otrzymywać informacji z innego tarifu.
9. Jako agent ERGO pytający o produkt Glas z kontekstem mieszkaniowym chcę, żeby system uwzględnił też Hausrat jako related_sparte, ale tylko gdy query to sugeruje, aby unikać szumu przy pytaniach czysto-glassowych.
10. Jako agent ERGO chcę, żeby `normalized_query` (wersja po normalizacji do niemieckiego) była używana do tarif-match, a nie raw query, aby działo się poprawnie dla pytań zadanych po polsku lub angielsku.
11. Jako deweloper chcę field `sparte_hints: list` zamiast `sparte_hint: Optional[str]` w `ExpandedQuery`, aby model mógł zwrócić ≥2 sparte dla pytań comparison/cross-sparte.
12. Jako deweloper chcę, żeby `ExpandedQuery` zawierał compat property `primary_sparte` (first of sparte_hints), aby backward-compatibility z cross-sell logic w `RAGAssistant` nie wymagała refaktoru.
13. Jako deweloper chcę, żeby wykrywanie tarifu było deterministycznym word-boundary matchem (nie LLM), aby działało przewidywalnie i deterministycznie w testach.
14. Jako deweloper chcę, żeby aliasy rider-tokenów były generowane z definicji produktu (`split("+")` na tarif string), aby nie wymagały ręcznego utrzymania w kodzie.
15. Jako deweloper chcę walidację `sparte_hints` w `ExpandedQuery` (dedup, cap ≤4, OOS→[]), aby LLM nie mógł zwrócić nieprawidłowych wartości.
16. Jako deweloper chcę few-shot przykłady multi-sparte w promptcie `QueryExpansion` (≥2 przykłady comparison/cross), aby niezawodność wykrywania multi-sparte była wysoka.
17. Jako deweloper chcę usunąć regułę `"null=cross-branch or unclear"` z `_SYSTEM_PROMPT` w `query_expansion.py`, aby LLM nie był instruowany zwracać null dla comparison (błąd z poprzedniej implementacji).
18. Jako deweloper chcę gate-semantykę w `CompositeDocFilter` (intersection nie union gdy gate ∩ rare), aby `RareTagMatcherAdapter` nie mógł rozszerzyć scope poza sparte/tarif gate.
19. Jako deweloper chcę metodę `resolve_doc_set(sparte_hints, tarif, documents_df)` wyodrębniającą logikę mapowania sparte+tarif → doc_ids, aby była testowalna w izolacji.
20. Jako deweloper chcę test dla warunku related_sparte safety-net (trigger: Glas/Schmuck ∧ query wspomina Hausrat/Haushalt/Wohnung ∧ Hausrat nie w sparte_hints), aby trigger był wąski i nie wprowadzał szumu.

### C — Cross-Encoder Reranker

21. Jako agent ERGO chcę, żeby system znajdował właściwe sekcje nawet jeśli są na pozycjach 6–20 w bi-encoderze, aby cross-encoder mógł je awansować na podstawie pełnego query+doc scoring.
22. Jako agent ERGO zadający pytanie z negacją (np. "czy X NIE jest objęte?") chcę, żeby retrieval preferował sekcje faktycznie omawiające wyklucenia, bo cross-encoder widzi query+doc razem — bi-encoder nie rozróżnia negacji.
23. Jako deweloper chcę `CrossEncoderReranker` jako opcjonalny komponent inject do `Retriever`, analogicznie do `ContextPruner`, aby był wymienialny i testowalny w izolacji.
24. Jako deweloper chcę, żeby `Retriever.retrieve_multi` przyjmował parametr `pool_k` (domyślnie 20) niezależny od `top_k` (domyślnie 5), aby pool był zawsze szerszy niż finalne wyniki.
25. Jako deweloper chcę wpis dla modelu reranker w `model_registry.py`, aby model był konfigurowalny w jednym miejscu.
26. Jako deweloper chcę, żeby `CrossEncoderReranker` ładował model leniwie (przy pierwszym wywołaniu), aby import `retriever.py` nie powodował pobierania modelu na starcie.

---

## Decyzje implementacyjne

### D.1 — Reguła index-section
Sekcja ma `is_retrieval_unit=False` gdy **którekolwiek** z:
- heading po `##` to dokładnie jedna litera A–Z (regex: `^[A-Z]$` po strippowaniu)
- body zawiera ≥50% linii pasujących do page-ref: `\([A-Z]+\.?\d*\)\s*\d+`

Reguła aplikowana w `build_parquets.py` po obliczeniu `is_retrieval_unit` z hierarchii (L1/L2 logika), jako override krok.

### A+B.1 — sparte_hints w ExpandedQuery
```python
sparte_hints: list[Literal["Kfz", "Hausrat", "Glas", "Schmuck"]] = Field(
    default_factory=list,
    description="List of insurance branches; ≥2 for comparison/cross-sell; empty if OOS",
)

@property
def primary_sparte(self) -> Optional[str]:
    return self.sparte_hints[0] if self.sparte_hints else None
```
Walidator: dedup zachowując kolejność, cap ≤4, filtruje OOS wartości.

### A+B.2 — Tarif detection (deterministic)
```python
def _detect_tarif(normalized_query: str, tarif_names: list[str]) -> Optional[str]:
    # longest-first, rider tokens from split("+"), word-boundary \b
    candidates = sorted(tarif_names, key=len, reverse=True)
    for t in candidates:
        tokens = [t] + [part for part in t.split("+") if part != t.split("+")[0]]
        for tok in tokens:
            if re.search(rf"\b{re.escape(tok)}\b", normalized_query, re.IGNORECASE):
                return t
    return None
```
Uwaga z prototypu: `split("+")` na `"Best+Fahrraddiebstahl"` daje `["Best", "Fahrraddiebstahl"]`. Alias `Fahrraddiebstahl`→compound tarif. Base token `"Best"` NIE jest samodzielnym aliasem (longest-match wins).

### A+B.3 — Gate composition w CompositeDocFilter
```
gate_result = ProductDetectorAdapter(sparte_hints, tarif).filter(query)
rare_result = RareTagMatcherAdapter(...).filter(query)

if gate_result is None:
    final = None  # no sparte known → no filter
elif rare_result is None:
    final = gate_result  # no rare signal → use gate as-is
else:
    narrowed = gate_result & rare_result
    final = narrowed if narrowed else gate_result  # rare cross-sparte → ignoruj
```

### A+B.4 — resolve_doc_set logika
- `sparte_hints=[]` → `None` (no filter)
- `sparte_hints=["Kfz"]`, `tarif=None` → wszystkie docs dla Kfz
- `sparte_hints=["Kfz"]`, `tarif="Spezial"` → doc_ids z `sparte=Kfz AND tarif=Spezial`
- `sparte_hints=["Hausrat", "Glas"]` → union docs dla obu sparte (comparison case)
- related_sparte safety-net: dodaje Hausrat docs gdy (Glas lub Schmuck ∈ sparte_hints) ∧ query_contains_hausrat_keywords ∧ ("Hausrat" ∉ sparte_hints)

### A+B.5 — Prompt changes
Usunąć z `_SYSTEM_PROMPT`:
```
- null: cross-branch or unclear
```
Dodać:
```
- return ALL detected Spartes; for comparison queries return ≥2 Spartes
- OOS query → empty list []
```
Dodać 2–3 few-shot przykłady (user→assistant) dla multi-sparte w messages list.

### C.1 — CrossEncoderReranker interface
```python
class CrossEncoderReranker:
    def __init__(self, model_name: str = REGISTRY["reranker"]) -> None: ...
    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        # cross-encoder scores pairs (query, doc.heading + doc.markdown[:512])
        # returns sorted desc by reranker score
```
`Retriever.retrieve_multi` zyskuje parametr `pool_k: int = 20`. Gdy `reranker` inject: pobiera `pool_k` kandydatów bi-encoderem → reranker → top `top_k`.

### C.2 — Model registry
```python
"reranker": "BAAI/bge-reranker-v2-m3",
```

### RAGAssistant — sparte_hints przekazanie
`ProductDetectorAdapter` otrzymuje `sparte_hints` (lista) zamiast `sparte_hint` (string). Cross-sell logic używa `primary_sparte` compat property.

---

## Decyzje testowe

Dobry test weryfikuje **zachowanie zewnętrzne** modułu przez jego publiczny interfejs — nie mockuje wewnętrznych wywołań LLM (chyba że testuje sam prompt). Testy unitarne mogą mockować `documents_df` / `sections_df` przez in-memory DataFrame.

### Moduły testowane (nowe/zmienione testy):

**`tests/test_build_parquets.py`**
- Reguła D: `is_index_section(heading, body)` → True/False dla konkretnych par (heading="A", body=index-like vs normal body)
- Integracja: po `build()` na testowym corpus, sekcje-indeksy mają `is_retrieval_unit=False`

**`tests/test_query_expansion.py`**
- `sparte_hints` validator: dedup, cap, OOS→[]
- `primary_sparte` property: first element / None dla pustej listy
- Tarif detection: word-boundary match, longest-first, alias rider, base-token nie aliasowany solo

**`tests/test_doc_filter.py`**
- `resolve_doc_set`: każda z 5 gałęzi logiki (no-sparte, single-sparte-no-tarif, single-sparte-tarif, multi-sparte, related-sparte net)
- Gate composition: rare cross-sparte → trzyma gate (nie rozszerza)
- Gate composition: rare ∩ gate → zawęża
- Gate composition: gate=None → None (no filter)
- related_sparte trigger: wąski (Glas+hausrat-keyword) vs brak triggera (Glas bez hausrat-keyword)

**`tests/test_retriever.py`**
- `CrossEncoderReranker.rerank`: poprawna kolejność na mock pairach (mocker cross-encoder score)
- `Retriever.retrieve_multi` z rerankerem: pool_k > top_k → wyniki ≤ top_k
- Lazy load: import Retriever nie ładuje modelu (sprawdzone przez mock patch)

Wzorce istniejących testów do naśladowania: `tests/test_doc_filter.py` (in-memory DataFrame), `tests/test_context_pruner.py` (inject pattern).

---

## Poza zakresem

- **COMPARISON/ADVISORY intent + synthesis mode** — wydzielony temat (#26 e-bike advisory). Obecny PRD obejmuje tylko scope extraction, nie zmienia logiki generatora dla comparison.
- **L3 rider-trap** ("mam Best, czy Fahrraddiebstahl objęty?") — known limitation. Wymaga NLU posiadanie-vs-zapytanie; poza zakresem.
- **Critic over-abstain (E)** — defer po re-eval. Jeśli A+C nie naprawią exclusion_18/19, dodać regułę do `critic.py` w osobnym issue.
- **P4 shadow ContextSelector O(n)** — trywialny dict, pre-produkcja, osobne issue.
- **P3 git credentials** — ręczna akcja (Windows Credential Manager). PAT widoczny w historii czatu — zrotować.
- **Full eval_set.yaml audit** — wiele `source_section_id` jest błędnych (generator eval-setu mylił sekcje). Osobne issue.

---

## Kolejność implementacji

```
D (build_parquets.py + rebuild parquet)
  ↓
A+B (doc_filter.py + query_expansion.py + ragassistant.py)
  ↓
C (retriever.py + model_registry.py)
  ↓
re-eval (subset21 + full)
  ↓
E (tylko jeśli re-eval wciąż pokazuje over-abstain)
```

**Uwaga:** D zmienia parquet → po D trzeba przebudować parquet i przeliczone embeddingi (ADR-007) zanim A+B+C będą mogły być przetestowane end-to-end.

---

## Pliki referencyjne

- `improvement.md` — pełna analiza decyzji architektonicznych z sesji /grill-me
- `analiza.md` — analiza 8 failujących case'ów per-case
- `problemy.md` — P1–P6 zidentyfikowane problemy pipeline
- `docs/adr/ADR-007-build-pipeline-separation.md` — build pipeline separation
- `docs/adr/ADR-008-retriever-architecture.md` — retriever architecture
- `eval_set_subset21.yaml` — test cases (po poprawkach eval-set z 2026-06-20)
