# Bieżące problemy — KCSP pipeline

> Dokument wygenerowany: 2026-06-20  
> Kontekst: sesja po E1 (eval_full z enriched parquets)  
> Stan repozytorium: branch `master`, ostatni commit `f28178f`

---

## Problem 1 — DocFilter zwraca pustą listę dla cross-branch COMPARISON ⚠️ PRIORYTET

**Objaw:** Dwa pytania COMPARISON z `sparte_hint=null` (cross-branch) abstainują z `src=[]` mimo że sekcje istnieją.

Konkretne zapytania:
- `Welche Unterschiede bestehen zwischen der Glasversicherung und der Hausratversicherung...`
- `Welche Versicherung bietet besseren Schutz für wertvolle Gegenstände im Haushalt, die Glas...`

**Przyczyna — dwa niezależne bugi:**

### Bug A — `src/retriever.py` linia ~122–128
```python
# OBECNY (błędny):
if not allowed_ids:
    return []   # ← traktuje "no-filter signal" jako "brak wyników"

# POPRAWNY:
if not allowed_ids:
    positions = list(range(len(self._sections)))  # fall-through: brak filtru = wszystkie sekcje
```
`CompositeDocFilter` zwraca `frozenset()` jako sygnał "no-filter" (per docstring), ale retriever interpretuje to jako brak kandydatów.

### Bug B — `src/doc_filter.py` + `src/ragassistant.py`
`RareTagMatcherAdapter.filter()` czyta `domain_terms` z obiektu query przez `getattr(query, "domain_terms", [])`. Retriever przekazuje fake query:
```python
type("_Q", (), {"sparte_hint": None, "domain_terms": []})()
```
`domain_terms` zawsze `[]` → RareTagMatcher nigdy nie matchuje przez real domain terms.

**Fix:** Przekazać `domain_terms` przez konstruktor adaptera (wzorzec jak `ProductDetectorAdapter.sparte`):
```python
# src/doc_filter.py
class RareTagMatcherAdapter:
    def __init__(self, sections_df, subsections_df, domain_terms=None):
        self._domain_terms = domain_terms or []

# src/ragassistant.py
RareTagMatcherAdapter(
    self._sections_df, self._subsections_df,
    domain_terms=expanded.domain_terms,   # ← przekazać z ExpandedQuery
)
```

**Dotknięte pliki:** `src/retriever.py`, `src/doc_filter.py`, `src/ragassistant.py`  
**Testy do napisania:** `tests/test_retriever.py`, `tests/test_doc_filter.py`  
**Test istniejący który ZMIENIA semantykę:**
```python
# tests/test_retriever.py — linia 173
def test_doc_filter_empty_set_returns_empty(retriever):  # ← ten test będzie PASS po fixie Bug A
```
Uwaga: ten test testuje `_AllowedDocFilter(set())` czyli celowo pusty zbiór — to inny przypadek niż "no-filter signal". Po poprawce semantyka powinna być:
- `frozenset()` od CompositeDocFilter = no-filter → wszystkie sekcje
- Ale test używa `_AllowedDocFilter` który zwraca `frozenset()` z innego powodu — wymaga przemyślenia (może osobna klasa `NoFilterSentinel` albo `Optional[frozenset]`).

---

## Problem 2 — Critic za agresywnie abstainuje

**Objaw:** 15/24 failów w `results_full_enriched.json` to `abstained=True` z niepustym `src` — Critic ma kandydatów ale decyduje "nie wiem".

**Breakdown po intentach:**
| Intent | Fail (abstain) |
|--------|---------------|
| EXCLUSION_QUERY | 7 |
| CLAIMS_PROCEDURE | 5 |
| COVERAGE_QUERY | 2 |
| COMPARISON | 1 |

**Hipoteza:** System prompt Critic ma zbyt wysoki próg pewności dla pytań o wykluczenia i procedury. Sekcje są trafne, ale odpowiedź wymaga syntezy z wielu miejsc → Critic abstainuje z powodu "niepełności".

**Dotknięte pliki:** `src/critic.py`  
**Dane referencyjne:** `results_full_enriched.json` (99 pytań), `eval_full_enriched.log`

---

## Problem 3 — Windows Credential Manager blokuje git push

**Objaw:** `git push origin master` daje 403 bo Credential Manager ma zapisane credentials dla `kchruniak` (nie `kchruniakprof`).

**Obecny workaround:** Tymczasowe osadzenie PAT w remote URL → push → przywrócenie czystego URL.

**Poprawne rozwiązanie:** Wyczyścić `kchruniak` z Windows Credential Manager:
```
Ustawienia systemu → Zarządzanie poświadczeniami → Poświadczenia Windows → github.com → usuń
```
Lub skonfigurować Git Credential Manager z właściwym kontem.

**Bezpieczeństwo:** PAT `ghp_Miq…` był widoczny w historii czatu — **należy go zrotować** w GitHub Settings → Developer settings → Personal access tokens.

---

## Problem 4 — Shadow ContextSelector w providerze jest O(n) per query

**Plik:** `src/promptfoo_provider.py` funkcja `call_api()`

Dla każdego zapytania iteruje przez `r._sections` żeby znaleźć sekcje po `section_id`:
```python
sec = next((s for s in r._sections if s["section_id"] == sec_id), None)
```
Przy 370 sekcjach i 5 sources = 1850 porównań per query. Bez znaczenia dla eval (wolny), ale do poprawki przed produkcją.

**Fix:** Zbudować `dict[int, dict]` w `_get_rag()` raz i reużywać.

---

## Stan E1 po sesji

| Kryterium akceptacji | Status |
|---------------------|--------|
| eval_full bez błędów | ✅ |
| hit-rate ≥ baseline | ✅ +2.1pp (73.7% → 75.8%) |
| Core-4 enrichment pomógł | ✅ |
| top_score zebrane | ✅ 74/99 pytań |
| EMBED_THRESHOLD wybrany | ✅ → `None` (bimodalna dystrybucja, żaden próg nie pomaga) |
| abstain-rate OOS ≥ 80% | ✅ 100% (przez QueryExpansion, nie ContextSelector) |
| hit-rate nie spada > 5pp | ✅ -1.0pp in-scope coverage |
| wyniki w `results_full_enriched.json` | ✅ |

---

## Sugerowane umiejętności dla następnej sesji

- `/tdd` — implementacja fixu DocFilter (Problem 1): napisać testy RED → fix retriever → fix RareTagMatcher → fix ragassistant → GREEN
- `/tdd` — kalibracja Critic (Problem 2): analiza failing cases, zmiana system prompt, re-eval na subset

## Pliki referencyjne

- Issue E1: `docs/issues/E1-eval-calibration.md`
- DocFilter: `src/doc_filter.py`, `src/retriever.py`, `src/ragassistant.py`
- Testy: `tests/test_doc_filter.py`, `tests/test_retriever.py`, `tests/test_ragassistant.py`
- Wyniki eval: `results_full_enriched.json`, `results_full.json` (baseline)
- ADR provider topology: `docs/adr/ADR-009-provider-topology.md`
