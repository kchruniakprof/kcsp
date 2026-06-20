# F4 — Critic `run_critic`: REGEN loop + graceful PASS

**Typ:** AFK  
**Blokowane przez:** F3  
**Dotyczy US:** 12, 13, 17, 18  
**PRD:** `docs/PRD-docfilter-critic-overhaul.md`

---

## Co należy zbudować

Dodać funkcję `run_critic()` orkiestrującą pełny cykl: primary evaluate → REGEN loop → graceful PASS przy awarii. REGEN nie wychodzi na zewnątrz — `CriticResult` zawiera tylko `PASS` lub `ABSTAIN`.

**`run_critic()` — logika:**

```
primary = evaluate(query, answer, sections, client)

jeśli primary raises → log warning → return CriticResult(PASS, answer=original, retried=False)

jeśli primary.verdict == REGEN:
    new_answer = generate_fn()
    recheck = evaluate(query, new_answer, sections, client)
    jeśli recheck raises → graceful PASS z new_answer
    jeśli recheck.verdict == ABSTAIN → return CriticResult(ABSTAIN, answer=None, retried=True)
    inaczej (REGEN lub PASS) → return CriticResult(PASS, answer=new_answer, retried=True)

jeśli primary.verdict == ABSTAIN:
    return CriticResult(ABSTAIN, answer=None, retried=False)

jeśli primary.verdict == PASS:
    return CriticResult(PASS, answer=original, retried=False)
```

**Uwaga:** `generate_fn: Callable[[], str]` — callback dostarczany przez wywołującego. W tym issue interfejs funkcji jest wystarczający; wiring do RAGAssistant = F5.

**Graceful PASS:** wyjątek primary (dowolny Exception) → `log.warning(...)` + `CriticResult(PASS, answer=original)`. Awaria bramki ≠ blokada użytkownika.

**`CriticResult` (finalna forma po F3+F4):**
```python
@dataclass
class CriticResult:
    verdict: CriticVerdict    # PASS | ABSTAIN (REGEN nie wychodzi)
    reason: str
    confidence: float
    answer: str | None        # non-None gdy PASS (oryginalna lub zregenerowana)
    retried: bool
    used_ensemble: bool = False
```

---

## Kryteria akceptacji

- [ ] `run_critic(query, answer, sections, client, generate_fn)` istnieje i zwraca `CriticResult`
- [ ] primary PASS → `CriticResult.verdict == PASS`, `answer == original`, `retried == False`
- [ ] primary ABSTAIN → `CriticResult.verdict == ABSTAIN`, `answer == None`, `retried == False`
- [ ] primary REGEN → `generate_fn()` wywołana dokładnie raz
- [ ] primary REGEN + recheck PASS → `CriticResult(PASS, answer=new_answer, retried=True)`
- [ ] primary REGEN + recheck REGEN → `CriticResult(PASS, answer=new_answer, retried=True)` (REGEN recheck = PASS)
- [ ] primary REGEN + recheck ABSTAIN → `CriticResult(ABSTAIN, answer=None, retried=True)`
- [ ] primary raises Exception → `CriticResult(PASS, answer=original, retried=False)`, log warning, brak re-raise
- [ ] primary REGEN + recheck raises → `CriticResult(PASS, answer=new_answer, retried=True)` (graceful)
- [ ] `generate_fn()` NIE wywołana gdy primary PASS lub ABSTAIN
- [ ] Testy deterministyczne (mock `evaluate()`, mock `generate_fn`) pokrywają wszystkie ścieżki powyżej
- [ ] `src/ragassistant.py` NIC nie zmienione w tym issue

## Blokowane przez

- F3 (`CriticOutput` Pydantic + `evaluate()` z instructor musi istnieć)
