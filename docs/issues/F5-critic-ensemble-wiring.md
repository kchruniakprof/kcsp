# F5 — Critic ensemble + pełne wiring (RAGAssistant + promptfoo provider)

**Typ:** AFK  
**Blokowane przez:** F4  
**Dotyczy US:** 14, 15  
**PRD:** `docs/PRD-docfilter-critic-overhaul.md`

---

## Co należy zbudować

Domknąć Critica: dodać opcjonalny ensemble (drugi model) do `run_critic`, zaktualizować `model_registry`, przepiąć `RAGAssistant` na `run_critic` (z `generate_fn`), zaktualizować `promptfoo_provider` (instructor client + flaga ensemble).

**Ensemble w `run_critic`:**

Dodać opcjonalne parametry `ensemble_client` i `enable_ensemble: bool = False`. Po PASS od primary (lub PASS po REGEN):
```
jeśli enable_ensemble i ensemble_client:
    ensemble_out = evaluate(query, current_answer, sections, ensemble_client, model=ENSEMBLE_MODEL)
    jeśli ensemble_out raises → log warning, ignore (graceful PASS)
    jeśli ensemble_out.verdict == ABSTAIN → return CriticResult(ABSTAIN, answer=None, used_ensemble=True)
    inaczej → kontynuuj PASS
return CriticResult(PASS, answer=current_answer, used_ensemble=used_ensemble)
```

**`model_registry.py`:** dodać klucz `"critic_ensemble": "llama-3.3-70b-versatile"` (Groq, inna rodzina niż primary `qwen/qwen3-32b`).

**`RAGAssistant`:** zastąpić `self._critic.evaluate(...)` wywołaniem `run_critic(...)` z:
- `generate_fn = lambda: self._generator.generate(expanded.normalized_query, generator_sections, mode=mode).answer`
- `enable_ensemble=self._enable_ensemble` (nowy parametr konstruktora, domyślnie `False`)
- Jeśli `critic_result.answer` jest non-None → użyć jako `answer_text` zamiast `generated.answer`
- REGEN loop działa tylko dla VERBATIM (COMPARE pomija critica — bez zmian)

**`promptfoo_provider.py`:** 
- Critic dostaje instructor-client: `instructor.from_groq(Groq(api_key=...))`
- Przekazać `enable_ensemble=False` (lub z `os.environ.get("ENABLE_ENSEMBLE", "false") == "true"`)

---

## Kryteria akceptacji

- [ ] `REGISTRY["critic_ensemble"] = "llama-3.3-70b-versatile"` w model_registry
- [ ] `run_critic` przyjmuje `ensemble_client` i `enable_ensemble: bool = False`
- [ ] `enable_ensemble=False` (domyślnie) → ensemble_client nigdy nie wywołany
- [ ] `enable_ensemble=True`, primary PASS, ensemble ABSTAIN → `CriticResult(ABSTAIN, used_ensemble=True)`
- [ ] `enable_ensemble=True`, ensemble raises → graceful PASS, `CriticResult(PASS, used_ensemble=False)` + log warning
- [ ] `RAGAssistant` wywołuje `run_critic` (nie `critic.evaluate`)
- [ ] `RAGAssistant` przekazuje `generate_fn` callback
- [ ] Gdy `critic_result.answer` non-None → `answer_text = critic_result.answer` (zregenerowana odpowiedź)
- [ ] `promptfoo_provider.py` przekazuje instructor-client do Critica
- [ ] `enable_ensemble` sterowany przez zmienną środowiskową lub stałą (domyślnie OFF)
- [ ] Testy ensemble: enable_ensemble=True + mock ensemble ABSTAIN → ABSTAIN, `used_ensemble=True`
- [ ] Testy ensemble: enable_ensemble=True + ensemble raises → PASS, `used_ensemble=False`
- [ ] Istniejące testy ragassistant PASS (lub zaktualizowane)

## Blokowane przez

- F4 (`run_critic` z REGEN loop musi istnieć)
