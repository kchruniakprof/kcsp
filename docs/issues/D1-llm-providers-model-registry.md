# D1 — Abstrakcja providerów: `llm_providers` + `model_registry`

**Typ:** AFK  
**Blokowane przez:** Brak — można rozpocząć natychmiast  
**Dotyczy US:** 28–29  

---

## Co należy zbudować

Dwa nowe moduły infrastrukturalne portowane z wzorca DKV (`d:/_FUN/DKV_Belgium/calude/accuracy/src/`):

**`src/llm_providers.py`** — fabryki klientów:
- `groq_client(model: str | None = None) → instructor-wrapped client` — czyta `GROQ_API_KEY` z `.env`
- `openrouter_client(model: str | None = None) → instructor-wrapped client` — czyta `OPENROUTER_API_KEY` z `.env`
- Oba zwracają `instructor.from_openai(openai.OpenAI(...))` — kompatybilne z istniejącym `openrouter_client()` z `src/enrichment.py` (ujednolicenie)
- Żaden moduł biznesowy nie importuje `openai` ani `instructor` bezpośrednio — tylko przez te fabryki

**`src/model_registry.py`** — słownik krok → konfiguracja:
```python
REGISTRY = {
    "query_expansion":   {"provider": "groq",        "model": "meta-llama/llama-4-scout-17b-16e-instruct", ...},
    "enrichment":        {"provider": "openrouter",   "model": "meta-llama/llama-3.3-70b-instruct", ...},
    "llm_selector":      {"provider": "groq",         "model": "...", ...},
    "critic":            {"provider": "groq",         "model": "...", ...},
}
```
Zmiana modelu per-krok = zmiana tylko w tym słowniku. Każdy krok ma `provider`, `model`, opcjonalnie `temperature`, `max_retries`.

---

## Kryteria akceptacji

- [ ] `src/llm_providers.py`: `groq_client()` i `openrouter_client()` działają i zwracają instructor-wrapped klienty
- [ ] Brakujące klucze API → `RuntimeError` z czytelnym komunikatem (nie `KeyError`)
- [ ] `src/enrichment.py` zaktualizowany: używa `openrouter_client()` z `llm_providers` zamiast własnej definicji (DRY)
- [ ] `src/query_expansion.py` zaktualizowany: używa `groq_client()` z `llm_providers`
- [ ] `src/model_registry.py`: REGISTRY zawiera wpisy dla wszystkich kroków pipeline (`query_expansion`, `enrichment`, `llm_selector`, `critic`); model dla `llm_selector` i `critic` do wypełnienia w D4
- [ ] Żaden nowo tworzony moduł (D2–D6) nie importuje `openai`/`instructor`/`groq` bezpośrednio
- [ ] Testy: `groq_client()` z dummy key → patch env, weryfikacja że klient ma właściwy `base_url`; analogicznie `openrouter_client()`

## Blokowane przez

Brak — można rozpocząć natychmiast.
