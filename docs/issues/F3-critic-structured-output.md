# F3 — Critic structured output + anti-over-abstain prompt

**Typ:** AFK  
**Blokowane przez:** Brak — można rozpocząć natychmiast  
**Dotyczy US:** 9–11, 16  
**PRD:** `docs/PRD-docfilter-critic-overhaul.md`

---

## Co należy zbudować

Przepisać warstwę `evaluate()` w `src/critic.py` na structured output z `instructor` + Pydantic, zastępując surowy `json_object` + `json.loads`. Jednocześnie zastąpić prompt ostrożnościowy promptem anti-over-abstain z projektu DKV (dostosowanym do ERGO).

**`CriticOutput` (Pydantic model):**
```python
# Z prototypu DKV — koduje kształt odpowiedzi LLM
class CriticOutput(BaseModel):
    chain_of_thought: List[str]   # max 5 pozycji, short bullets (≤15 słów)
    reasoning: List[str]           # max 2 pozycje; 1 dla PASS
    verdict: Literal["PASS", "REGEN", "ABSTAIN"]
    confidence_score: float        # 0.0–1.0, ge=0.0, le=1.0

    @field_validator("chain_of_thought", "reasoning", mode="before")
    @classmethod
    def _coerce_str_list(cls, v): ...
    # flattenuje list-of-dict → list-of-str (anti-crash gdy LLM zwróci {"claim": "..."})
```

**Prompt anti-over-abstain (kluczowe reguły):**
- "DEFAULT TO PASS"
- Hedging ("nicht explizit erwähnt", "Quellen geben nicht an", "nicht separat aufgeführt") ≠ halucynacja → PASS
- Niekompletna odpowiedź → PASS (nie ABSTAIN)
- ABSTAIN TYLKO gdy odpowiedź zawiera wymyślone kwoty, daty, lub nazwane warunki NIEOBECNE w kontekście
- REGEN TYLKO gdy konkretny błąd faktyczny, który kontekst może poprawić

**`evaluate()` — wymagania:**
- Używa `instructor`-client (nie surowy `Groq()`): `instructor.from_groq(client)`
- `response_model=CriticOutput` (auto-retry przy błędzie parsowania)
- Zwraca `CriticOutput`

**`CriticResult` — minimalna zmiana kształtu:** dodać pole `answer: str | None` (potrzebne w F4 dla REGEN path). W tym issue `answer` zawsze `None` (wypełniane w F4).

**Instalacja dependency:** `instructor` musi być w `requirements.txt` / `pyproject.toml`.

---

## Kryteria akceptacji

- [ ] `instructor` dodany do zależności projektu
- [ ] `CriticOutput` Pydantic z `chain_of_thought`, `reasoning`, `verdict`, `confidence_score` + `_coerce_str_list` validator
- [ ] `evaluate()` używa `instructor.from_groq(client)` zamiast surowego `Groq()`
- [ ] Prompt zawiera "DEFAULT TO PASS" i reguły anty-over-abstain dostosowane do ERGO
- [ ] `CriticResult` ma pole `answer: str | None` (domyślnie `None`)
- [ ] Testy z mock instructor-client (bez LLM): PASS verdict → `CriticResult.verdict == PASS`
- [ ] Testy: ABSTAIN verdict → `CriticResult.verdict == ABSTAIN`
- [ ] Testy: `_coerce_str_list` flattenuje list-of-dict do list-of-str
- [ ] Istniejące testy critic PASS (lub zaktualizowane do nowego interfejsu)
- [ ] `src/ragassistant.py` NIC nie zmienione w tym issue

## Blokowane przez

Brak — można rozpocząć natychmiast.
