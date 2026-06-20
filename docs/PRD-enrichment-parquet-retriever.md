# PRD — Enrichment + Parquet + Retriever Rebuild

> Źródła decyzji: [`HANDOFF_enrichment_2026-06-20.md`](../HANDOFF_enrichment_2026-06-20.md) + ADR-002 przez ADR-009 (`docs/adr/`).
> Tło systemu (pipeline, kontrakty, glosariusz DE): [`PRD.md`](../PRD.md). Ten dokument NIE powtarza tamtych decyzji.

---

## Opis problemu

Obecny pipeline RAG ma trzy nierozwiązane problemy blokujące jakość retrievalu:

1. **Brak enrichmentu semantycznego.** Embeddingi bazują na `heading + markdown[:512]` — surowym legalese, które produkuje wektory podobne do "generic insurance contract" zamiast odzwierciedlać konkretną treść §. Brak pól `title`, `description`, `questions` → retriever nie rozumie, jakie pytania sekcja odpowiada; `topic_tags` są wszędzie `[]` → Rare-tag Matcher jest martwy.

2. **Błędna granularność retrievalu.** Pool zawiera 53 L1-rodziców, których `markdown` jest sklejonym tekstem ich dzieci. Duplikat ten saturuje top_k fałszywymi pozycjami i uniemożliwia jednoznaczną cytację breadcrumb.

3. **Retriever bez progu i bez DocFiltra.** Nie ma podłogi similarności (zawsze zwraca top_k niezależnie od score), brak `DocFilter` jako izolowanego komponentu (filtr Sparte/Tarif jest inline), brak `ContextSelector` z confidence-path, brak pruner-a. Nic nie blokuje nisko-relevantnych sekcji zanim trafią do generatora.

---

## Rozwiązanie

Trzyetapowy rebuild (fazy A–C) prowadzący do przebudowy retrievera (faza D) i kalibracji progu (faza E):

- **Faza A:** Refaktor `build_parquets.py` — dodanie `is_retrieval_unit`, usunięcie embeddingu.
- **Faza B (cost-gated):** `enrich_sections.py` — batch LLM Core-4 dla ~370 liści.
- **Faza C:** `build_embeddings.py` — nowy skład embed-text (heading+title+desc+5Q+body[:400]).
- **Faza D:** Port retrievera z wzorca DKV (`DocFilter`, `ContextSelector`, `ContextPruner`), adaptowany do gwarancji verbatim KCSP.
- **Faza E:** Re-run `eval_full`, kalibracja progu similarności.

---

## Historie użytkownika

### Dane / build-time

1. Jako developer, chcę aby `build_parquets.py` emitował kolumnę `is_retrieval_unit` dla każdej sekcji, aby retriever i embedder mogły deterministycznie odfiltrować L1-rodziców bez re-parsowania.
2. Jako developer, chcę aby L1-rodzice (sekcje z subsekcjami) mieli `is_retrieval_unit=False`, a L1-liście i L2 mieli `is_retrieval_unit=True`, aby pula retrievalu wynosiła ~370, nie ~423.
3. Jako developer, chcę aby L1-rodzice pozostawali w parquet z `is_retrieval_unit=False`, aby można było z nich budować breadcrumb bez osobnego lookup.
4. Jako developer, chcę aby `build_parquets.py` NIE liczył embeddingów, aby Faza A była deterministyczna i kosztowała 0 PLN/$.
5. Jako developer, chcę walidacji po budowie: `sum(is_retrieval_unit) == 370 ± tolerancja`, aby ktokolwiek nie mógł cicho zmienić granularności.

### Enrichment

6. Jako developer, chcę skryptu `enrich_sections.py` iterującego tylko po `is_retrieval_unit=True`, aby nie płacić za enrichment L1-rodziców.
7. Jako developer, chcę checkpoint/resume w `enrich_sections.py` (na podstawie `section_id` lub `section_code`), aby przerwa wywołana błędem API lub limitem OpenRouter nie resetowała całego bacha.
8. Jako developer, chcę skip-done: sekcje już wzbogacone są pomijane przy kolejnym uruchomieniu, aby ponowne uruchomienie po przerwie nie duplikowało kosztów.
9. Jako developer, chcę aby przed startem enrichmentu skrypt wypisał szacowany koszt i pytał `[y/N]` (chyba że przekazano `--yes`), aby nigdy nie zacząć batch LLM bez świadomej zgody.
10. Jako developer, chcę aby wygenerowane pola Core-4 (`title`, `description`, `questions`, `topic_tags`) były zapisywane z powrotem do odpowiednich parquet, aby Faza C mogła z nich korzystać bez osobnego pipeline.
11. Jako developer, chcę aby prompt enrichmentu był po angielsku, a wszystkie pola wyjściowe — po niemiecku, aby utrzymać jakość instrukcji LLM bez ryzyka dryfu terminów prawniczych.
12. Jako developer, chcę walidacji pydantic na outputach enrichmentu (zakres `questions` 5-10, `topic_tags` List[str]), aby malformed LLM-response był retry-owany przez instructor automatycznie.

### Embedding

13. Jako developer, chcę skryptu `build_embeddings.py` wydzielonego od parsera i enrichmentu, aby można było przeliczyć embeddingi po zmianie składu embed-text bez ponownego enrichmentu (koszt LLM).
14. Jako developer, chcę aby embed-text dla każdej jednostki był złożony z: `heading + title + description + questions[:5] + body[:400]`, gdzie `body` to sanityzowany `markdown`, aby enrichment niósł ciężar semantyczny a raw legalese pełnił rolę kotwicy.
15. Jako developer, chcę aby `build_embeddings.py` iterował tylko po `is_retrieval_unit=True`, aby L1-rodzice nie wchodziły do indeksu wektorowego.
16. Jako developer, chcę walidacji kształtu i norm embeddingów po build (shape `(N, 1024)`, normy ≈ 1.0), aby błąd BGE-M3 nie skutkował cichym błędem retrievalu.

### DocFilter

17. Jako operator, chcę aby filtr dokumentów był zaimplementowany jako `DocFilter` Protocol z `CompositeDocFilter` (union frozenset `doc_id`), aby każdy adapter był testowany w izolacji i komponowany bez modyfikacji retriever core.
18. Jako operator, chcę `ProductDetectorAdapter`, który tłumaczy Sparte/Tarif z `QueryExpansion` na `frozenset[doc_id]` via `documents.parquet`, zastępując obecny inline-filter w `retriever.py`.
19. Jako operator, chcę `RareTagMatcherAdapter`, który mapuje `domain_terms` z `QueryExpansion` na `frozenset[doc_id]` via `topic_tags` w parquet, aby aktywować rzadko-tagowaną ścieżkę retrievalu gdy użytkownik pyta o konkretny termin (np. "GAP", "Tierbiss").
20. Jako operator, chcę aby `CompositeDocFilter` zwracał unię zbiorów adapterów, a nie iloczyn, aby pojedynczo słaby sygnał (topic_tag bez pewnego Sparte) nie zerował wyników.

### ContextSelector (LLM reranking)

21. Jako operator, chcę `ContextSelector` z primary `TopKReranker` i fallback `BruteForceReranker` (nad pełnym korpusem), aktywowanym gdy `confidence < próg`, wzorowanym na DKV `reranker_strategy.py`.
22. Jako operator, chcę aby `confidence < EMBED_THRESHOLD` w KCSP powodowało **abstain** (nie fallback do BruteForce jak w DKV), bo w kontekście ubezpieczeniowym niska pewność = ryzyko halucynacji.
23. Jako operator, chcę aby `EMBED_THRESHOLD` był konfigurowalny (env var / registry), NIE hardcoded, bo wartość jest kalibrowana na eval secie a nie kopiowana z DKV (BGE-M3 ≠ MiniLM, inne skale cosine).
24. Jako operator, chcę aby `LLMSelector` używał Groq jako providera (latency-sensitive runtime path), z modelem konfigurowanym w `model_registry.py`.

### Pruner + verbatim guarantee

25. Jako operator, chcę `ContextPruner` (zdaniowy) i `EmbeddingPruner` portowanych z DKV, z global bypass dla chunków < 2500 znaków i empty-guard.
26. Jako operator, chcę aby pruner produkował **dwa widoki** tego samego chunku: `pruned_text` (do LLM selectora/critic) i `verbatim_text` (oryginalne `markdown`), aby generator zawsze cytował pełny verbatim §, nie skróconą wersję.
27. Jako operator, chcę aby `verbatim_text` był nienaruszony przez cały pipeline (selector, pruner, critic), ponieważ generator cytuje wyłącznie z whitelisty `{markdown, heading, section_code}` i jakakolwiek modyfikacja łamie gwarancję kontraktu.

### Ragassistant + provider abstraction

28. Jako developer, chcę `llm_providers.py` z funkcjami fabrycznymi `groq_client()` i `openrouter_client()`, zwracającymi instructor-wrapped OpenAI-compatible clienty, aby żaden moduł biznesowy nie importował SDK bezpośrednio.
29. Jako developer, chcę `model_registry.py` mapującego krok pipeline → `{provider, model_id, params}`, aby zmiana modelu per-krok (np. po A/B teście) wymagała zmiany tylko w rejestrze, nie w logice.
30. Jako developer, chcę aby `ragassistant.py` był zaktualizowany do nowego interfejsu retriever (DocFilter → ContextSelector → dual-view), zachowując kontrakt `FinalAnswer` z breadcrumb i audit metadata.

### Kalibracja (Faza E)

31. Jako developer, chcę re-run `eval_full` po przebudowie retrievera, aby zmierzyć retrieval hit-rate z nowym składem embeddingów.
32. Jako developer, chcę skalibrować `EMBED_THRESHOLD` na eval secie: próg wybrać tam gdzie abstain-rate na pytaniach poza-zakresem jest wysoki, a hit-rate na pytaniach w-zakresie spada minimalnie.

---

## Decyzje wdrożeniowe

### Moduły

**Modyfikowane:**
- `src/build_parquets.py` — dodać kolumnę `is_retrieval_unit` (True = L2 ∪ L1-liście); usunąć embedder ze skryptu; zaktualizować walidację.
- `src/retriever.py` — usunąć inline Sparte/Tarif filter; przyjąć `DocFilter` z zewnątrz; wpiąć `ContextSelector`; emitować RetrievalResult z dwoma polami: `markdown` (verbatim) i `pruned_markdown` (dla LLM).

**Nowe:**
- `src/enrich_sections.py` — skrypt batchowy: load parquet → pętla `is_retrieval_unit` → `enrich_section()` → checkpoint → write back; cost-gate print+prompt.
- `src/build_embeddings.py` — skrypt: load parquet → `_embed_text(row)` → BGE-M3 → kolumna `embedding` → write back.
- `src/doc_filter.py` — `DocFilter` Protocol + `ProductDetectorAdapter` + `RareTagMatcherAdapter` + `CompositeDocFilter`.
- `src/llm_selector.py` — `ContextSelector`: `TopKReranker` primary + abstain-on-low-score; Groq via model_registry.
- `src/context_pruner.py` — zdaniowy pruner; dual-view output; bypass < 2500 chars; empty-guard.
- `src/embedding_pruner.py` — embedding-based pruner (port DKV); dual-view output.
- `src/llm_providers.py` — `groq_client()` + `openrouter_client()` factory functions.
- `src/model_registry.py` — dict: krok → `{provider, model_id, params}`.

### Kontrakt `is_retrieval_unit`

```python
# deterministyczna reguła — bez LLM:
is_retrieval_unit = (level == 2) or (level == 1 and len(children) == 0)
```

Liczba True w parquet po build: suma L1-liście + wszystkie L2 ≈ 370.

### Skład embed-text (ADR-006)

```
{heading}\n{title}\n{description}\n{q1}\n{q2}\n{q3}\n{q4}\n{q5}\n{markdown[:400]}
```

Brakujące pola pomijane (nie zastępowane placeholderem). `questions[:5]` — przy dostępności < 5 pytań użyj tyle ile jest.

### Dual-view chunk

`RetrievalResult` po przebudowie musi zawierać:
- `markdown` — pełna oryginalna treść (whitelist; idzie do generatora i usera)
- `pruned_markdown` — widok dla LLM (selector, critic); może być krótszy; NIGDY nie idzie do usera

### Checkpoint enrichmentu

Plik `parquet/enrichment_checkpoint.json`: `{section_id: true}` dla sekcji już wzbogaconych. Przy resume: load → skip sekcje z checkpoint → dopisz po każdym sukesie (nie batch-zapis na końcu).

### Kolejność faz

```
A (det., 0$) → B (LLM cost ⟵ GATE: explicit go) → C (det.) → D (dev) → E (eval)
```

Fazy B i D są niezależnie re-uruchamialne; C wymaga B gotowe; E wymaga A+B+C+D.

### Provider topology (ADR-009)

| Ścieżka | Provider | Model |
|---|---|---|
| enrich_sections.py (batch) | OpenRouter | meta-llama/llama-3.3-70b-instruct |
| LLMSelector, Critic (runtime) | Groq | per model_registry |
| QueryExpansion (runtime) | Groq | meta-llama/llama-4-scout-17b-16e-instruct (ADR-001) |
| Embeddings | Local BGE-M3 | BAAI/bge-m3 |

---

## Decyzje testowe

### Cechy dobrego testu

- Testuje **zewnętrzne zachowanie**: dla `DocFilter` test asercjuje które `doc_id` są zwracane dla danego query — NIE jak adapter wewnętrznie buduje frozenset.
- Moduły deterministyczne (`is_retrieval_unit`, `build_embeddings._embed_text`, `DocFilter`, `ContextPruner`) testowane bez mocka LLM — zero kosztu.
- Moduły z LLM (`enrich_sections`, `llm_selector`) testowane z deterministycznym stubbingiem `SectionDetails` / `confidence` response — instructor structured output łatwo replikować jako fixture.
- **Kluczowy test verbatim:** każdy portowany moduł (selector, pruner) musi mieć test sprawdzający że `result.markdown == original_markdown` nawet gdy `pruned_markdown` jest krótszy.

### Moduły z testami TDD (deterministyczne — zero $)

**`build_parquets` — `is_retrieval_unit`:**
- L1 z subsekcjami → `is_retrieval_unit=False`
- L1 bez subsekcji → `is_retrieval_unit=True`
- L2 → zawsze `is_retrieval_unit=True`
- Asercja: `df[df.is_retrieval_unit].shape[0] == 370 ± tolerancja`
- Asercja: sekcja o `section_code=="A"` (Kfz, ma subsekcje) → `is_retrieval_unit=False`

**`build_embeddings._embed_text`:**
- Sekcja z pełnymi Core-4 → embed-text zawiera heading, title, desc, 5 pytań, body[:400]
- Sekcja bez `title` (L1-rodzic, nie powinna trafić) → raise lub skip, nie crashuje
- body >= 400 znaków → embed-text zawiera dokładnie body[:400]

**`doc_filter.ProductDetectorAdapter`:**
- `QueryExpansion(sparte="Hausrat", tarif="Smart")` → zwraca doc_id Hausrat-Smart, nie Kfz
- `QueryExpansion(sparte="Kfz", tarif=None)` → zwraca oba Kfz doc_id
- Nieznany tarif → pusta frozenset (nie rzuca wyjątku)

**`doc_filter.RareTagMatcherAdapter`:**
- `domain_terms=["Glasbruch"]` → zwraca doc_id dokumentów z `topic_tags` zawierającym "Glasbruch"
- `domain_terms=["Schaden"]` (generyk z blocklista) → pusta frozenset
- `domain_terms=[]` → pusta frozenset

**`context_pruner` — verbatim guarantee:**
- `result.pruned_markdown != result.markdown` (pruner faktycznie skraca)
- `result.markdown == original_markdown` (verbatim nienaruszony)
- Chunk < 2500 znaków → `pruned_markdown == markdown` (bypass)
- Chunk po pruning == "" → fallback do pełnego `markdown`

### Moduły testowane przez eval (end-to-end, Faza E)

- `enrich_sections` — walidacja przez sanity check parquet po batchie: brak null w `title`/`description`/`questions` dla `is_retrieval_unit=True`
- `retriever` po przebudowie — pokryty przez 100-pytaniowy `eval_full`
- `llm_selector` abstain-rate — mierzony na OUT_OF_SCOPE pytaniach z eval setu
- `EMBED_THRESHOLD` kalibracja — grid search na eval wynikach Fazy E

### Dotychczasowe rozwiązania testów

`tests/test_build_parquets.py`, `tests/test_enrichment.py`, `tests/test_retriever.py` już istnieją. Nowe testy: `tests/test_doc_filter.py`, `tests/test_context_pruner.py`, `tests/test_build_embeddings.py`. Konwencja: pytest, fixtures w `tests/conftest.py`.

---

## Poza zakresem

- **Faza B (enrichment batch) nie startuje bez explicit go usera** — wymagane potwierdzenie przed realnym kosztem LLM.
- **`EMBED_THRESHOLD` nie jest ustawiony** do czasu kalibracji w Fazie E — kod musi obsługiwać `threshold=None` (brak abstain, zachowanie jak przed przebudową).
- **Zmiana modeli generatora lub critic** — nie jest częścią tego PRD; modele są w `model_registry.py` ale nie zmieniamy ich w tym rebuildu.
- **UI/front-end, auth, deployment** — poza zakresem (zgodnie z `PRD.md`).
- **Dodanie nowych Sparten lub dokumentów** — poza zakresem; parser i parquet obsługują to już przez `is_retrieval_unit`.
- **Pełna lista blocklista topic_tags** — istniejąca lista z `plan.md §5` wystarczy na Fazę B; rozszerzenie po Fazie E.

---

## Dodatkowe uwagi

- **Klucz OpenRouter został wklejony jawnie w czacie 2026-06-20** → zrotować w panelu OpenRouter przed uruchomieniem Fazy B.
- **`OPENROUTER_API_KEY`** musi być w `.env` (NIE commitować); template: `.env.example`.
- Enrichment smoke-test (2026-06-20): `enrich_section()` zwróciło DE title/description/5 pytań/tags dla sekcji Kfz-Spezial — API działa, klucz był ważny (teraz do rotacji).
- `topic_tags` ze smoke-testu zawierały lekkie generyki (`Kraftfahrzeuge`, `Anhänger`) → prompt wymaga zaostrzenia; blocklista po stronie `RareTagMatcherAdapter` (US-23 z `PRD.md`).
- Wzorzec DKV (`d:/_FUN/DKV_Belgium/calude/accuracy/src/`) — read-only referencja. Pliki do czytania per moduł: `enrich_sections.py`, `embedder.py` (`_section_text`), `doc_filter.py`, `reranker_strategy.py`, `llm_selector.py`, `context_pruner.py`, `embedding_pruner.py`, `model_registry.py`.
- Re-embed bez re-enrich jest możliwy dzięki separacji faz (ADR-007) — to jest główna oszczędność w przyszłych eksperymentach z compose fn.
