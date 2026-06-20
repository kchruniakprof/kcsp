# PRD — ERGO P&C Agent-Bot v1

> Źródło decyzji: [`plan.md`](./plan.md) (sesja discovery, decyzje finalne). Ten dokument to implementacyjny kontrakt dla developera.
> Wzorzec architektoniczny: `d:/_FUN/DKV_Belgium/calude/accuracy/` (reuse pattern, nie kopia treści).

---

## Opis problemu

Agenci ERGO i salespartnerzy (B2B) muszą podczas rozmów z klientami szybko wyszukiwać informacje z Bedingungen (warunki ubezpieczenia P&C). Teraz: ręczne przeglądanie PDF — minuty. Cel: kilka sekund.

Domena jest regulowana (prawo ubezpieczeniowe DE). **Halucynacja = niedopuszczalna** — błędna odpowiedź o zakresie ochrony jest gorsza niż brak odpowiedzi.

Dodatkowe wyzwania:
- 4 Sparten (Kfz, Hausrat, Glas, Schmuck) z różnymi strukturami i schematami numeracji.
- Hausrat ma 4 prawie identyczne (~90%) taryfy → retrieval musi filtrować po taryfie.
- Ligatury OCR (docling artefakty: `Haftpfl icht`, `fi nden`) rozbijają nagłówki → parser fail bez sanityzacji.
- Queries w DE + PL + EN, dokumenty tylko w DE.

## Rozwiązanie

System RAG oparty na zasadzie *pure classification + semantic safety net*:

- **Verbatim markdown** — generator NIE przepisuje treści; cytuje fragmenty Bedingungen dosłownie.
- **Whitelista pól** — do odpowiedzi użytkownika trafiają tylko `{markdown, heading, section_code}`; pola enrichmentu (`insights`, `description`) są niedostępne dla generatora.
- **Pipeline z guardrailami** — każda warstwa może zakończyć przepływ: `OUT_OF_SCOPE` → abstain bez retrievalu, `critic.verdict == ABSTAIN` → finalne abstain.
- **Tarif filter** — retrieval pre-filtruje po Sparte + Tarif (z Product Detectora), aby nie mieszać Smart z Best.
- **Breadcrumb citation** — każda odpowiedź zawiera "Kfz > Spezial > §A Beginn des Versicherungsschutzes" jako citation.
- **Cross-lingual** — QueryExpansion normalizuje query (PL/EN) do DE terminów domenowych; BGE-M3 (local, multilingual) jako embedding.
- **Groq stack** — fast inference (jak chatbot DKV); finalne modele per-krok po benchmarku (Task #1).

---

## Historie użytkownika

### Wyszukiwanie (agent / salespartner)

1. Jako agent ERGO, chcę zadać pytanie w DE/PL/EN o konkretny zapis w Bedingungen (np. "Co obejmuje Hausrat Best w przypadku szkód wodociągowych?"), aby otrzymać dosłowny fragment w kilka sekund.
2. Jako salespartner, chcę sprawdzić czy konkretne zdarzenie lub przedmiot jest objęty polisą (COVERAGE_CHECK), aby odpowiedzieć klientowi bez konsultacji z centralą.
3. Jako agent, chcę zobaczyć dokładne brzmienie paragrafu DE z breadcrumb (Sparte > Tarif > §X Nagłówek), aby cytować klientowi precyzyjne warunki umowy.
4. Jako agent, chcę zadać pytanie po polsku lub angielsku i otrzymać odpowiedź z dokumentów DE, aby korzystać z bota niezależnie od języka roboczego.
5. Jako salespartner, chcę porównać warunki dwóch taryf Hausrat (np. Smart vs Best) i zobaczyć konkretne różnice (COMPARE), aby rekomendować właściwy produkt.
6. Jako agent, chcę zapytać o procedury i obowiązki ubezpieczonego (PROCESS_OBLIGATION), aby przekazać klientowi prawidłowe kroki zgłoszenia szkody.
7. Jako agent, chcę sprawdzić kto lub co może być ubezpieczone w danym produkcie (ELIGIBILITY), aby uniknąć błędów przy zawieraniu umowy.
8. Jako salespartner, chcę uzyskać argument sprzedażowy (SALES_USP) dla danego produktu, aby użyć go w rozmowie z klientem.
9. Jako agent, chcę zapytać o przykład z życia codziennego ilustrujący zakres polisy (EXAMPLE), aby klient lepiej rozumiał ochronę.
10. Jako agent, chcę zapytać o taryfy i zniżki SF-Klasse (PRICING), aby wstępnie wycenić polisę podczas rozmowy.
11. Jako agent, chcę dostać jednoznaczne "Dazu habe ich keine Information" zamiast wymyślonej odpowiedzi, gdy temat jest poza zakresem dokumentów.
12. Jako salespartner, chcę zobaczyć sugestię cross-sell (np. Glas→"można dołączyć Hausrat"), gdy pytam o powiązany produkt.

### COMPARE (porównanie taryf)

13. Jako agent, chcę aby bot rozbił pytanie porównawcze (COMPARE) na sub-pytania per tarif i zsyntezował różnice w jednej odpowiedzi.
14. Jako agent, chcę COMPARE tylko dla taryf tej samej Sparte (np. Hausrat Smart vs Best), nie Kfz vs Hausrat, aby porównanie było semantycznie sensowne.
15. Jako operator, chcę aby COMPARE używał mocniejszego modelu (diff-step generator na Groq), aby różnice były precyzyjnie wyodrębnione z prawie identycznych sekcji.
16. Jako agent, chcę wynik COMPARE z jasnym wyróżnieniem różnic (co jest w jednym tarifu a nie w drugim), aby szybko odczytać kluczowe punkty.

### Pipeline RAG (operator)

17. Jako operator, chcę aby każde query przechodziło przez QueryExpansion (Groq) z normalizacją terminów do DE i klasyfikacją intent (COMPARE | FACT_LOOKUP | COVERAGE_CHECK | PROCESS_OBLIGATION | ELIGIBILITY | PRICING | SALES_USP | EXAMPLE | OUT_OF_SCOPE).
18. Jako operator, chcę aby `intent == OUT_OF_SCOPE` skutkował natychmiastowym abstain bez wywoływania retrievalu (oszczędność kosztu i latencji).
19. Jako operator, chcę aby SubQuestion Decomposer (max 3 sub-pytania) odpalał się TYLKO gdy `intent == COMPARE`, aby uniknąć zbędnych LLM calls dla innych intencji.
20. Jako operator, chcę aby Product Detector działał 4-warstwowo: Aho-Corasick (exact) → RapidFuzz (typo-tolerant) → BGE-M3 Embeddings (semantic) → LLM fallback (Groq), z progiem confidence per warstwie.
20a. Jako agent, chcę dostać automatyczną sugestię cross-sell (np. "Warto rozważyć Glasversicherung do polisy Hausrat") gdy kontekst obejmuje Glas lub Schmuck z `related_sparte`.
21. Jako operator, chcę aby retrieval pre-filtrowało po Sparte i Tarif (wykrytych przez Product Detector), aby wykluczyć dokumenty niewłaściwego produktu i nie mieszać taryf.
22. Jako operator, chcę aby intent→section_type routing zawężał retrieval według mapy (COVERAGE_CHECK→WHAT_IS_INSURED/EXCLUSIONS itd.), aby zwiększyć precyzję.
23. Jako operator, chcę aby Rare-tag Matcher używał topic_tags (SF-Klasse, Glasbruch, Tierbiss, Fahrerschutz, GAP itd.) do łapania query bez nazwy produktu, z blocklistą generycznych terminów.
24. Jako operator, chcę aby Generator emitował wyłącznie verbatim markdown z whitelisty pól (`{markdown, heading, section_code}`), bez żadnego LLM rewrite treści.
25. Jako operator, chcę aby Critic (gpt-oss-120b Groq, prowizoryczny) gate'ował odpowiedź verdiktem PASS / REGEN(×1) / ABSTAIN, aby blokować halucynacje przed userem.
26. Jako operator, chcę aby każda odpowiedź zawierała breadcrumb (Sparte > Tarif > §X Nagłówek) jako citation, aby agent mógł wskazać klientowi konkretny paragraf.
27. Jako operator, chcę aby wszystkie LLM-calle używały pydantic + instructor, temperature=0, aby zachować determinizm i structured output.

### Warstwa danych (developer)

28. Jako developer, chcę `md_sanitizer` scalający ligatury OCR (`fi`→fi, `fl`→fl) deterministycznie (regex, bez LLM), działający PRZED parserem, aby nagłówki parsowały się poprawnie.
29. Jako developer, chcę generyczny `hierarchy_parser` + deklaratywną per-Sparte mapę (Kfz: litery A–N schema GDV AKB; Hausrat: numery 1–30 KT; Glas: 1–16; Schmuck: własne), aby jeden parser obsłużył 4 Sparten bez duplikacji kodu.
30. Jako developer, chcę aby parser generował `breadcrumb` dla każdej sekcji i `section_code` (litera lub numer), aby retrieval mógł cytować precyzyjne lokalizacje w dokumentach.
31. Jako developer, chcę aby `sections.parquet` miał pola: `doc_id`, `section_id`, `sparte`, `tarif`, `section_code`, `section_types` (List[enum-16], multi-label), `topic_tags` (List[str]), `heading`, `markdown`, `breadcrumb`, `confidence_score=1.0`.
32. Jako developer, chcę aby `documents.parquet` miał pola: `doc_id`, `sparte`, `tarif`, `numbering_scheme`, `related_sparte`, `source_file`.
33. Jako developer, chcę aby `enrichment` (Groq LLM + instructor) generował `title`, `description`, `topic_tags` dla każdej sekcji, ze structured output validation.
34. Jako developer, chcę aby `build_parquets.py` walidował: 8 dokumentów na wyjściu, każdy ma niepuste sekcje, każdy `section_type` z enum-16 fires ≥ raz w korpusie, multi-label działa poprawnie (jedna sekcja = wiele typów).
35. Jako developer, chcę aby Glas i Schmuck miały `related_sparte="Hausrat"` w `documents.parquet`, aby umożliwić cross-sell linkage w retrieval i odpowiedziach.

### Eval i jakość

36. Jako developer, chcę eval set 100 pytań w natywnym formacie promptfoo z rozkładem zatwierdzonym w `plan.md §8` (COMPARE 15, FACT_LOOKUP 30, COVERAGE_CHECK 20, PROCESS_OBLIGATION 10, Kfz 30, Hausrat 40, Glas 15, Schmuck 10, stress 12/100).
37. Jako developer, chcę aby Claude wygenerował szablon eval YAML z propozycjami pytań i pustymi polami `expected_*`, a ekspert ERGO wypełnił je ręcznie po wygenerowaniu parquetów.
38. Jako developer, chcę two-layer metric: (1) retrieval-hit js-assert na `expected_doc_ids`/`expected_section_types`; (2) answer-faithfulness llm-rubric "odpowiedź cytuje tylko fakty z podanego kontekstu DE".

### Bezpieczeństwo i operacje

39. Jako operator, chcę aby `GROQ_API_KEY` był wymagany w `.env` (NIE commitowany; template w `.env.example`).
40. Jako operator, chcę aby stress-test prompt injection (2 z 100 pytań eval) zawierał testy odporności bota na próby modyfikacji systemu — treść pisze osoba z security context.

---

## Decyzje wdrożeniowe

### Hierarchia produktów (4 Sparten)

```
Sparte (Kfz | Hausrat | Glas | Schmuck)
  └── Tarif (Spezial | Standard | Smart | Best | Best+Naturgefahren | Best+Fahrraddiebstahl | KT2021GLHR | KT Schmuck)
        └── Baustein (sekcja/§ w dokumencie)
```

Glas i Schmuck = prawnie samodzielne kontrakty GDV (własne §Beitrag, §Kündigung, §Beschwerde), nie Bausteine Hausrat. Cross-sell przez `related_sparte="Hausrat"` w metadata.

### Moduły (11)

**Build-time (deterministyczne):**
- `md_sanitizer` — ligature fix (`fi`→fi, `fl`→fl), strip artefaktów OCR, normalizacja whitespace. Wejście: surowy MD. Wyjście: czysty MD. **MUSI działać przed parserem.**
- `hierarchy_parser` — generyczny parser hierarchii + per-Sparte config (mapa numeracja→section_code, regex nagłówka). Wejście: 8 plików MD. Wyjście: lista `Document` + `Section` pydantic z breadcrumb.
- `build_parquets` — orchestrator: sanitizer → parser → enrichment → emit `documents.parquet` + `sections.parquet`. Walidacja schematu na końcu.

**LLM enrichment (one-time batch):**
- `enrichment` — Groq (llama-3.3-70b-versatile) + instructor; generuje `title`, `description`, `topic_tags` dla każdej sekcji. Structured output validation przez pydantic.

**Runtime pipeline (deep modules):**
- `query_expansion` — Groq; output: `paraphrases`, `domain_terms` (w DE), `intent` (enum), `sparte_hints`. Wejście: raw query (DE/PL/EN). Wyjście: `QueryExpansion` pydantic.
- `decomposer` — Groq; max 3 sub-pytania; trigger TYLKO `intent==COMPARE`.
- `product_detector` — 4-layer: Aho-Corasick → RapidFuzz → BGE-M3 → LLM fallback. Kontrakt: `query → detected_sparte, detected_tarif, confidence`.
- `retriever` — BGE-M3 vector retrieval + tarif/section_type pre-filter + Rare-tag Matcher. **Najgłębszy moduł.** Kontrakt: `QueryExpansion → RetrievedContext` (lista chunks z doc_id, section_id, breadcrumb).
- `generator` — split: (a) verbatim cheap (llama-3.3-70b, dla nie-COMPARE) + (b) diff-step strong (gpt-oss-120b, dla COMPARE). Whitelist: `{markdown, heading, section_code}`.
- `critic` — gpt-oss-120b Groq (PROVISIONAL); verdict `PASS | REGEN | ABSTAIN`; max 1 retry przy REGEN.
- `ragassistant` — top-level orchestrator; składa pipeline w sekwencję; emituje `FinalAnswer` z breadcrumb + audit trail.

**Eval:**
- `promptfoo_provider` — Python wrapper `RAGAssistant.run(query) → dict` (`retrieved_doc_ids`, `retrieved_section_types`, `abstained`, `answer_markdown`).
- `ergo_eval.yaml` — 100 pytań (szablon generuje Claude, ekspert ERGO wypełnia `expected_*`).

### Schema `sections.parquet`

| Pole | Typ | Uwagi |
|---|---|---|
| `doc_id` | str | FK → documents, nazwa pliku bez .md |
| `section_id` | int | Global unikalny |
| `sparte` | str | Denorm z documents |
| `tarif` | str | Denorm, używany jako filter w retrieval |
| `section_code` | str | "A" (Kfz) lub "1" (Hausrat/Glas) |
| `section_types` | List[str] | Multi-label, enum-16 |
| `topic_tags` | List[str] | Rzadkie słowa kluczowe domeny |
| `heading` | str | Nagłówek verbatim |
| `markdown` | str | Treść verbatim, sanityzowana |
| `breadcrumb` | str | "Kfz > Spezial > §A Nagłówek" |
| `confidence_score` | float | 1.0 (parser deterministyczny) |

**Whitelist generatora:** `{markdown, heading, section_code}` — reszta NIGDY do user-facing response.

### Enum-16 `section_type` (multi-label)

```
INSURER_ID, PRODUCT_STRUCTURE, RISK_OBJECT, WHAT_IS_INSURED,
EXCLUSIONS, LIMITS_COMPENSATION, CLAIMS_SETTLEMENT, INSURED_PERSONS,
WHERE_COVERED, OBLIGATIONS, PAYMENT, CONTRACT_FORMATION,
TERM_CANCELLATION, PRICING_DISCOUNT, COMPLAINTS_LAW, SPECIAL_PROVISIONS
```

Subsection może mieć listę typów (multi-label) — np. §A Kfz może być `[CONTRACT_FORMATION, TERM_CANCELLATION]`.

### Pipeline (sekwencja końcowa)

```
USER QUERY (DE | PL | EN)
  → [1] QueryExpansion + IntentClassifier (Groq llama-3.3-70b)
         intent == OUT_OF_SCOPE → ABSTAIN (bez retrievalu)
         intent == COMPARE → [1b] Decomposer (max 3 sub-pytania)
  → [2] Product Detector 4-layer (Aho-Corasick→RapidFuzz→BGE-M3→LLM)
  → [3] Sparte + Tarif filter
  → [4] Rare-tag Matcher (topic_tags)
  → [5] Section type pre-filter (intent → section_types)
  → [6] Vector retrieval (BGE-M3 local)
  → [7a] COMPARE: diff-step generator (gpt-oss-120b Groq)
      [7b] pozostałe: verbatim generator (llama-3.3-70b Groq)
  → [8] Critic (gpt-oss-120b Groq, PROVISIONAL) → PASS / REGEN(×1) / ABSTAIN
  → FINAL ANSWER + breadcrumb + sparte/tarif + audit metadata
```

### Modele (Groq stack)

| Krok | Model | Provider |
|---|---|---|
| Query expansion + intent | `llama-3.3-70b-versatile` | Groq |
| Decomposer | `llama-3.3-70b-versatile` | Groq |
| Product detector LLM fallback | `llama-3.3-70b-versatile` | Groq |
| Enrichment (batch) | `llama-3.3-70b-versatile` | Groq |
| Generator verbatim (nie-COMPARE) | `llama-3.3-70b-versatile` | Groq |
| Generator diff-step (COMPARE) | `gpt-oss-120b` | Groq |
| Critic (PROVISIONAL) | `gpt-oss-120b` | Groq |
| Embeddings | `BAAI/bge-m3` | Local |
| Promptfoo eval judge | `llama-4-maverick` | Groq |

**Task #1** = benchmark i finalny wybór modeli. Do uruchomienia po zbudowaniu eval setu.

### Per-Sparte konfiguracja parsera (deklaratywna)

```python
# Prototyp — koduje decyzję hierarchii (nie implementacja finalna)
SPARTE_CONFIG = {
    "Kfz": {
        "numbering": "letters",   # A, B, C ... N
        "standard": "GDV_AKB",
        "section_regex": r"^##\s+([A-N])\s+",
    },
    "Hausrat": {
        "numbering": "numbers",   # 1, 2 ... 30
        "standard": "KT",
        "section_regex": r"^##\s+§?\s*(\d+)\s+",
    },
    "Glas": {
        "numbering": "numbers",   # 1 ... 16
        "standard": "KT2021GLHR",
        "section_regex": r"^##\s+§?\s*(\d+)\s+",
    },
    "Schmuck": {
        "numbering": "numbers",
        "standard": "KT",
        "section_regex": r"^##\s+§?\s*(\d+)\s+",
    },
}
```

### Intent → section_type routing

| Intent | Section_types pre-filter |
|---|---|
| `FACT_LOOKUP` | (bez pre-filtru) |
| `COVERAGE_CHECK` | `WHAT_IS_INSURED`, `EXCLUSIONS` |
| `PROCESS_OBLIGATION` | `OBLIGATIONS`, `CLAIMS_SETTLEMENT` |
| `ELIGIBILITY` | `INSURED_PERSONS`, `RISK_OBJECT` |
| `PRICING` | `PRICING_DISCOUNT`, `PAYMENT` |
| `SALES_USP` | `WHAT_IS_INSURED`, `SPECIAL_PROVISIONS` |
| `EXAMPLE` | `WHAT_IS_INSURED`, `LIMITS_COMPENSATION` |
| `COMPARE` | (wszystkie sekcje obu taryf — diff-step) |

---

## Decyzje testowe

### Cechy dobrego testu

- **Testuje zewnętrzne zachowanie**, nie wewnętrzną implementację — dla `retriever` test asercjuje *jakie chunki* są zwrócone dla danego query, NIE jak warstwy się komunikują.
- **Mock LLM, nie cały świat** — `query_expansion`, `decomposer`, `critic` testowane z deterministycznym stubbingiem odpowiedzi LLM (instructor structured output łatwo replikować jako fixture).
- **Deterministyczne fixtures** — `md_sanitizer` i `hierarchy_parser` testowane na realnych 8 plikach MD, asercje na konkretne liczby.

### Moduły z testami TDD (deterministyczne — zero kosztu LLM)

**`md_sanitizer`** — kanoniczny TDD, pierwszy moduł:
- Ligatury: `Haftpfl icht` → `Haftpflicht`, `fi nden` → `finden`, `vorläufi g` → `vorläufig`, `Aufl ieger` → `Auflieger`
- Artefakty: brak `<!-- image -->` po sanityzacji
- Whitespace: wielokrotne puste linie → max 1
- Zachowuje: nagłówki `#`, bullets, tabele, numery sekcji, treść

**`hierarchy_parser`** — TDD, drugi moduł:
- 8 dokumentów na wejściu → 8 `Document` na wyjściu
- Każdy dokument ma niepuste sekcje
- Kfz: `section_code` ∈ {"A","B","C",...,"N"}
- Hausrat: `section_code` ∈ {"1","2",...,"30"}
- Glas: `section_code` ∈ {"1",...,"16"}
- Każdy `section_type` z enum-16 fires ≥ raz w korpusie
- Multi-label: wskazana sekcja Hausrat §1 ma `section_types` = lista z ≥2 elementami
- Breadcrumb format: "Sparte > Tarif > §X Nagłówek"
- Ligatury w nagłówkach: sekcja "Haftpfl icht" w surowym MD parser finduje poprawnie po sanityzacji

### Moduły testowane przez eval YAML (end-to-end)

- `retriever`, `generator`, `critic`, `ragassistant`, `product_detector` — pokryte przez 100-pytaniowy eval set.
- `promptfoo_provider` — sprawdzany przez sam fakt że eval działa.
- `enrichment`, `build_parquets` — weryfikowane przez sanity `read_parquet` + asercje wartości po build (counts, joinability, no nulls w required fields).

### Dotychczasowe rozwiązania

W katalogu ERGO brak istniejących testów. Pierwszy plik testowy: `tests/test_md_sanitizer.py`, drugi: `tests/test_hierarchy_parser.py`. Konwencja (z DKV Belgium): pytest, fixtures w `tests/conftest.py`, fixtures danych = ścieżki do 8 plików MD.

---

## Poza zakresem

- **UI / front-end** — bot jest backend-only. Interfejs (web/Slack/Teams) poza tym PRD.
- **Klienci końcowi** — bot jest B2B (agenci + salespartnerzy). NIE dla klientów ERGO.
- **Inne linie biznesowe ERGO** — v1 = 4 Sparten P&C (Kfz, Hausrat, Glas, Schmuck), Bedingungen only.
- **FAQ / inne źródła** — tylko Bedingungen (8 plików MD). NIE scrape stron ERGO.
- **Streaming** — odpowiedzi jako kompletny blok JSON; brak SSE/token streaming.
- **Multi-turn conversation** — bot stateless, każde query niezależne.
- **Auth / rate limit per user** — brak warstwy API gateway.
- **Produkcyjny deployment** (Docker, K8s, EU hosting) — projekt kończy się na lokalnym uruchomieniu evalu.
- **Inne języki dokumentów** — dokumenty zostają w DE; cross-lingual = query expansion, nie tłumaczenie dokumentów.

---

## Dodatkowe uwagi

- **Pierwszy kod: `md_sanitizer` + `hierarchy_parser`** (deterministyczne, zero kosztu LLM). TDD first. Przed Task #2+ (LLM calls = real $): explicit zgoda usera + `GROQ_API_KEY` w `.env`.
- **Ligatura sanitizer MUSI działać PRZED parserem** — inaczej nagłówki jak "§A Haftpfl icht" nie matchują regex i sekcje są tracone.
- **Eval set `expected_*` MUSI być wypełniony przez eksperta ERGO** — Claude generuje propozycje + puste pola; bez wypełnienia benchmark bezużyteczny.
- **`GROQ_API_KEY`** → `.env`, NIGDY nie commitować, NIGDY nie wpisywać w kod ani docs. Template: `.env.example`.
- **Task #1 (benchmark modeli)** = uruchomić po zbudowaniu eval setu. Kandydaci: cheap tier (llama-3.3-70b vs gpt-oss-20b), diff generator (gpt-oss-120b vs llama-4-maverick), eval judge (llama-4-maverick vs gpt-oss-120b). Metryki: accuracy + latency Groq + koszt + jakość DE prawnicza.
- **BGE-M3** — model lokalny (~600MB). Wymaga ≥8GB RAM lub GPU. Startup latency ~30s przy pierwszym załadowaniu.
- **Near-duplikaty Hausrat** — Smart/Best/Best+Naturgefahren/Best+Fahrraddiebstahl dzielą ~90% sekcji. Bez tarif filter retrieval zwróci mieszankę taryf → błędna odpowiedź. Tarif filter = kluczowy guardrail.
- **Wzorzec repo** → skopiować z `d:/_FUN/DKV_Belgium/calude/accuracy/` (gitignore, .env.example, src/, tests/, eval/).

---

## Glosariusz DE

| Termin | Znaczenie |
|---|---|
| Bedingungen | Warunki ubezpieczenia (ogólne/szczególne) |
| Sparte | Linia produktowa (Kfz, Hausrat, Glas, Schmuck) |
| Tarif | Wariant taryfy w ramach Sparte |
| Baustein | Moduł/sekcja w dokumencie Bedingungen |
| AKB | Allgemeine Kraftfahrt-Bedingungen (GDV standard, Kfz) |
| KT | Klauseltarif (standard dla Hausrat/Glas/Schmuck) |
| GDV | Gesamtverband der Deutschen Versicherungswirtschaft (izba ubezp.) |
| SF-Klasse | Schadenfreiheitsklasse (klasa bezszkodowości, Kfz) |
| Teilkasko | Ubezpieczenie częściowe (Kfz) |
| Vollkasko | Ubezpieczenie pełne (Kfz) |
| Wertsachen | Przedmioty wartościowe (Schmuck, Hausrat) |
| Hausrat | Mienie domowe |
| Glasversicherung | Ubezpieczenie szyb |
| Schmucksachen | Biżuteria, kosztowności |
| Abstain | Świadoma odmowa odpowiedzi (brak halucynacji) |
| Breadcrumb | Ścieżka hierarchiczna: "Sparte > Tarif > §X Nagłówek" |
| verbatim markdown | Cytowanie bez przepisywania przez LLM |

---

*PRD wygenerowany ze [`plan.md`](./plan.md) sesji discovery 2026-06-19. Decyzje architektoniczne nie podlegają renegocjacji bez explicit zgody usera. Nowa niejasność → zapytaj zanim improwizujesz.*
