# Handoff — Enrichment + Parquet + Retriever rebuild

**Data:** 2026-06-20
**Następna sesja:** implementacja planu rozszerzenia enrichmentu sekcji, rozszerzenia parquet i pełnej przebudowy retrievera na wzór `d:/_FUN/DKV_Belgium/calude/accuracy/` (pattern reuse, adaptowany do naszego celu/danych).

> Tło, kontrakt, schematy, glosariusz DE → **[`PRD.md`](./PRD.md)** i **[`plan.md`](./plan.md)**. Ten dokument NIE powtarza ich — zawiera tylko decyzje z sesji grill (2026-06-20), których nigdzie indziej nie ma, oraz stan kodu.

---

## 1. Co ustalono w tej sesji (grill — decyzje finalne)

Sesja przeszła drzewo decyzji enrichment → parquet → retriever, korzeń→liście. Wszystkie zablokowane:

### Dane / enrichment
- **Język:** output pól enrichmentu w **DE**, system prompt w **EN** (LLM lepiej rozumie zadanie, ale emituje DE → zero dryfu terminów prawniczych). Reframe: pola enrichmentu są **retrieval-only (whitelist)** — nigdy nie trafiają do usera, więc „utrata subtelności przy tłumaczeniu" NIE dotyczy odpowiedzi. Cross-lingual = BGE-M3 (multilingual) + QueryExpansion→DE, **nie** tłumaczenie korpusu.
- **Pola = Core-4:** `title`, `description`, `questions` (generuj 5-10), `topic_tags` (rzadkie terminy DE, verbatim). Odrzucono `practical_applications`/`key_insights`/generic `tags` z wzorca — **brak konsumenta** w naszym pipeline (DKV miał selektor z label-injection; my nie).
- **Konsumenci pól:** `title+description+questions` → embedding retrievera; `topic_tags` → Rare-tag Matcher (exact match, dlatego musi być DE).
- **Engine:** **OpenRouter** (`meta-llama/llama-3.3-70b-instruct`) + **instructor** (pydantic + retry) + checkpoint/resume/skip-done. 70b > 8b dla idiomatycznych DE pytań.

### Granularność / parquet
- **Jednostka retrievalu = liść.** Pool ~370 = **152 liść-L1** (sekcje bez subsekcji) + **218 L2**. **53 rodziców-L1 WYKLUCZONE** z pooli/embeddingu — ich `markdown` to sklejony tekst dzieci (potwierdzone: L1 „A" len 2092 ≈ suma A.1+A.2+A.3 = 2062) → duplikacja, marnuje top_k, dwuznaczny cytat. Rodzice zostają w parquet tylko do breadcrumb.
- **Oznaczenie:** nowa kolumna bool **`is_retrieval_unit`** liczona deterministycznie w build_parquets (True = liść-L1 ∪ wszystkie L2). Pliki zostają **split** (sections.parquet / subsections.parquet).
- **Schema +kolumny:** `title`(str), `description`(str), `questions`(List[str]); zapełnić istniejące `topic_tags` (teraz `[]`); regen `embedding`; +`is_retrieval_unit`.
- **Skład embeddingu:** `heading + title + description + 5×questions + body[:400]`. Skrót body 2000→400 — enrichment przejmuje ładunek semantyczny, długi legalese rozmywa wektor BGE-M3 (mean-pooling). Generujemy 5-10 pytań, **embedujemy 5**, reszta w parquet.

### Build — 3 rozdzielone etapy (drogi LLM ≠ deterministyczny parse)
1. `build_parquets.py` — sanitize→parse→emit + `is_retrieval_unit`, **BEZ embeddingu** (0 $).
2. `enrich_sections.py` (NOWY) — load parquet → Core-4 przez OpenRouter, checkpoint+resume+skip-done → write back.
3. `build_embeddings.py` (NOWY/wydzielony) — skład embed-text → kolumna `embedding`.

   → re-embed bez re-enrich; enrich raz.

### Retriever — PEŁNY port wzorca, adaptowany do naszego celu
| Pattern DKV (`src/`) | Nasza adaptacja |
|---|---|
| `doc_filter.py` — Protocol + adaptery + `CompositeDocFilter` (union frozenset doc_id) | refaktor inline sparte/tarif filtra → `ProductDetectorAdapter` (Sparte+Tarif→doc_ids via documents.parquet) + `RareTagMatcherAdapter` (topic_tags→doc_ids) |
| `reranker_strategy.py` + `llm_selector.py` — `ContextSelector`: primary `TopKReranker`, jeśli `confidence < próg` → `BruteForceReranker` nad całym korpusem | nowy `llm_selector` na **Groq**; `confidence < próg → abstain` |
| `EMBED_THRESHOLD_L3 = 0.40` — podłoga score | **kalibrować na eval secie** — NIE kopiować 0.40 (BGE-M3 ≠ MiniLM, inne skale); niski top score → abstain |
| `context_pruner.py` + `embedding_pruner.py` — zdaniowy pruning, global bypass <2500 znaków, empty-guard | **okraja TYLKO kontekst LLM** (selektor/critic); finalny cytat = **pełny verbatim §** (whitelist nietknięta) |

**Provider topology:** runtime (selector, query_expansion, critic) = **Groq**; batch enrichment = **OpenRouter**. Trzymać provider/model **wymienny per-krok** (pattern `model_registry.py`/`llm_providers.py` z DKV) — przyszłe eksperymenty na korpusie mogą wskazać lepszy model per zadanie.

---

## 2. ⚠️ KRYTYCZNY konflikt do pilnowania: verbatim vs port

Nasz generator jest **verbatim** (cytuje § dosłownie z whitelisty `{markdown, heading, section_code}`). Wzorzec DKV **przepisuje** treść, więc mógł zdaniowo prunować. **U nas pruning cytowanego bloku łamie gwarancję dosłowności.** Rozwiązanie: dwa widoki chunku — `pruned-for-reasoning` (do LLM) vs `verbatim-for-citation` (do usera). **Każdy portowany moduł (selector, pruner) musi przepuścić verbatim nienaruszony — wymaga testu.**

---

## 3. Stan kodu

- ✅ **`src/enrichment.py`** — przepisany: OpenRouter + instructor, `SectionDetails` (Core-4), EN-prompt/DE-output, `openrouter_client()`, `enrich_section()`, smoke-test w `__main__`. **Smoke-test przeszedł** (call zwrócił DE title/description/5 pytań/tags). Uwaga: `topic_tags` wyciągnął lekkie generyki (`Kraftfahrzeuge`, `Anhänger`) → potrzebna **blocklista** generyków przy match (PRD US-23) i/lub doostrzenie promptu.
- ⬜ `build_parquets.py` — wymaga: dodać `is_retrieval_unit`, **usunąć** liczenie embeddingu w środku (przenieść do etapu 3).
- ⬜ `enrich_sections.py` — NOWY (batch resumable nad ~370 liśćmi).
- ⬜ `build_embeddings.py` — NOWY/wydzielony (nowy skład embed-text).
- ⬜ `retriever.py` — port: DocFilter, ContextSelector, próg→abstain, ContextPruner. Obecnie: `heading+markdown[:512]`, zawsze top_k, **brak progu**.
- ⬜ `src/llm_selector.py`, `src/doc_filter.py`, `src/context_pruner.py`, `src/embedding_pruner.py` — NOWE (port).
- Obecny stan parquet: `sections`(205, kol. m.in. `topic_tags=[]`, `embedding`), `subsections`(218, +`parent_section_id`), `documents`(8). Pool ładowany w `src/promptfoo_provider.py` (skleja sec+sub).

**Sekrety:** `OPENROUTER_API_KEY` + `GROQ_API_KEY` + `HF_TOKEN` w `.env` (NIE commitować; template w `.env.example`). **UWAGA bezpieczeństwo:** klucz OpenRouter został wklejony jawnie w czacie tej sesji → **zrotować** w panelu OpenRouter przy najbliższej okazji.

---

## 4. Kolejność wdrożenia (uzgodniona)

**A** build_parquets refactor (det., 0 $) → **B** `enrich_sections` batch ~370 ⟵ **GATE: realny $, wymaga explicit go usera** (PRD) → **C** build_embeddings → **D** retriever port + ragassistant wiring → **E** re-run `eval_full` + **kalibracja progu** podobieństwa.

Następna sesja zaczyna od **Fazy A** (czeka na potwierdzenie usera; user był pytany „zapisać plan + ruszyć Fazą A?" — odpowiedział wywołaniem handoff).

## 5. Referencje wzorca (pliki do czytania per moduł)

`d:/_FUN/DKV_Belgium/calude/accuracy/src/`: `enrich_sections.py`, `embedder.py` (`_section_text` — skład embeddingu!), `corpus_store.py`, `retriever.py`, `doc_filter.py`, `reranker_strategy.py`, `llm_selector.py`, `context_pruner.py`, `embedding_pruner.py`, `model_registry.py`. Wzorzec enrichmentu: `data_preparation.ipynb` cela 20-21 (`describe_section_full`, `SectionDetails`).

---

## Sugerowane umiejętności

- **`/tdd`** — `md_sanitizer`/`hierarchy_parser` mają być TDD-first (PRD); nowe moduły portu (DocFilter, selector, pruner) i `is_retrieval_unit` to deterministyczne kontrakty idealne na TDD.
- **`/dw-to-openrouter`** — jeśli pojawią się kolejne notebooki DataWorkshop do przełożenia na OpenRouter (już używamy OpenRouter do enrichmentu).
- **`/code-review`** lub **`/myreview`** — po Fazie D (port retrievera) zrewidować zachowanie verbatim-guarantee przez selector+pruner.
- **`/diagnose`** — przy kalibracji progu w Fazie E, jeśli eval hit-rate spadnie po zmianie składu embeddingu.
- **`/grill-me`** — kontynuacja jeśli wyniknie nowa gałąź decyzji (np. kształt `llm_selector` confidence, format blocklisty topic_tags).
