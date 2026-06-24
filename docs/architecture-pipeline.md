# Architektura pipeline'u — ERGO P&C Agent-Bot

Dokładny opis działania pipeline'u RAG, podzielony na **build offline** (budowa indeksu)
i **runtime online** (obsługa pytania). Przy każdym elemencie zaznaczono, czy używa LLM
i jaki model. Stan na: 2026-06-21.

## Diagram

```
══════════════════════ A. BUILD OFFLINE (indeksowanie, jednorazowo) ══════════════════════

  8× .md (OCR)
      │
      ▼
 ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────────────┐
 │ md_sanitizer    │──▶│ hierarchy_parser │──▶│ build_parquets (orchestrator)│
 │ (reguły)        │   │ (reguły/regex)   │   │ strip_noise + is_retrieval…  │
 └─────────────────┘   └──────────────────┘   └──────────────────────────────┘
                                                     │ documents/sections/subsections.parquet
                                                     ▼
                                        ┌────────────────────────────┐
                                        │ enrichment   🤖 LLM         │
                                        │ OpenRouter gpt-4o-mini      │
                                        │ → title, description,       │
                                        │   5-10 questions, topic_tags│
                                        └────────────────────────────┘
                                                     │
                                                     ▼
                                        ┌────────────────────────────┐
                                        │ build_embeddings (BGE-M3)   │
                                        │ lokalny model, 1024-dim     │
                                        └────────────────────────────┘
                                                     │  parquet + embeddings
                                                     ▼  (indeks gotowy)

══════════════════════ B. RUNTIME ONLINE (na każde pytanie) ══════════════════════════════

  pytanie (de/pl/en)
      │
      ▼
 ┌──────────────────────────┐
 │ QueryExpansion   🤖 LLM   │  Groq llama-4-scout
 │ intent, język→DE, sparte, │
 │ section_types, paraphrasy,│
 │ domain_terms, confidence  │
 └──────────────────────────┘
      │           │ OUT_OF_SCOPE ─────────────▶ ABSTAIN
      ▼
 ┌──────────────────────────┐    ┌────────────────────────────────────┐
 │ _detect_tarif (regex)     │───▶│ DocFilter (reguły)                 │
 └──────────────────────────┘    │  ProductDetector ∩ RareTagMatcher  │
                                  └────────────────────────────────────┘
                                              │ allowed doc_ids
                                              ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ Retriever (BGE-M3 bi-encoder, BEZ LLM)                             │
 │ 1.doc_filter → 2.section_type(guard≥3) → 3.multi-query dot-product │
 │ → 4.top_k/pool_k → 5.[opc.] CrossEncoder reranker (lokalny)        │
 └───────────────────────────────────────────────────────────────────┘
      │  brak wyników ─────────────────────▶ ABSTAIN
      ▼  (+ ContextPruner: reguły, dual-view)
 ┌──────────────────────────┐
 │ Generator        🤖 LLM   │  VERBATIM: Groq llama-3.1-8b-instant
 │ VERBATIM lub COMPARE      │  COMPARE: Groq llama-3.3-70b-versatile
 └──────────────────────────┘
      │
      ▼  (Critic pomijany dla COMPARE)
 ┌──────────────────────────┐
 │ Critic           🤖 LLM   │  Groq qwen3-32b (+ opc. ensemble 70b)
 │ PASS / REGEN / ABSTAIN    │  run_critic: pętla REGEN + graceful PASS
 └──────────────────────────┘
      │ PASS                      │ ABSTAIN
      ▼                           ▼
 +cross-sell (reguły)         ABSTAIN
      ▼
  FinalAnswer
```

## A. Build offline (budowa indeksu)

**`md_sanitizer`** — *reguły, bez LLM.* Naprawia tekst po OCR, w tym rozbite ligatury
(`Haftpfl icht`→`Haftpflicht`) w nagłówkach. Musi działać przed parserem, bo regexy sekcji
łapią się na nagłówkach.

**`hierarchy_parser`** — *reguły/regex, bez LLM.* Ma deklaratywny katalog 8 dokumentów
(plik→sparte/tarif/schemat numeracji) i per-Sparte konfigurację regexów. Tnie dokument na
sekcje L1 i podsekcje L2, nadaje `section_code`, `breadcrumb`, `section_types`,
`parent_section_id`. **To tutaj rodzi się problem z Naturgefahren** — §4 nie jest cięte
na 4.1…4.9.

**`build_parquets`** — *reguły, bez LLM.* Orkiestruje sanitizer→parser→zapis 3 plików parquet
(documents/sections/subsections). `strip_noise` usuwa spis treści, stopkę marketingową i
**tabele >5 linii**; `is_index_section` wyrzuca indeksy alfabetyczne. Liczy
`is_retrieval_unit` (L2 zawsze; L1 tylko gdy nie ma dzieci) — to decyduje, co trafia do indeksu.

**`enrichment`** — 🤖 **LLM: OpenRouter `gpt-4o-mini`.** Dla każdej jednostki retrievalu
generuje „Core-4": tytuł, opis, 5–10 realnych pytań klienta/agenta oraz `topic_tags`
(rzadkie terminy domenowe, dosłownie po niemiecku). Pytania i opis wzmacniają embedding,
a `topic_tags` zasilają RareTagMatcher.

**`build_embeddings`** — *lokalny model BGE-M3, bez LLM-API.* Skleja tekst do embedowania
wg ADR-006: `heading + title + description + Q1..Q5 + markdown[:400]`, koduje do wektora
1024-dim znormalizowanego. Tylko `is_retrieval_unit=True` dostają wektor.

## B. Runtime online (na każde pytanie)

**`RAGAssistant.ask`** — *orkiestrator, bez własnego LLM.* Spina kroki: QueryExpansion →
(filtr) → Retriever → Generator → Critic → FinalAnswer. Ma 3 ścieżki ABSTAIN: intent
OUT_OF_SCOPE, pusty retrieval, werdykt Critic=ABSTAIN.

**`QueryExpansion`** — 🤖 **LLM: Groq `llama-4-scout-17b`** (instructor + Pydantic, few-shot).
Klasyfikuje intent, wykrywa język i **normalizuje pytanie zawsze do niemieckiego**, zwraca
`sparte_hints`, `section_types`, 3–5 parafraz, `domain_terms` i confidence. To jedyny krok
rozumienia pytania — od jego trafności zależy cały filtr i retrieval.

**`_detect_tarif`** — *deterministyczny regex, bez LLM.* Dopasowuje nazwę taryfy po granicy
słowa (najdłuższe najpierw, aliasy z `split('+')`). **To tu poległo pytanie o Überschwemmung**
— brak słowa-taryfy → `None`.

**`DocFilter`** — *reguły, bez LLM.* `CompositeDocFilter` = bramka ∩ zawężenie:
`ProductDetectorAdapter` mapuje sparte+tarif→doc_ids, `RareTagMatcherAdapter` mapuje
`domain_terms`→`topic_tags`→doc_ids. Gdy bramka zwróci `None` (brak sparte) → brak filtra,
przeszukiwane jest wszystko (stąd zalew duplikatami).

**`Retriever`** — *bi-encoder BGE-M3, bez LLM.* Pięć kroków: (1) doc_filter zawęża kandydatów,
(2) filtr `section_type` z bezpiecznikiem (jeśli <3 trafień — pomijany), (3) iloczyn skalarny
macierzy `kandydaci × [zapytanie+parafrazy]`, branie maksimum, (4) top_k lub szerszy pool_k,
(5) opcjonalny **CrossEncoder reranker** (`BAAI/bge-reranker-v2-m3`, lokalny, ładowany leniwie).

**`ContextPruner`** — *reguły, bez LLM.* Dual-view: `verbatim_text` (oryginał dla
generatora/użytkownika) i `pruned_text` (skrót zdaniowy dla wewnętrznych potrzeb). Chunki
<2500 znaków przepuszczane bez zmian.

**`Generator`** — 🤖 **LLM: Groq.** Tryb VERBATIM (`llama-3.1-8b-instant`) — odtwarza treść
dosłownie z sekcji + breadcrumb, ma reguły interpretacji (synonimy „Fahrzeugwechsel"=
„Veräußerung", priorytet jawnych list wykluczeń). Tryb COMPARE (`llama-3.3-70b-versatile`) —
buduje tabelę porównawczą taryf. Jeśli brak treści → pusty string.

**`Critic` / `run_critic`** — 🤖 **LLM: Groq `qwen3-32b`** (+ opcjonalny ensemble
`llama-3.3-70b`). Sprawdza, czy każda teza odpowiedzi ma pokrycie w źródłach: PASS / REGEN
(poprawialny błąd → regeneracja raz) / ABSTAIN (zmyślone kwoty/daty/warunki). Prompt jest
„anti-over-abstain" (hedging i niepełność = PASS); błąd techniczny Critic = graceful PASS;
**pomijany dla COMPARE**.

**Cross-sell + FinalAnswer** — *reguły, bez LLM.* Jeśli włączone, dokleja podpowiedź
uzupełniających produktów wg mapy (Hausrat→Glas/Schmuck). Zwraca odpowiedź, `sources`,
`breadcrumbs`, intent, flagę `abstained`.

## Podsumowanie użycia LLM

| Krok | LLM? | Model | Provider |
|------|------|-------|----------|
| enrichment (build) | ✅ | gpt-4o-mini | OpenRouter |
| embeddings (build) | ⚙️ lokalny | BGE-M3 | — |
| QueryExpansion | ✅ | llama-4-scout-17b | Groq |
| _detect_tarif / DocFilter | ❌ reguły | — | — |
| Retriever (bi-encoder) | ⚙️ lokalny | BGE-M3 | — |
| Reranker (opc.) | ⚙️ lokalny | bge-reranker-v2-m3 | — |
| ContextPruner | ❌ reguły | — | — |
| Generator | ✅ | llama-3.1-8b / llama-3.3-70b | Groq |
| Critic (+ensemble) | ✅ | qwen3-32b / llama-3.3-70b | Groq |

Czyli **4 punkty LLM** (1 w buildzie, 3 w runtime) + 2 modele lokalne (embedder, reranker).
Reszta to deterministyczne reguły. To wyjaśnia, dlaczego diagnozowane porażki były „ciche":
treść istnieje, ale deterministyczny filtr taryfy (regex) i gruba granulacja parsera (reguły)
nie dostarczają właściwego chunku do LLM-generatora.

## Powiązania

- Mapowanie modeli per krok: `src/model_registry.py`
- Klienci LLM (Groq runtime / OpenRouter batch): `src/llm_providers.py`
- Diagnoza luk retrievalu (Naturgefahren, załącznik Kfz): issues G1 (section filter),
  G3 (tarif detection gate), G5 (cross-encoder reranker)
