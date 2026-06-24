# ADR-012: Reranker + Embed Latency Experiment — lokalny BGE vs API (Cohere / Jina / Fireworks) vs brak

**Status:** Accepted  
**Date:** 2026-06-24  
**Deciders:** K. Chruniak  

---

## Context

Po przejściu na Qwen3-Embedding-8B (ADR-011) retriever nadal zajmował ~16s/pytanie. Analiza per-krok ujawniła, że bottleneck to **nie** embedding API (Fireworks ~3s), lecz **lokalny CrossEncoderReranker** (`BAAI/bge-reranker-v2-m3`, 568M parametrów, CPU inference).

### Baseline (ADR-011): Qwen3-8B Fireworks + lokalny BGE reranker

| Krok | Czas |
|---|---|
| query_expansion (Groq) | ~2s |
| retriever: embed (Fireworks API) | ~1–3s |
| retriever: CrossEncoder rerank (CPU, 30 par) | **~13–14s** |
| generator (Groq) | ~0.7s |
| critic (Groq) | ~1.5s |
| **razem** | **~18–21s/pytanie** |

CrossEncoder liczy oddzielny forward pass przez 568M model dla każdej z 30 par (query, doc[:512]) na CPU — stąd dominacja w latencji.

W ramach eksperymentu:
1. Przetestowano różne kombinacje embed model + reranker
2. Stały zestaw eval: broker50 (50q) + full99 (99q)

---

## Eksperymenty — pełna macierz wyników

| Embed | Reranker | broker50 | full99 | combined | retriever avg |
|---|---|---|---|---|---|
| qwen3-8b (Fireworks) | BGE lokalny (CPU) | 96% | 98% | 97.3% | ~14s |
| **te3-small (OpenRouter)** | **`cohere/rerank-4-fast`** | **98%** | **98%** | **98%** | **~1.5s** |
| te3-large (OpenRouter) | `cohere/rerank-4-fast` | 98% | 96% | 96.6% | 1.5s |
| te3-large (OpenRouter) | `jina-reranker-v2-base-multilingual` | 96% | 97% | 96.6% | 1.2s |
| qwen3-8b (Fireworks) | `qwen3-reranker-8b` (Fireworks) | 96% | 93.9% | 94.9% | ~0.8s |
| te3-large (OpenRouter) | brak | 90% | 92.9% | 91.9% | **0.6s** |
| te3-large (OpenRouter) | `cohere/rerank-4-pro` | 100% | ❌ | — | — |

### Uwagi

- **te3-small + Cohere fast**: **najlepszy wynik** — combined 98%, broker50=98%, full99=98%. Tańszy i szybszy embed niż te3-large (1536 dim vs 3072 dim), lepszy combined o +1.4pp. Niespodziewanie wysoki wynik.
- **BGE lokalny**: combined 97.3% ale retriever 14s (CPU, 568M params × 30 forward passes). Nieakceptowalna latencja w produkcji.
- **Bez rerankera**: -6pp vs baseline, retriever 24× szybszy (0.6s). Akceptowalny tylko gdy latencja jest priorytetem absolutnym.
- **cohere/rerank-4-fast z te3-large**: broker50 +2pp, full99 -2pp vs baseline. Retriever 1.5s — 10× szybszy.
- **cohere/rerank-4-pro**: broker50=100%, full99 ❌ — OpenRouter 403 tier restriction po ~5 requestach.
- **jina-reranker-v2-base-multilingual**: combined=96.6%, retriever 1.2s. Free tier wyczerpany (403 Insufficient balance) — nie testowano z qwen3-8b embed.
- **qwen3-reranker-8b (Fireworks)**: combined=94.9%, najszybszy API reranker (0.8s). Response format `data` zamiast `results` — klasa `FireworksReranker`. Nie rekomendowany.

---

## Decision

**Zwycięzca: `openai/text-embedding-3-small` (OpenRouter) + `cohere/rerank-4-fast` (OpenRouter)**  
Combined 98%, retriever ~1.5s, jeden dostawca (OpenRouter), tańszy embed niż te3-large.

**Uzasadnienie:**
- Najwyższy combined ze wszystkich testowanych kombinacji (+0.7pp vs BGE lokalny, +1.4pp vs Jina/Cohere+te3-large)
- broker50=98% i full99=98% — symetryczny wynik, brak tradeoff między zbiorami
- te3-small (1536 dim) tańszy i szybszy w embed niż te3-large (3072 dim) — lepszy wynik przy niższym koszcie
- Jeden dostawca (OpenRouter) dla embed i reranker — prostszy deployment
- Retriever ~1.5s — 9× szybszy od lokalnego BGE

**Porównanie finałowe:**

| Metryka | **te3-small+Cohere** | Lokalny BGE | te3-large+Cohere | te3-large+Jina | Bez rerankera |
|---|---|---|---|---|---|
| broker50 | **98%** | 96% | 98% | 96% | 90% |
| full99 | **98%** | 98% | 96% | 97% | 92.9% |
| combined | **98%** | 97.3% | 96.6% | 96.6% | 91.9% |
| retriever | 1.5s | 14s | 1.5s | 1.2s | 0.6s |
| deployment | OpenRouter | CPU | OpenRouter | OpenRouter+Jina | — |

---

## Consequences

- `retriever.py`: klasy `JinaAPIReranker`, `CohereAPIReranker`, `FireworksReranker` z retry/backoff
- `promptfoo_provider.py`: wybór rerankera przez env var `RERANKER_MODEL`:
  - `DISABLE_RERANKER=true` → brak
  - `jina-*` → `JinaAPIReranker` (JINA_API_KEY)
  - `accounts/fireworks/*` → `FireworksReranker` (FIREWORKS_API_KEY)
  - `cohere/*` → `CohereAPIReranker` (OPENROUTER_API_KEY)
  - (default) → lokalny `CrossEncoderReranker`
- `.env`: dodano `JINA_API_KEY`
- **Rekomendowana konfiguracja produkcyjna:**
  - `EMBED_MODEL=openai/text-embedding-3-small`
  - `EMBED_DIM=1536`
  - `EMBED_BASE_URL=https://openrouter.ai/api/v1`
  - `EMBED_API_KEY_ENV=OPENROUTER_API_KEY`
  - `RERANKER_MODEL=cohere/rerank-4-fast`

## Stability run (2026-06-24) — te3-small + Cohere fast, run 2

Drugi run tego samego zestawu: broker50=**96%**, full99=**96%**, combined=**96%** (vs 98% w run 1, delta −2pp).

**Wnioski:**
- Wynik niestabilny na poziomie ±2pp między runami — prawdopodobnie losowość critic (Qwen3-32b)
- Zakres wyników: 96–98% combined, prawdziwa "jakość" systemu to przedział, nie punkt
- Potrzebne ≥3 runy dla miarodajnej oceny stabilności

**Timing per intent (run 2, 149 pytań):**

| Intent | N | expand | retriever | generator | critic | total |
|---|---|---|---|---|---|---|
| CLAIMS_PROCEDURE | 9 | 1.7s | 1.5s | 0.8s | 2.0s | 6.3s |
| COMPARISON | 15 | 2.0s | 1.6s | 2.7s | — | 6.2s |
| COMPLAINT | 2 | 3.6s | 1.5s | 0.9s | 2.2s | 8.2s |
| COVERAGE_QUERY | 62 | 1.9s | 1.9s | 0.8s | 2.0s | 7.4s |
| EXCLUSION_QUERY | 26 | 1.8s | 2.2s | 0.8s | 2.7s | 8.1s |
| **GENERAL_INFO** | 30 | 1.9s | **7.3s** | 0.8s | 2.3s | **13.0s** |
| OUT_OF_SCOPE | 4 | 1.2s | — | — | — | 1.2s |
| PRICE_QUOTE | 1 | 0.9s | 1.8s | 0.5s | 1.5s | 4.8s |
| **TOTAL** | **149** | **1.9s** | **3.0s** | **1.0s** | **2.2s** | **8.3s** |

*COMPARISON i OUT_OF_SCOPE nie przechodzą przez krok critic.*

**Bottleneck GENERAL_INFO:** retriever 7.3s (4× wolniejszy od innych). Pytania GENERAL_INFO nie mają sprecyzowanej sparte/tarif → query_expansion generuje więcej parafraz → więcej Cohere rerank calls na szerszym pool. Candidate fix: ograniczyć paraphrases_count dla GENERAL_INFO lub dodać early-stop w rerankingu.

## Revisit triggers

- Cohere `/rerank-4-pro` odblokowany na OpenRouter → retest (broker50=100% obiecujące)
- Jina doładowana → retest qwen3-8b + Jina (potencjalnie najwyższy combined)
- qwen3-reranker-8b poprawiony (nowy checkpoint) → retest, aktualnie -3.1pp vs te3-small+Cohere
- combined < 96% w kolejnym stability run → zbadać GENERAL_INFO retriever bottleneck
- ≥3 stability runy → obliczyć mean ± std dla combined
