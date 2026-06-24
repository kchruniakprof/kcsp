# ADR-011: Embedding Model Switch — BGE-M3 (local) → Qwen3-Embedding-8B (Fireworks API)

**Status:** Accepted  
**Date:** 2026-06-23  
**Deciders:** K. Chruniak  

---

## Context

Retrieval pipeline używa dwóch kanałów (RRF fusion): dense (cosine similarity) + sparse (BM25). Oryginalny model to BGE-M3 uruchomiony lokalnie — generował zarówno dense jak i sparse (lexical) wektory.

### Poprzedni model: BAAI/bge-m3 (lokalny)

| Właściwość | Wartość |
|---|---|
| Deployment | Lokalny (CPU), FP16 |
| Wymiar dense | 1024 |
| Sparse | Własne wagi leksykalne (colbert-style) |
| Czas/zapytanie | ~35–40s (CPU) |
| RAM | ~1.1 GB |
| Eval accuracy | brak ustalonego baseline przed switchem |

Główny problem: **35–40s/zapytanie** uniemożliwia uruchomienie równoległych ewaluacji (dwie instancje + konflikty indeksu BM25 → crash workera). Pełny eval 99-przypadkowy zajmował >1h w praktyce.

---

## Decision

Przejście na **Qwen3-Embedding-8B** hostowany przez **Fireworks AI** (`accounts/fireworks/models/qwen3-embedding-8b`), z lokalnym BM25 jako sparse.

| Właściwość | Wartość |
|---|---|
| Deployment | Fireworks AI API (serverless) |
| Model | `accounts/fireworks/models/qwen3-embedding-8b` |
| Wymiar dense | 4096 |
| Sparse | BM25 lokalny (TF × IDF, `bm25_encoder.py`) |
| Czas/zapytanie (wall) | ~25–28s (bottleneck: Fireworks API latency) |
| Endpoint | `https://api.fireworks.ai/inference/v1` |

RRF fusion zachowana bez zmian: `score = 1/(k+rank_dense) + 1/(k+rank_sparse)`, k=60.

### Dlaczego Qwen3-Embedding-8B

- MTEB Multilingual: najlepszy dostępny model przez API w momencie decyzji
- Fireworks AI: brak per-minute rate limitów na on-demand (vs OpenRouter: kolejki, rate limity)
- BM25 jako sparse: czyste Python, zero GPU, deterministyczne — zachowuje RRF architecture bez drugiego API calla

---

## Wyniki ewaluacji (Qwen3-Embedding-8B, 2026-06-23)

Dwa niezależne runy potwierdziły powtarzalność wyników:

| Eval set | Pytań | Passed | Failed | Accuracy |
|---|---|---|---|---|
| broker50 | 50 | 48 | 2 | **96.0%** |
| full99 | 99 | 97 | 2 | **98.0%** |
| **łącznie** | **149** | **145** | **4** | **97.3%** |

### Stałe faile (identyczne w obu runach)

**broker50:**
- *Schmuck na wyjeździe* — odpowiedź nie zawierała `Mehrfachversicherung` / `weltweit`
- *Verzugszinsen* — odpowiedź nie zawierała `6 %` / `Zinsen`

**full99:**
- *Führerschein verloren* — odpowiedź nie zawierała `Fahrerlaubnis` / `Führerschein`
- *Daten-/Bildträger Kasko* — odpowiedź nie zawierała `nicht versichert` / `Ausschluss`

Wszystkie 4 faile to **retrieval gaps** (właściwa sekcja nie trafia do top-10), nie błędy generatora ani logiki systemu.

---

## Consequences

- `build_embeddings.py`: `_EMBED_DIM = 4096`, `_MODEL_NAME = "accounts/fireworks/models/qwen3-embedding-8b"`, batch=16, sleep 2s między batchami (rate limit mitigation)
- `promptfoo_provider.py`: `_Qwen3Embedder` używa `FIREWORKS_API_KEY` + Fireworks base_url
- Parquet przebudowany 2026-06-23 (4096-dim dense + BM25 sparse, 483 retrieval units)
- ADR-006 (`body[:400]`) pozostaje w mocy — embedding composition bez zmian
- Bottleneck przesunął się z CPU inference (BGE-M3) na API latency (Fireworks); 4 faile to kandydaci do następnej iteracji retrieval improvements

## Revisit triggers

- Fireworks latency > 30s mediana → sprawdzić alternatywne providery (Voyage AI, Cohere)
- Accuracy < 95% po dodaniu nowych dokumentów → retune BM25 IDF + sprawdzić wymiar 4096 vs kompresja
- Retrieval gaps dla 4 failujących pytań → issue do osobnego ADR (force-include exact terms, D1)
