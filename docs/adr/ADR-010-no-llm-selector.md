# ADR-010 — LLMSelector zastąpiony przez CrossEncoderReranker

**Status:** Accepted  
**Data:** 2026-06-21

## Kontekst

ADR-008 przewidywał `LLMSelector` (ContextSelector) jako osobną warstwę reranking/abstain między Retrieverem a Generatorem. G5 zaimplementował `CrossEncoderReranker` wewnątrz Retrievera, który pełni tę samą rolę (`pool_k` → rerank → `top_k`).

`EmbeddingPruner` istniał jako alternatywny pruner semantyczny dla `ContextPruner`, ale nigdy nie został aktywowany w pipeline.

`promptfoo_provider.py` używał `ContextSelector` do shadow-scoringu w ramach kalibracji `EMBED_THRESHOLD` (E1). Kalibracja jest zakończona — wynik: `EMBED_THRESHOLD = None` (bimodalna dystrybucja, żaden próg nie pomaga). Shadow scoring nie jest już potrzebny.

## Decyzja

`LLMSelector`, `ContextSelector`, `SelectedChunk`, `Abstain` oraz `EmbeddingPruner` zostają usunięte z codebase. `CrossEncoderReranker` w `Retriever` jest kanonicznym mechanizmem rerankingu. Warstwa `ContextSelector` nie jest potrzebna przy obecnej architekturze.

## Konsekwencje

- Przyszłe przeglądy architektury nie powinny sugerować przywrócenia `LLMSelector` jako osobnej warstwy.
- Jeśli potrzebny będzie scoring semantyczny na poziomie zdania — `EmbeddingPruner` można odtworzyć z git history.
- -187 LOC obciążenia poznawczego; pipeline ma jeden mechanizm rerankingu (`CrossEncoder` w `Retriever`).
- `selector_confidence` usunięte z metadanych `promptfoo_provider.py` (było zawsze `None` od momentu zakończenia kalibracji E1).
