# H3 — Usuń martwy kod: LLMSelector i EmbeddingPruner

> Typ: AFK
> Zablokowane przez: brak — start natychmiast

## Co należy zbudować

Usuwa dwa moduły, które nie są wywoływane w żadnej aktywnej ścieżce pipeline'u:

- **`src/llm_selector.py`** (114 LOC) — LLMSelector/ContextSelector przewidziany w ADR-008 jako warstwa reranking+abstain. Nigdy nie wstrzyknięty do RAGAssistant. Funkcję tę przejął `CrossEncoderReranker` w `Retriever` (G5).
- **`src/embedding_pruner.py`** (73 LOC) — alternatywny pruner semantyczny; nie importowany w żadnym aktywnym module.

Test usunięcia: usuń oba → żaden import w aktywnym kodzie nie pęka. Złożoność nie pojawia się ponownie u innych wywołujących — ponieważ nie ma wywołujących.

Razem: -187 LOC obciążenia poznawczego; AI nawigująca codebase nie musi rozstrzygać czy "LLMSelector jest nieużywany czy używany inaczej".

Jako część tego zgłoszenia należy napisać **ADR-010** dokumentujący decyzję:

```markdown
# ADR-010 — LLMSelector zastąpiony przez CrossEncoderReranker

## Status: Accepted

## Kontekst
ADR-008 przewidywał LLMSelector (ContextSelector) jako osobną warstwę reranking/abstain między Retrieverem a Generatorem. G5 zaimplementował CrossEncoderReranker wewnątrz Retrievera, który pełni tę samą rolę (pool_k → rerank → top_k).

## Decyzja
LLMSelector i EmbeddingPruner zostają usunięte. CrossEncoderReranker w Retriever jest kanonicznym mechanizmem rerankingu. Warstwa ContextSelector nie jest potrzebna przy obecnej architekturze.

## Konsekwencje
Przyszłe przeglądy architektury nie powinny sugerować przywrócenia LLMSelector. Jeśli potrzebny będzie scoring semantyczny na poziomie zdania — EmbeddingPruner można odtworzyć z git history.
```

## Kryteria akceptacji

- [ ] `src/llm_selector.py` usunięty z repozytorium
- [ ] `src/embedding_pruner.py` usunięty z repozytorium
- [ ] Żaden aktywny moduł (`ragassistant.py`, `retriever.py`, `critic.py`, `generator.py`, `query_expansion.py`, `doc_filter.py`) nie importuje usuniętych modułów
- [ ] Testy dotyczące `LLMSelector` / `EmbeddingPruner` usunięte lub przeniesione do git history
- [ ] `docs/adr/ADR-010-no-llm-selector.md` napisany i zacommitowany
- [ ] `pytest --ignore=tests/test_hierarchy_parser.py -q` — zielony (247+ testów, możliwy spadek liczby jeśli usunięto testy martwego kodu)

## Blokowane przez

Brak — można rozpocząć natychmiast.
