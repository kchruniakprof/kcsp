# Handoff — kcsp Webapp (implementation focus)

**Date:** 2026-06-24
**Next session goal:** implement the kcsp webapp per the locked plan below.
**Status:** planning complete (grill-me), nothing built yet. Plan doc and memory writes were rejected by the user, so this handoff is the only persisted copy of the decisions — treat it as source of truth.

---

## Goal

Build a webapp for **kcsp** (ERGO P&C insurance RAG bot) and publish it at
`https://ai-datalab.duckdns.org/kcsp`.

Model it on the existing reference app at `/app/dkv_belgium_v3/webapp` (FastAPI + SQLModel/SQLite backend, React+Vite + Porsche Design System frontend, Caddy + DuckDNS + Docker). Features wanted: Google login, admin approval of users, client panel with threads/messages, per-answer trace drawer, ERGO branding, kcsp RAG logic plugged in.

Reference app internals already studied — do not re-explore from scratch, read the files directly:
- Backend: `/app/dkv_belgium_v3/webapp/backend/{main,db,auth,google_oauth,auth_routes,admin,chat,budget,cache}.py`
- Frontend: `/app/dkv_belgium_v3/webapp/frontend/src/*.tsx` (uses `@porsche-design-system/components-react`, ~250 inline styles, hardcoded colors `#004b43` green / `#c07000` amber)
- Deploy: `/app/dkv_belgium_v3/{Caddyfile,docker-compose.yml,.env.example}`, `webapp/Dockerfile`

---

## Locked decisions (from grill-me interview)

| # | Topic | Choice |
|---|-------|--------|
| 1 | RAG integration | **Thin adapter** around `RAGAssistant.ask()`. NO change to kcsp RAG logic. |
| 2 | URL | subpath `/kcsp`, single shared Caddy |
| 3 | Cost tracking | **full `TracingClient`** — token-counter wrapper, additive, does not change answers |
| 4 | DB / identity | separate `kcsp.db`; **new** Google OAuth client (not reused from dkv) |
| 5 | Design | re-skin Porsche DS → ERGO tokens (keep Porsche components) |
| 6 | UI language | **English** chrome; user questions pass through unchanged (kcsp auto-detects language) |
| 7 | Trace drawer | **full per-stage** (4 kcsp stages) + cost/time |
| 8 | Deploy | separate compose for kcsp + docker network bridge into dkv's Caddy |
| 9 | Cross-sell | **ON** (kcsp appends "Ergänzende Produkte: …") |

User-confirmed: "details like the original bot" = mirror dkv chat behavior (threads, SSE stage streaming, abstain message, polling) adapted to kcsp's 4 stages.

---

## Host facts (detected on THIS machine, `docker ps`)

- dkv runs here. Container `dkv_belgium_v3-caddy-1` owns host ports **80/443**. Docker network: `dkv_belgium_v3_default`.
- Host port **8000 is already published** by container `dkv-webapp` → kcsp-app must NOT publish 8000; `expose` only.
- → **Deploy Variant A (same host):** kcsp-app joins `dkv_belgium_v3_default` as an *external* network; Caddy reaches it by service name. Only one Caddy may hold 443, so kcsp gets NO own Caddy.

---

## kcsp RAG surface (what the adapter wraps)

- **Builder to reuse:** lru-cached factory at `src/promptfoo_provider.py:156` already constructs a working `RAGAssistant`. Lift it into a reusable function.
- **Entry point:** `RAGAssistant.ask(query) -> FinalAnswer` (`src/ragassistant.py:77`). `FinalAnswer = {answer, sources: list[int], breadcrumbs, intent, abstained, cross_sell}`.
- **Stages + rich intermediates** (all available for the trace):
  1. `query_expansion.expand()` → intent, normalized_query, paraphrases, section_types, domain_terms, sparte_hints, confidence, chain_of_thought, detected_language
  2. doc_filter → detected_tarif
  3. `retriever.retrieve_multi()` → `RetrievalResult{section_id, heading, markdown, breadcrumb, score}`
  4. `generator.generate()` → mode, answer, sources(int), breadcrumbs
  5. `run_critic()` → verdict, confidence, reason
- **LLM clients:** `instructor.from_openai(...)` — groq (runtime) + openrouter (batch). Token usage via `create_with_completion`. See `src/llm_providers.py`.
- **GAP vs dkv:** dkv webapp's `chat.py` calls `rag.run(query, stage_callback=...)` and expects rich `Trace` fields. kcsp only has `ask()`. The adapter bridges this.
- **Data:** parquet ready in `/app/kcsp/parquet/`. Embedder via `EMBED_*` env (Fireworks / te3-small — see `promptfoo_provider.py`).
- **Caveat:** `src/model_registry.py` holds bare model-name strings, **no provider/pricing**. The `TracingClient` cost feature needs a pricing table (groq + openrouter rates) added.

---

## Implementation phases

**Phase 0 — Prereqs (user, outside code):**
- Register a NEW Google OAuth web client → client id/secret; redirect URI `https://ai-datalab.duckdns.org/kcsp/auth/google/callback`.
- Provide `ADMIN_EMAILS` for kcsp and the embedder API key (`EMBED_*`).
- (Secrets go in `.env.kcsp`, never committed.)

**Phase 1 — Backend scaffold:** copy domain-neutral backend from dkv into `/app/kcsp/webapp/backend/` (`db, auth, google_oauth, auth_routes, admin, chat, budget, cache, main`). Change app title, `DB_PATH=kcsp.db`, `root_path="/kcsp"`.

**Phase 2 — RAG adapter (core):**
- `src/webapp_runner.py` → `KcspRunner.run(query, stage_callback)`: reuse the factory, emit stage markers (query_expansion → retriever → generator → critic), return a trace payload mapped onto dkv `Trace` fields (`chunks_json` from RetrievalResult heading+breadcrumb+markdown; `cited_sources` from generated.sources; `critic_detail`; `query_expansion_detail`).
- `TracingClient` for kcsp: wrap instructor via `create_with_completion` → usage tokens; add pricing table → `cost_eur`. Wrapper only.

**Phase 3 — Subpath awareness:** FastAPI `root_path="/kcsp"`, cookie `path="/kcsp"`, OAuth redirect prefix, SPA fallback under prefix.

**Phase 4 — Frontend:** copy `webapp/frontend/`, strings → English, `vite base="/kcsp/"`. Add `ergo-tokens.css` (`--ergo-red:#EE0138`, neutrals, FS Me + Mulish fallback). Replace hardcoded `#004b43`/`#c07000` → ERGO. ERGO logo in header. Build → `backend/static/`.

**Phase 5 — Deploy (Variant A):** `webapp/Dockerfile` (dkv pattern); `docker-compose.kcsp.yml` (service `kcsp-app`, volume `kcsp_data`, `.env.kcsp`, `expose: 8000` only, external network `dkv_belgium_v3_default`). Edit dkv's Caddyfile: add `handle_path /kcsp/* { reverse_proxy kcsp-app:8000 }`.

**Phase 6 — Tests + smoke:** adapt dkv backend tests (auth/admin/chat); adapter test (stages emitted, trace populated, cost>0); E2E local (login → admin approve → ask → trace), then prod.

---

## ERGO branding reference (from public web search)

- **ERGO Rot** `#EE0138` (RGB 238,1,56 · Pantone 185C · RAL 3026) — primary.
- Secondary 2019 palette (MetaDesign): pastel violet / green / ice blue / yellow / orange / warm grey — official hex not found; define tokens manually, let user correct.
- Corporate font **FS Me** (FontSmith, **paid**) — confirmed embedded in ERGO PDFs. Use free **Mulish** (Google Fonts) as web fallback.
- No public ERGO React/NPM design package ("ERGO Brand Coach" is gated) → tokens by hand.
- Sources: schemecolor.com/ergo-group-logo-colors.php ; designtagebuch.de ERGO Markenauftritt.

---

## Open / watch-outs

- Editing dkv's Caddyfile + adding kcsp to `dkv_belgium_v3_default` touches the dkv deployment — minimal but real. Coordinate.
- kcsp `model_registry` lacks pricing — required for cost tracking (Phase 2).
- Confirm the running `dkv-webapp` container (publishing :8000) is intentional; it occupies the host port kcsp would otherwise want.
- Embeddings need a live `EMBED_*` API key at runtime (Fireworks/OpenAI).

---

## Suggested skills for next session

- **`improve-codebase-architecture`** — when shaping the adapter + TracingClient so kcsp RAG logic stays untouched.
- **`tdd`** — Phase 2/6: write the adapter/trace/cost tests first.
- **`verify`** / **`run`** — Phase 6: launch the app and confirm login → approve → ask → trace end-to-end.
- **`update-config`** — if Bash permission prompts (docker, npm, vite build) need allow-listing.
- **`code-review`** — before deploy, review the adapter + backend diff.
- **`security-review`** — auth/OAuth/cookie/subpath surface before going public.

## Unfinished from this session
- Re-run broker50 to list the 4 failures was interrupted (full99 = 97/99 passed; broker50 = 46/50, 4 abstains identified earlier: Smart→Best Tarifwechsel, Sachverständigenverfahren 25k€, Schmuck Außenversicherung weltweit, Verzugszinsen 6%). Not related to webapp; separate track.
