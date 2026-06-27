# PRD — kcsp Webapp

*Source of decisions: `HANDOFF_webapp_2026-06-24.md` (grill-me session, all decisions locked).*

---

## Problem description

ERGO P&C brokers and internal users who use the kcsp RAG system have no web interface. The only way to query the bot today is through the CLI or the promptfoo evaluation harness. There is no way to maintain conversation threads, track per-answer reasoning, manage users, or monitor LLM costs from a browser. The RAG pipeline (`RAGAssistant.ask()`) is production-ready and validated by evals; what is missing is the web layer on top.

## Solution

Build a webapp at `https://ai-datalab.duckdns.org/kcsp` that wraps `RAGAssistant.ask()` behind a FastAPI + React frontend. It mirrors the UX of the reference `dkv_belgium_v3` webapp (Google login, admin approval, threaded chat, per-answer trace drawer, budget/kill-switch controls) but is re-skinned to ERGO branding and adapted to kcsp's 4-stage pipeline. The app deploys as a separate Docker service that joins the existing Caddy/dkv Docker network — no new Caddy container is needed.

## User stories

### Authentication & access

1. As a new visitor, I want to log in with my Google account so that I do not need a separate password.
2. As a new user after Google login, I want to see a "pending approval" screen so that I know my account is awaiting review.
3. As a blocked user, I want to see a clear blocked page so that I understand I cannot access the bot.
4. As an admin, I want to see new sign-ups in a pending queue so that I can approve or block them without touching the database directly.
5. As an admin, I want to approve a pending user so that they gain chat access immediately.
6. As an admin, I want to block a user so that they cannot send further queries.
7. As an approved user, I want my session to persist across browser refreshes via a secure HTTP-only cookie so that I do not have to log in repeatedly.
8. As an approved user, I want to log out so that my session cookie is cleared.

### Threads & chat

9. As an approved user, I want to create a new conversation thread so that I can organise questions by topic.
10. As an approved user, I want to see all my past threads in a sidebar so that I can return to previous conversations.
11. As an approved user, I want to rename a thread so that I can give it a meaningful title.
12. As an approved user, I want to delete a thread and all its messages so that I can clean up old conversations.
13. As an approved user, I want to type a question in any language and press Send so that the kcsp RAG pipeline is triggered.
14. As an approved user, I want to see a live stage indicator while the answer is being computed (query expansion → retrieval → generation → critic) so that I know the system is working.
15. As an approved user, I want the answer rendered as Markdown so that policy text with bullets and structure is readable.
16. As an approved user, I want to see a clear abstain message in English when the system cannot answer confidently, so that I know to consult a specialist instead of trusting a potentially wrong answer.
17. As an approved user, I want cross-sell hints (e.g. "Ergänzende Produkte: Glas, Schmuck") appended to the answer when the RAG pipeline determines them to be relevant.
18. As an approved user, I want repeated identical questions to be served instantly from cache (at €0 cost) so that common queries are fast and free.
19. As an approved user, I want to see the EUR cost of each assistant message so that I can track my usage.

### Trace drawer

20. As an approved user, I want to open a trace drawer for any assistant message so that I can inspect how the answer was produced.
21. As an approved user, I want the trace drawer to show total cost in EUR and total wall-clock time so that I can see the overall latency and cost for a single request.
22. As an approved user, I want the trace drawer to show the **Query Expansion** stage output (intent, paraphrases, domain terms, section types, sparte hints, confidence, chain-of-thought) so that I can verify the system understood my question correctly.
23. As an approved user, I want the trace drawer to show the **detected tarif** (doc_filter output) so that I can confirm the right product scope was applied before retrieval.
24. As an approved user, I want the trace drawer to show **retrieved section chunks** (heading, breadcrumb, relevance score, markdown preview) so that I can see what policy text the generator had access to.
25. As an approved user, I want the trace drawer to show **Generator** details (model ID, provider, prompt/completion tokens, duration, confidence, chain-of-thought) so that I can understand where the answer came from.
26. As an approved user, I want the trace drawer to show **Critic** details (verdict PASS/ABSTAIN, confidence, reasoning bullets, chain-of-thought, retry/ensemble flags) so that I can see the quality gate result.
27. As an approved user, I want each LLM call in the trace to show model ID, provider, token counts, and latency so that I can understand detailed cost drivers.
28. As an approved user, I want cached answers to display a "cached" badge and €0 cost so that I know no LLM was invoked.

### Admin panel

29. As an admin, I want a user list with name, email, status, budget, and spend so that I have a full view of the user base.
30. As an admin, I want to set a per-user token budget in EUR so that I can control individual spending.
31. As an admin, I want to set a global daily spend limit so that I can cap total LLM costs for the day.
32. As an admin, I want a kill switch that pauses all RAG queries so that I can respond to cost overruns or incidents immediately.
33. As an admin, I want to view the full conversation history of any user so that I can audit questions and answers.

### ERGO branding & UX

34. As any user, I want to see the ERGO logo and ERGO red (`#EE0138`) as the primary accent colour so that the app is recognisably ERGO-branded.
35. As any user, I want all UI labels and chrome in English so that the interface language is consistent regardless of the user's query language (kcsp auto-detects the query language internally).
36. As any user, I want the app to be accessible at `/kcsp` with no path conflicts with the existing dkv app at the domain root.

## Implementation decisions

### Module structure

| Module | Path | Responsibility |
|---|---|---|
| `KcspRunner` | `src/webapp_runner.py` | Adapter: calls `RAGAssistant.ask()`, emits stage markers, returns a `Trace`-compatible dict |
| `TracingClient` | `src/tracing_client.py` | Wraps `instructor.from_openai`, intercepts `create_with_completion` to capture tokens and compute `cost_eur` from a pricing table |
| Backend | `webapp/backend/` | Copied from `dkv_belgium_v3/webapp/backend/`; only `DB_PATH`, `root_path`, and app title change |
| Frontend | `webapp/frontend/` | Copied from `dkv_belgium_v3/webapp/frontend/`; strings → English, `vite base="/kcsp/"`, ERGO tokens applied |

### RAG adapter (`KcspRunner`)

`KcspRunner.run(query, stage_callback)` bridges the gap between the dkv `rag.run(query, stage_callback=...)` contract (expected by `chat.py`) and kcsp's `RAGAssistant.ask(query)` (which has no callback):

- Reuses the lru-cached factory from `src/promptfoo_provider.py` to obtain a `RAGAssistant` instance (no cold-start per request).
- Calls `stage_callback` with stage names at the start of each stage: `"query_expansion"` → `"retrieval"` → `"generation"` → `"critic"`.
- After `ask()` returns, maps `FinalAnswer` fields onto the trace payload:
  - `chunks_json` ← `RetrievalResult` objects (heading, breadcrumb, markdown, section_id, score)
  - `cited_sources_json` ← `FinalAnswer.sources` (list of `section_id` ints)
  - `query_expansion_detail_json` ← intent, normalized_query, paraphrases, section_types, domain_terms, sparte_hints, confidence, chain_of_thought
  - `critic_detail_json` ← verdict, confidence, reason, retry flags
  - `total_cost_eur` ← sum of all `TracingClient`-tracked calls during the `ask()` run
  - `answer_markdown` ← `FinalAnswer.answer`
  - `abstained` ← `FinalAnswer.abstained`
- Internally uses a thread-local `TraceCollector` (same pattern as dkv) to collect per-LLM-call costs that `TracingClient` emits.
- Fields with no kcsp equivalent (`selector_detail_json`, `pruning_detail_json`, `used_brute_force`) are written as `null` / `False`.

### TracingClient

A wrapper around `instructor.from_openai(...)` that intercepts every LLM call:

- Uses `create_with_completion` to get both the parsed output and the raw `ChatCompletion` (which carries `usage`).
- Reads `usage.prompt_tokens` / `usage.completion_tokens`.
- Looks up `cost_eur` from a static pricing table keyed on model ID covering all entries in `src/model_registry.REGISTRY` (Groq + OpenRouter rates). The cross-encoder reranker has no token cost and is excluded from the table.
- Records a `StepCost(name, model_id, provider, prompt_tokens, completion_tokens, cost_eur, duration_ms)` into the active `TraceCollector`.
- Acts as a drop-in replacement: wraps the existing `groq_client()` and `openrouter_client()` factories; no other pipeline code changes.

### Subpath awareness

- FastAPI `create_app()` receives `root_path="/kcsp"`.
- Session cookie uses `path="/kcsp"` so it is never sent to the dkv app.
- Google OAuth redirect URI: `https://ai-datalab.duckdns.org/kcsp/auth/google/callback` (requires a new OAuth web client — not reused from dkv).
- SPA fallback serves `index.html` for all unmatched routes within the prefix.
- Vite compiled with `base: "/kcsp/"`.

### Database schema

Identical tables to dkv (`users`, `email_otps`, `threads`, `messages`, `traces`, `answer_cache`, `settings`), stored in a separate file `kcsp.db`. The two apps share no tables and have independent user bases.

### Deploy (Variant A — same host, no separate Caddy)

- `webapp/Dockerfile`: same multi-stage pattern as dkv (Node build → Python runtime; Vite output copied to `backend/static/`).
- `docker-compose.kcsp.yml`: single service `kcsp-app`, `expose: 8000` only (no `ports:` — host port 8000 is already taken by dkv), `kcsp_data` volume for `kcsp.db`, external network `dkv_belgium_v3_default`.
- dkv's `Caddyfile` gets one new block inserted before the root catch-all:
  ```
  handle_path /kcsp/* {
      reverse_proxy kcsp-app:8000
  }
  ```
  (`caddy reload` on the running Caddy container — zero-downtime for Caddy 2).
- `.env.kcsp` (never committed) holds: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `SESSION_SECRET`, `ADMIN_EMAILS`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `EMBED_MODEL`, `EMBED_API_KEY`, `EMBED_BASE_URL`, `DB_PATH=/data/kcsp.db`.

### Frontend re-skin

All hardcoded DKV colours (`#004b43`, `#c07000`, `#98a92a`) replaced via `ergo-tokens.css`:
- `--ergo-primary: #EE0138` — stage block headers, primary buttons, accent borders
- `--ergo-primary-dark: #B8001A` — hover states
- `--ergo-neutral-100: #F5F5F5` / `--ergo-neutral-200: #E0E0E0`

Font: Mulish (Google Fonts) as FS Me fallback. ERGO logo SVG in header. All UI strings in English.

DKV-specific elements removed: FAQ links, DKV PDF URL tiles in the trace drawer (kcsp uses breadcrumbs as source references, not external PDF links).

### Trace drawer for kcsp

Four named stage blocks in order:

1. **Query Expansion** — intent badge, paraphrases, domain terms, sparte hints, section types, confidence, chain-of-thought (collapsible); LLM call metadata (model, tokens, ms).
2. **Retrieval** — detected tarif chip, list of retrieved chunks (heading, breadcrumb, score, markdown preview). No LLM cost here (vector + BM25 + reranker).
3. **Generator** — model, tokens, duration, confidence, chain-of-thought.
4. **Critic** — verdict badge (PASS/ABSTAIN), confidence, reasoning bullets, chain-of-thought, ensemble/retry flags.

Sections absent in kcsp (decomposition, context pruning, selector) are omitted entirely.

## Testing decisions

**What a good test looks like:** tests verify externally observable behaviour through public interfaces. They do not assert on private attributes, internal call ordering, or which sub-function was invoked. Unit tests for the adapter inject a fake `RAGAssistant` returning a known `FinalAnswer`; backend tests use a real in-memory SQLite DB passed to `create_app()`; frontend tests assert rendered output via React Testing Library.

**Modules to test:**

| Module | Test type | Key assertions |
|---|---|---|
| `KcspRunner` | Unit (fake `RAGAssistant`) | `stage_callback` called with all 4 stage names in order; `chunks_json`, `cited_sources_json`, `critic_detail_json`, `answer_markdown` all non-empty/non-null; `total_cost_eur ≥ 0` |
| `TracingClient` | Unit (mocked `create_with_completion`) | `cost_eur > 0` when token counts > 0; `StepCost` recorded in collector; zero cost when both token counts are 0 |
| Backend auth/admin/chat | Integration (in-memory SQLite) | Login → pending → admin approve → active; ask → poll → trace; cache hit returns `cached=True` and `cost_eur=0` |
| `_build_trace_payload` | Unit (in-memory DB with a Trace row) | Returns all required keys; `steps`, `chunks`, `cited_sources` are lists; `total_cost_eur` is numeric |

Existing patterns to follow: `/app/dkv_belgium_v3/webapp/backend/` pytest files (use FastAPI `TestClient`, in-memory engine passed to `create_app()`).

## Out of scope

- Any changes to `src/` (RAG pipeline logic is frozen — the adapter wraps, never modifies).
- OTP / email verification (Google login is sufficient; kcsp has no existing email-OTP requirement).
- Shared user base or SSO between kcsp and dkv apps.
- Custom ERGO npm design package (no public package exists; tokens are defined manually in CSS).
- PDF source links in the trace drawer (kcsp uses breadcrumbs; no external PDF URLs).
- Broker50 / promptfoo eval regressions (separate track, unrelated to the webapp).
- Mobile-responsive design (internal tool, desktop browser assumed).
- Multi-turn context (kcsp RAG is stateless per query; threads provide history display only, not RAG context).

## Additional notes

- **Phase 0 user action required:** register a new Google OAuth web client with redirect URI `https://ai-datalab.duckdns.org/kcsp/auth/google/callback`; provide `ADMIN_EMAILS` and `EMBED_*` API key. Auth cannot be built until these are provided.
- **Caddyfile edit risk:** modifying dkv's live Caddyfile touches the dkv deployment. `caddy reload` is zero-downtime in Caddy 2 but coordinate a reload window as a precaution.
- **Host port 8000:** already published by `dkv-webapp`. The kcsp compose file must use `expose: 8000` (not `ports:`).
- **Pricing table:** must be added manually to `TracingClient` — `src/model_registry.py` stores only model name strings. Add Groq rates for `meta-llama/llama-4-scout-17b-16e-instruct`, `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `qwen/qwen3-32b`; OpenRouter rate for `openai/gpt-4o-mini-2024-07-18`.
- **Version tag for cache keying:** should encode the kcsp parquet + model registry version so that an eval improvement does not serve stale cached answers from a prior configuration.
