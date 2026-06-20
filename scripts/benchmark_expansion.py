"""
Benchmark query_expansion across Groq models + Gemini 2.5 Flash Lite (OpenRouter).
Tests against the 26 failing queries with known expected intent/sparte.
Reports: latency, intent accuracy, sparte accuracy, error rate.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

import yaml
import openai
import instructor
from pydantic import ValidationError

# Safe print for Windows consoles with limited encoding
import io
_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

def _p(*args, **kwargs):
    kwargs.setdefault("file", _stdout)
    print(*args, **kwargs)

from src.query_expansion import ExpandedQuery, _SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------

GROQ_BASE = "https://api.groq.com/openai/v1"
OR_BASE   = "https://openrouter.ai/api/v1"

MODELS: list[tuple[str, str, str, str]] = [
    # (label, base_url, api_key_env, model_id)
    ("llama-3.1-8b",       GROQ_BASE, "GROQ_API_KEY", "llama-3.1-8b-instant"),
    ("llama-3.3-70b",      GROQ_BASE, "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    ("llama-4-scout",      GROQ_BASE, "GROQ_API_KEY", "meta-llama/llama-4-scout-17b-16e-instruct"),
    ("gpt-oss-20b",        GROQ_BASE, "GROQ_API_KEY", "openai/gpt-oss-20b"),
    ("gpt-oss-120b",       GROQ_BASE, "GROQ_API_KEY", "openai/gpt-oss-120b"),
    ("qwen3-32b",          GROQ_BASE, "GROQ_API_KEY", "qwen/qwen3-32b"),
    ("qwen3.6-27b",        GROQ_BASE, "GROQ_API_KEY", "qwen/qwen3.6-27b"),
    ("gemini-2.5-flash-lite", OR_BASE, "OPENROUTER_API_KEY", "google/gemini-2.5-flash-lite"),
]

# ---------------------------------------------------------------------------
# Test queries with ground truth (from eval_failing.yaml)
# ---------------------------------------------------------------------------

FAILING_YAML = Path(__file__).parent.parent / "eval_failing.yaml"

def load_test_cases() -> list[dict]:
    with open(FAILING_YAML, encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    cases = []
    for e in entries:
        v = e.get("vars", {})
        q = v.get("question", "")
        if not q:
            continue
        cases.append({
            "question":        q,
            "expected_intent": v.get("expected_intent"),
            "expected_sparte": v.get("expected_sparte"),
            "language":        v.get("language", "de"),
        })
    return cases


# ---------------------------------------------------------------------------
# Run one model
# ---------------------------------------------------------------------------

def make_client(base_url: str, api_key_env: str) -> instructor.Instructor:
    key = os.environ[api_key_env]
    raw = openai.OpenAI(api_key=key, base_url=base_url)
    return instructor.from_openai(raw, mode=instructor.Mode.MD_JSON)


def run_model(label: str, base_url: str, api_key_env: str, model_id: str,
              cases: list[dict]) -> dict:
    client = make_client(base_url, api_key_env)

    intent_ok = 0
    sparte_ok = 0
    errors    = 0
    latencies: list[float] = []

    _p(f"\n[{label}] {model_id}")
    _p(f"  {'Q':<55} {'intent':>15}  {'sparte':>8}  {'ms':>6}")
    _p(f"  {'-'*55} {'-'*15}  {'-'*8}  {'-'*6}")

    for c in cases:
        q      = c["question"]
        e_int  = c["expected_intent"]
        e_sp   = c["expected_sparte"]

        t0 = time.perf_counter()
        try:
            result: ExpandedQuery = client.chat.completions.create(
                model=model_id,
                response_model=ExpandedQuery,
                temperature=0,
                top_p=1,
                seed=42,
                max_retries=2,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": q},
                ],
            )
            ms = int((time.perf_counter() - t0) * 1000)
            latencies.append(ms)

            got_int = result.intent.value
            got_sp  = result.sparte_hint

            ok_int = (got_int == e_int)
            ok_sp  = (got_sp == e_sp) or (e_sp is None and got_sp is None)

            if ok_int: intent_ok += 1
            if ok_sp:  sparte_ok += 1

            int_mark = "+" if ok_int else "x"
            sp_mark  = "+" if ok_sp  else "x"
            _p(f"  {q[:55]:<55} {int_mark}{got_int:>14}  {sp_mark}{(got_sp or 'null'):>7}  {ms:>6}")

        except Exception as ex:
            ms = int((time.perf_counter() - t0) * 1000)
            errors += 1
            _p(f"  {q[:55]:<55} {'ERROR':>15}  {'':>8}  {ms:>6}  [{type(ex).__name__}: {str(ex)[:40]}]")

    n = len(cases)
    valid = n - errors
    avg_ms = int(sum(latencies) / len(latencies)) if latencies else 0
    p95_ms = int(sorted(latencies)[int(len(latencies) * 0.95)]) if latencies else 0

    _p(f"\n  SUMMARY  intent={intent_ok}/{valid} ({100*intent_ok//max(valid,1)}%)  "
       f"sparte={sparte_ok}/{valid} ({100*sparte_ok//max(valid,1)}%)  "
       f"errors={errors}  avg={avg_ms}ms  p95={p95_ms}ms")

    return {
        "label":      label,
        "model":      model_id,
        "n":          n,
        "errors":     errors,
        "intent_pct": 100 * intent_ok // max(valid, 1),
        "sparte_pct": 100 * sparte_ok // max(valid, 1),
        "avg_ms":     avg_ms,
        "p95_ms":     p95_ms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cases = load_test_cases()
    _p(f"Loaded {len(cases)} test cases from {FAILING_YAML}")

    results = []
    for label, base_url, api_key_env, model_id in MODELS:
        try:
            r = run_model(label, base_url, api_key_env, model_id, cases)
            results.append(r)
        except Exception as ex:
            _p(f"  SKIP {label}: {ex}")

    # Final leaderboard
    _p("\n" + "="*80)
    _p("LEADERBOARD")
    _p("="*80)
    _p(f"  {'Model':<22} {'Intent%':>8} {'Sparte%':>8} {'Errors':>7} {'Avg ms':>8} {'P95 ms':>8}")
    _p(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

    results.sort(key=lambda x: (-x["intent_pct"], x["avg_ms"]))
    for r in results:
        _p(f"  {r['label']:<22} {r['intent_pct']:>7}% {r['sparte_pct']:>7}% "
           f"{r['errors']:>7} {r['avg_ms']:>7}ms {r['p95_ms']:>7}ms")


if __name__ == "__main__":
    main()
