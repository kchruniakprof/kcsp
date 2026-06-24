"""
Stability run: te3-small (OpenRouter) + cohere/rerank-4-fast
Measures accuracy + pipeline timing per intent/question type.
"""
import subprocess, os, json, time, re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).parent
LOG  = ROOT / "eval_te3small_stability.log"

ENV = {
    "EMBED_MODEL":       "openai/text-embedding-3-small",
    "EMBED_DIM":         "1536",
    "EMBED_BASE_URL":    "https://openrouter.ai/api/v1",
    "EMBED_API_KEY_ENV": "OPENROUTER_API_KEY",
    "RERANKER_MODEL":    "cohere/rerank-4-fast",
    "PYTHONIOENCODING":  "utf-8",
}

def run(cmd, capture_stderr_to=None):
    e = os.environ.copy(); e.update(ENV)
    if capture_stderr_to:
        with open(capture_stderr_to, "ab") as f:
            return subprocess.run(cmd, shell=True, cwd=ROOT, env=e, stderr=f)
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=e)

def parse_results(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = data["results"]["results"]
    passed = sum(1 for r in results if r.get("success"))
    total  = len(results)
    fails  = [r["vars"]["question"] for r in results if not r.get("success") and not r.get("error")]
    return {"passed": passed, "total": total, "pct": round(passed/total*100,1), "fails": fails}

def parse_timing(log_path):
    """Parse structured JSON logs → per-question timing + intent breakdown."""
    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()

    events = []
    for line in lines:
        line = line.strip()
        if line.startswith("Python worker stderr: "):
            line = line[len("Python worker stderr: "):]
        try:
            obj = json.loads(line)
            if "timestamp" in obj:
                obj["_ts"] = datetime.fromisoformat(obj["timestamp"].replace("Z", "+00:00"))
                events.append(obj)
        except Exception:
            pass

    # Group events by pipeline (each pipeline_start begins a new question)
    pipelines = []
    current = None
    for ev in events:
        if ev.get("event") == "pipeline_start":
            if current:
                pipelines.append(current)
            current = {"query": ev.get("query",""), "start": ev["_ts"], "steps": []}
        elif current is not None:
            current["steps"].append(ev)
    if current:
        pipelines.append(current)

    # Per-pipeline: extract intent + step durations
    stats = []
    for p in pipelines:
        rec = {"query": p["query"], "intent": None, "sparte": None,
               "t_expansion": None, "t_retriever": None, "t_generator": None,
               "t_critic": None, "t_total": None, "verdict": None}

        step_times = {}
        prev_ts = p["start"]
        for ev in p["steps"]:
            if ev.get("event") == "step_done":
                step = ev.get("step")
                dt = (ev["_ts"] - prev_ts).total_seconds()
                step_times[step] = dt
                prev_ts = ev["_ts"]
                if step == "query_expansion":
                    rec["intent"]  = ev.get("intent")
                    rec["sparte"]  = (ev.get("sparte_hints") or ["?"])[0]
                if step == "critic" and rec["verdict"] is None:
                    rec["verdict"] = ev.get("verdict")

        rec["t_expansion"] = step_times.get("query_expansion")
        rec["t_retriever"] = step_times.get("retriever")
        rec["t_generator"] = step_times.get("generator")
        rec["t_critic"]    = step_times.get("critic")
        if p["steps"]:
            rec["t_total"] = (p["steps"][-1]["_ts"] - p["start"]).total_seconds()
        stats.append(rec)

    return stats

def summarize(stats):
    by_intent = defaultdict(list)
    for s in stats:
        by_intent[s["intent"] or "UNKNOWN"].append(s)

    print(f"\n{'='*80}")
    print(f"TIMING + ACCURACY PER INTENT ({len(stats)} pytań)")
    print(f"{'='*80}")
    hdr = f"{'Intent':<22} {'N':>3} {'Pass%':>6} {'expand':>7} {'retriev':>8} {'gen':>6} {'critic':>7} {'total':>7}"
    print(hdr)
    print("-"*80)

    totals = defaultdict(list)
    for intent, recs in sorted(by_intent.items()):
        passed = sum(1 for r in recs if r["verdict"] == "PASS")
        pct    = round(passed/len(recs)*100) if recs else 0
        def avg(key): vals = [r[key] for r in recs if r[key]]; return round(sum(vals)/len(vals),1) if vals else None
        row = (f"{intent:<22} {len(recs):>3} {pct:>5}%"
               f" {str(avg('t_expansion'))+'s':>7}"
               f" {str(avg('t_retriever'))+'s':>8}"
               f" {str(avg('t_generator'))+'s':>6}"
               f" {str(avg('t_critic'))+'s':>7}"
               f" {str(avg('t_total'))+'s':>7}")
        print(row)
        for k in ["t_expansion","t_retriever","t_generator","t_critic","t_total"]:
            if avg(k): totals[k].append(avg(k))

    print("-"*80)
    all_passed = sum(1 for r in stats if r["verdict"] == "PASS")
    def gavg(k): v = [r[k] for r in stats if r[k]]; return round(sum(v)/len(v),1) if v else None
    print(f"{'TOTAL':<22} {len(stats):>3} {round(all_passed/len(stats)*100):>5}%"
          f" {str(gavg('t_expansion'))+'s':>7}"
          f" {str(gavg('t_retriever'))+'s':>8}"
          f" {str(gavg('t_generator'))+'s':>6}"
          f" {str(gavg('t_critic'))+'s':>7}"
          f" {str(gavg('t_total'))+'s':>7}")

    # Fails per intent
    fails_by_intent = {i: [r["query"][:80] for r in recs if r["verdict"] != "PASS"]
                       for i, recs in by_intent.items()}
    has_fails = {i: f for i, f in fails_by_intent.items() if f}
    if has_fails:
        print(f"\nFAILS per intent:")
        for intent, qs in sorted(has_fails.items()):
            print(f"  {intent}:")
            for q in qs:
                print(f"    - {q}")

print("="*65)
print("STABILITY RUN: te3-small + cohere/rerank-4-fast")
print("="*65)

LOG.write_bytes(b"")  # reset log

run("promptfoo cache clear")
print("\n[stab] broker50...")
t0 = time.time()
run("promptfoo eval --config promptfooconfig.broker50.yaml --output results_stability_broker50.json", capture_stderr_to=LOG)
b = parse_results(ROOT / "results_stability_broker50.json")
print(f"broker50: {b['passed']}/{b['total']} ({b['pct']}%) in {round(time.time()-t0,1)}s")

run("promptfoo cache clear")
print("\n[stab] full99...")
t0 = time.time()
run("promptfoo eval --config promptfooconfig.full.yaml --output results_stability_full99.json", capture_stderr_to=LOG)
f = parse_results(ROOT / "results_stability_full99.json")
print(f"full99:   {f['passed']}/{f['total']} ({f['pct']}%) in {round(time.time()-t0,1)}s")

combined = round((b['pct']*50 + f['pct']*99)/149, 1)
print(f"\ncombined: {combined}%")
if b.get("fails"): print("broker50 FAILS:", b["fails"])
if f.get("fails"): print("full99 FAILS:",   f["fails"])

# Timing analysis from log
print("\n[stab] Parsing timing from log...")
try:
    stats = parse_timing(LOG)
    summarize(stats)
except Exception as exc:
    print(f"[stab] Timing parse error: {exc}")
