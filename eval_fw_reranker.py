"""
Eval 2: qwen3-embedding-8b (Fireworks) + qwen3-reranker-8b (Fireworks)
Runs broker50 + full99, parses results, prints summary.
"""
import subprocess, os, json, time
from pathlib import Path

ROOT = Path(__file__).parent

ENV = {
    "EMBED_MODEL":      "accounts/fireworks/models/qwen3-embedding-8b",
    "EMBED_DIM":        "4096",
    "EMBED_BASE_URL":   "https://api.fireworks.ai/inference/v1",
    "EMBED_API_KEY_ENV": "FIREWORKS_API_KEY",
    "RERANKER_MODEL":   "accounts/fireworks/models/qwen3-reranker-8b",
    "PYTHONIOENCODING": "utf-8",
}

def run(cmd):
    e = os.environ.copy()
    e.update(ENV)
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=e)

def parse(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results = data["results"]["results"]
        passed = sum(1 for r in results if r.get("success"))
        total  = len(results)
        errs   = sum(1 for r in results if r.get("error"))
        fails  = [r["vars"]["question"] for r in results if not r.get("success") and not r.get("error")]
        return {"passed": passed, "total": total, "pct": round(passed/total*100,1), "errors": errs, "fails": fails}
    except Exception as exc:
        return {"error": str(exc)}

print("="*65)
print("EVAL: qwen3-8b embed + qwen3-reranker-8b (Fireworks)")
print("="*65)

# broker50
run("promptfoo cache clear")
print("\n[fw-reranker] broker50...")
t0 = time.time()
run("promptfoo eval --config promptfooconfig.broker50.yaml --output results_qwen8b-fw-reranker_broker50.json")
b = parse(ROOT / "results_qwen8b-fw-reranker_broker50.json")
print(f"broker50: {b.get('passed')}/{b.get('total')} ({b.get('pct')}%) in {round(time.time()-t0,1)}s | errors={b.get('errors')}")
if b.get("fails"):
    print("FAILS:", b["fails"])

# full99
run("promptfoo cache clear")
print("\n[fw-reranker] full99...")
t0 = time.time()
run("promptfoo eval --config promptfooconfig.full.yaml --output results_qwen8b-fw-reranker_full99.json")
f = parse(ROOT / "results_qwen8b-fw-reranker_full99.json")
print(f"full99:   {f.get('passed')}/{f.get('total')} ({f.get('pct')}%) in {round(time.time()-t0,1)}s | errors={f.get('errors')}")
if f.get("fails"):
    print("FAILS:", f["fails"])

print("\n" + "="*65)
print("WYNIKI FINALNE")
print("="*65)
print(f"broker50: {b.get('pct','?')}%  full99: {f.get('pct','?')}%")
