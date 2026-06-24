"""
Eval pipeline: text-embedding-3-large + Cohere reranker (fast vs pro) via OpenRouter.
Parquet already built for text-embedding-3-large — no rebuild needed.
"""
import subprocess, os, json, time
from pathlib import Path

ROOT = Path(__file__).parent

RERANKERS = [
    {"slug": "cohere-rerank-4-fast", "model": "cohere/rerank-4-fast"},
    {"slug": "cohere-rerank-4-pro",  "model": "cohere/rerank-4-pro"},
]

BASE_ENV = {
    "EMBED_MODEL":       "openai/text-embedding-3-large",
    "EMBED_DIM":         "3072",
    "EMBED_BASE_URL":    "https://openrouter.ai/api/v1",
    "EMBED_API_KEY_ENV": "OPENROUTER_API_KEY",
    "PYTHONIOENCODING":  "utf-8",
}

def run(cmd, env=None):
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=e, capture_output=False)

def parse(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results = data["results"]["results"]
        passed = sum(1 for r in results if r.get("success"))
        total = len(results)
        fails = [r["vars"]["question"] for r in results if not r.get("success")]
        return {"passed": passed, "total": total, "pct": round(passed / total * 100, 1), "fails": fails}
    except Exception as exc:
        return {"error": str(exc)}

all_results = {}

for cfg in RERANKERS:
    slug = cfg["slug"]
    env = {**BASE_ENV, "RERANKER_MODEL": cfg["model"]}

    print(f"\n{'='*60}")
    print(f"[cohere] RERANKER: {cfg['model']}")
    print(f"{'='*60}")

    # broker50
    run("promptfoo cache clear", env=env)
    print(f"[cohere] broker50 ...")
    t0 = time.time()
    run(f"promptfoo eval --config promptfooconfig.broker50.yaml --output results_{slug}_broker50.json", env=env)
    b = parse(ROOT / f"results_{slug}_broker50.json")
    print(f"[cohere] broker50: {b.get('passed')}/{b.get('total')} ({b.get('pct')}%) in {round(time.time()-t0,1)}s")

    # full99
    run("promptfoo cache clear", env=env)
    print(f"[cohere] full99 ...")
    t0 = time.time()
    run(f"promptfoo eval --config promptfooconfig.full.yaml --output results_{slug}_full99.json", env=env)
    f = parse(ROOT / f"results_{slug}_full99.json")
    print(f"[cohere] full99: {f.get('passed')}/{f.get('total')} ({f.get('pct')}%) in {round(time.time()-t0,1)}s")

    all_results[slug] = {"model": cfg["model"], "broker50": b, "full99": f}

out = ROOT / "eval_cohere_results.json"
out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
print(f"{'Model':<35} {'broker50':>9} {'full99':>9}")
print("-"*55)
for slug, r in all_results.items():
    b = f"{r['broker50'].get('pct','?')}%" if "pct" in r.get("broker50", {}) else "ERR"
    f2 = f"{r['full99'].get('pct','?')}%" if "pct" in r.get("full99", {}) else "ERR"
    print(f"{r['model']:<35} {b:>9} {f2:>9}")

print(f"\nSaved: {out}")
print("[cohere] DONE.")
