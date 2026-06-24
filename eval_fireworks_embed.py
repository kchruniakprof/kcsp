"""
Fireworks embed comparison: nomic-embed-text-v1.5 vs gte-large
Reranker: cohere/rerank-4-fast via OpenRouter (fixed)
"""
import subprocess, os, json, shutil, time
from pathlib import Path

ROOT = Path(__file__).parent
PARQUET = ROOT / "parquet"
BACKUP_TE3 = ROOT / "parquet_backup_te3large"
BACKUP_FW  = ROOT / "parquet_backup_fireworks_8b"

MODELS = [
    {
        "slug": "nomic-embed-text-v1.5",
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "base_url": "https://api.fireworks.ai/inference/v1",
        "key_env": "FIREWORKS_API_KEY",
    },
    {
        "slug": "gte-large",
        "model": "thenlper/gte-large",
        "dim": 1024,
        "base_url": "https://api.fireworks.ai/inference/v1",
        "key_env": "FIREWORKS_API_KEY",
    },
]

RERANKER_ENV = {
    "RERANKER_MODEL":    "cohere/rerank-4-fast",
    "EMBED_BASE_URL":    "https://openrouter.ai/api/v1",  # for reranker key resolution only
    "PYTHONIOENCODING":  "utf-8",
}

def run(cmd, env=None):
    e = os.environ.copy()
    if env: e.update(env)
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=e)

def parse(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results = data["results"]["results"]
        passed = sum(1 for r in results if r.get("success"))
        total = len(results)
        fails = [r["vars"]["question"] for r in results if not r.get("success") and not r.get("error")]
        errs  = sum(1 for r in results if r.get("error"))
        return {"passed": passed, "total": total, "pct": round(passed/total*100, 1), "fails": fails, "errors": errs}
    except Exception as exc:
        return {"error": str(exc)}

# Backup current parquet (te3-large) before touching it
if not BACKUP_TE3.exists():
    print(f"[fw] Backing up te3-large parquet to {BACKUP_TE3}")
    shutil.copytree(PARQUET, BACKUP_TE3)
else:
    print(f"[fw] te3-large backup exists: {BACKUP_TE3}")

all_results = {}

for cfg in MODELS:
    slug = cfg["slug"]
    print(f"\n{'='*65}\n[fw] MODEL: {cfg['model']} (dim={cfg['dim']})\n{'='*65}")

    embed_env = {
        "EMBED_MODEL":      cfg["model"],
        "EMBED_DIM":        str(cfg["dim"]),
        "EMBED_BASE_URL":   cfg["base_url"],
        "EMBED_API_KEY_ENV": cfg["key_env"],
        **RERANKER_ENV,
    }

    # Rebuild parquet
    print(f"[fw] Rebuilding parquet...")
    t0 = time.time()
    run("python -m src.build_embeddings --parquet-dir parquet --no-validate", env=embed_env)
    rebuild_s = round(time.time() - t0, 1)
    print(f"[fw] Rebuild done in {rebuild_s}s")

    # broker50
    run("promptfoo cache clear", env=embed_env)
    print(f"[fw] broker50...")
    t0 = time.time()
    run(f"promptfoo eval --config promptfooconfig.broker50.yaml --output results_{slug}_broker50.json", env=embed_env)
    b = parse(ROOT / f"results_{slug}_broker50.json")
    print(f"[fw] broker50: {b.get('passed')}/{b.get('total')} ({b.get('pct')}%) in {round(time.time()-t0,1)}s | errors={b.get('errors')}")

    # full99
    run("promptfoo cache clear", env=embed_env)
    print(f"[fw] full99...")
    t0 = time.time()
    run(f"promptfoo eval --config promptfooconfig.full.yaml --output results_{slug}_full99.json", env=embed_env)
    f = parse(ROOT / f"results_{slug}_full99.json")
    print(f"[fw] full99:   {f.get('passed')}/{f.get('total')} ({f.get('pct')}%) in {round(time.time()-t0,1)}s | errors={f.get('errors')}")

    all_results[slug] = {"model": cfg["model"], "dim": cfg["dim"], "rebuild_s": rebuild_s, "broker50": b, "full99": f}

# Restore te3-large parquet
print(f"\n[fw] Restoring te3-large parquet from {BACKUP_TE3}")
shutil.rmtree(PARQUET)
shutil.copytree(BACKUP_TE3, PARQUET)

out = ROOT / "eval_fireworks_results.json"
out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n{'='*65}\nSUMMARY\n{'='*65}")
print(f"{'Model':<32} {'Dim':>5} {'broker50':>9} {'full99':>9}")
print("-"*57)
for slug, r in all_results.items():
    b  = f"{r['broker50'].get('pct','?')}%" if "pct" in r.get("broker50",{}) else "ERR"
    f2 = f"{r['full99'].get('pct','?')}%"   if "pct" in r.get("full99",{})   else "ERR"
    print(f"{r['model']:<32} {r['dim']:>5} {b:>9} {f2:>9}")

print(f"\nSaved: {out}\n[fw] DONE.")
