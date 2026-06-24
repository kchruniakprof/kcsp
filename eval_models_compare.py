"""
Multi-model embedding comparison: rebuild parquet + run broker50 + full99 for each model.
Results saved to eval_compare_results.json
"""
import subprocess, os, json, shutil, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
PARQUET = ROOT / "parquet"
BACKUP = ROOT / "parquet_backup_fireworks_8b"

MODELS = [
    {
        "slug": "openai-text-embedding-3-large",
        "model": "openai/text-embedding-3-large",
        "dim": 3072,
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    {
        "slug": "google-gemini-embedding-2",
        "model": "google/gemini-embedding-2",
        "dim": 3072,
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    {
        "slug": "mistralai-mistral-embed-2312",
        "model": "mistralai/mistral-embed-2312",
        "dim": 1024,
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    {
        "slug": "baai-bge-m3",
        "model": "baai/bge-m3",
        "dim": 1024,
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
]

def run(cmd, env=None, cwd=ROOT):
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(cmd, shell=True, cwd=cwd, env=e, capture_output=False)

def parse_json_results(path):
    try:
        data = json.loads(Path(path).read_text())
        results = data["results"]["results"]
        passed = sum(1 for r in results if r.get("success"))
        failed = sum(1 for r in results if not r.get("success"))
        total = len(results)
        fails = [r["vars"]["question"] for r in results if not r.get("success")]
        return {"passed": passed, "failed": failed, "total": total, "pct": round(passed/total*100, 1), "fail_questions": fails}
    except Exception as e:
        return {"error": str(e)}

# ── Backup current Fireworks parquet ─────────────────────────────────────────
if not BACKUP.exists():
    print(f"[compare] Backing up Fireworks parquet to {BACKUP}")
    shutil.copytree(PARQUET, BACKUP)
else:
    print(f"[compare] Backup already exists: {BACKUP}")

all_results = {}

for cfg in MODELS:
    slug = cfg["slug"]
    print(f"\n{'='*70}")
    print(f"[compare] MODEL: {cfg['model']} (dim={cfg['dim']})")
    print(f"{'='*70}")
    t_start = time.time()

    env = {
        "EMBED_MODEL": cfg["model"],
        "EMBED_DIM": str(cfg["dim"]),
        "EMBED_BASE_URL": cfg["base_url"],
        "EMBED_API_KEY_ENV": cfg["key_env"],
    }

    # 1. Rebuild parquet
    print(f"[compare] Rebuilding parquet...")
    t0 = time.time()
    run("python -m src.build_embeddings --parquet-dir parquet --no-validate", env=env)
    rebuild_time = round(time.time() - t0, 1)
    print(f"[compare] Rebuild done in {rebuild_time}s")

    # 2. broker50 eval
    print(f"[compare] Running broker50...")
    t0 = time.time()
    run(f"promptfoo eval --config promptfooconfig.broker50.yaml --output results_{slug}_broker50.json", env=env)
    broker_time = round(time.time() - t0, 1)
    broker = parse_json_results(ROOT / f"results_{slug}_broker50.json")
    print(f"[compare] broker50: {broker.get('passed')}/{broker.get('total')} ({broker.get('pct')}%) in {broker_time}s")

    # 3. full99 eval
    print(f"[compare] Running full99...")
    t0 = time.time()
    run(f"promptfoo eval --config promptfooconfig.full.yaml --output results_{slug}_full99.json", env=env)
    full_time = round(time.time() - t0, 1)
    full = parse_json_results(ROOT / f"results_{slug}_full99.json")
    print(f"[compare] full99: {full.get('passed')}/{full.get('total')} ({full.get('pct')}%) in {full_time}s")

    all_results[slug] = {
        "model": cfg["model"],
        "dim": cfg["dim"],
        "rebuild_s": rebuild_time,
        "broker50": broker,
        "broker50_s": broker_time,
        "full99": full,
        "full99_s": full_time,
        "total_s": round(time.time() - t_start, 1),
    }

# ── Restore Fireworks parquet ─────────────────────────────────────────────────
print(f"\n[compare] Restoring Fireworks parquet from {BACKUP}")
shutil.rmtree(PARQUET)
shutil.copytree(BACKUP, PARQUET)
# Restore env (reset to Fireworks defaults via no env override needed - code defaults)

# ── Save + print summary ──────────────────────────────────────────────────────
out = ROOT / "eval_compare_results.json"
out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

print(f"\n\n{'='*70}")
print("FINAL COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"{'Model':<38} {'Dim':>5} {'broker50':>9} {'full99':>9} {'total_t':>8}")
print("-"*70)
for slug, r in all_results.items():
    b = f"{r['broker50'].get('pct','?')}%" if 'pct' in r.get('broker50',{}) else "ERR"
    f = f"{r['full99'].get('pct','?')}%" if 'pct' in r.get('full99',{}) else "ERR"
    print(f"{r['model']:<38} {r['dim']:>5} {b:>9} {f:>9} {r['total_s']:>7}s")

print(f"\nSaved: {out}")
print("\n[compare] DONE — Fireworks parquet restored.")
