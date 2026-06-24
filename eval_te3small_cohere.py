"""
Eval: openai/text-embedding-3-small (OpenRouter, 1536 dim) + cohere/rerank-4-fast (OpenRouter)
"""
import subprocess, os, json, shutil, time
from pathlib import Path

ROOT = Path(__file__).parent
PARQUET = ROOT / "parquet"
BACKUP_FW8B = ROOT / "parquet_backup_fireworks_8b"
BACKUP_TE3S = ROOT / "parquet_backup_te3small"

ENV = {
    "EMBED_MODEL":       "openai/text-embedding-3-small",
    "EMBED_DIM":         "1536",
    "EMBED_BASE_URL":    "https://openrouter.ai/api/v1",
    "EMBED_API_KEY_ENV": "OPENROUTER_API_KEY",
    "RERANKER_MODEL":    "cohere/rerank-4-fast",
    "PYTHONIOENCODING":  "utf-8",
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

# Backup current parquet (qwen3-8b) if not already done
if not BACKUP_FW8B.exists():
    print(f"[te3s] Backing up qwen3-8b parquet to {BACKUP_FW8B}")
    shutil.copytree(PARQUET, BACKUP_FW8B)
else:
    print(f"[te3s] qwen3-8b backup exists: {BACKUP_FW8B}")

print("="*65)
print("EVAL: text-embedding-3-small (OpenRouter) + cohere/rerank-4-fast")
print("="*65)

# Rebuild parquet
print("\n[te3s] Rebuilding parquet (1536 dim)...")
t0 = time.time()
run("python -m src.build_embeddings --parquet-dir parquet --no-validate")
print(f"[te3s] Rebuild done in {round(time.time()-t0,1)}s")

# Backup te3-small parquet
if not BACKUP_TE3S.exists():
    shutil.copytree(PARQUET, BACKUP_TE3S)
    print(f"[te3s] te3-small parquet backed up to {BACKUP_TE3S}")

# broker50
run("promptfoo cache clear")
print("\n[te3s] broker50...")
t0 = time.time()
run("promptfoo eval --config promptfooconfig.broker50.yaml --output results_te3small-cohere-fast_broker50.json")
b = parse(ROOT / "results_te3small-cohere-fast_broker50.json")
print(f"broker50: {b.get('passed')}/{b.get('total')} ({b.get('pct')}%) in {round(time.time()-t0,1)}s | errors={b.get('errors')}")
if b.get("fails"):
    print("FAILS:", b["fails"])

# full99
run("promptfoo cache clear")
print("\n[te3s] full99...")
t0 = time.time()
run("promptfoo eval --config promptfooconfig.full.yaml --output results_te3small-cohere-fast_full99.json")
f = parse(ROOT / "results_te3small-cohere-fast_full99.json")
print(f"full99:   {f.get('passed')}/{f.get('total')} ({f.get('pct')}%) in {round(time.time()-t0,1)}s | errors={f.get('errors')}")
if f.get("fails"):
    print("FAILS:", f["fails"])

# Restore qwen3-8b parquet
print(f"\n[te3s] Restoring qwen3-8b parquet from {BACKUP_FW8B}")
shutil.rmtree(PARQUET)
shutil.copytree(BACKUP_FW8B, PARQUET)

print("\n" + "="*65)
print("WYNIKI FINALNE")
print("="*65)
print(f"broker50: {b.get('pct','?')}%  full99: {f.get('pct','?')}%")
combined = round((b.get('pct',0)*50 + f.get('pct',0)*99) / 149, 1) if b.get('pct') and f.get('pct') else '?'
print(f"combined: {combined}%")
