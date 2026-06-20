"""
Benchmark Groq models for each RAG pipeline step.
Usage: python scripts/benchmark_models.py
"""
import os, sys, time, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

STEPS = {
    "query_expansion": {
        "system": "Du bist ein Assistent für ERGO Versicherung. Klassifiziere die Anfrage. Antworte NUR mit JSON: {original_query, normalized_query, detected_language, intent, sparte_hint}. intent ∈ COVERAGE_QUERY|EXCLUSION_QUERY|CLAIMS_PROCEDURE|PRICE_QUOTE|COMPARISON|COMPLAINT|GENERAL_INFO|OUT_OF_SCOPE",
        "user": "Co jest objęte ubezpieczeniem Hausrat Smart?",
        "json": True,
    },
    "enrichment": {
        "system": "Du bist Experte für Versicherungsbedingungen. Antworte NUR mit JSON: {title, description, topic_tags}",
        "user": "## 1. Was ist versichert\nVersichert sind Sachen, die sich zur Zeit des Schadens in der Wohnung befinden. Dazu gehören Haushalts- und Gebrauchsgegenstände.",
        "json": True,
    },
    "generator_verbatim": {
        "system": "Du bist ein ERGO-Assistent. Gib relevante Bedingungsabschnitte WÖRTLICH aus. Kein Umschreiben. Nur Fakten aus den Quellen.",
        "user": "Frage: Was ist bei Hausrat Smart gegen Einbruchdiebstahl versichert?\n\nAbschnitt:\n## 3. Einbruchdiebstahl\nVersichert ist der Diebstahl von Sachen, wenn der Dieb in den Raum einbricht.",
        "json": False,
    },
    "critic": {
        "system": "Prüfe die Antwort auf Korrektheit. Antworte NUR mit JSON: {verdict, reason, confidence}. verdict ∈ PASS|REGEN|ABSTAIN",
        "user": "Frage: Was ist versichert?\nQuelle: Versichert ist der Hausrat.\nAntwort: Versichert ist der Hausrat.",
        "json": True,
    },
}

SEP = "-" * 72

def call(model, system, user, use_json):
    kwargs = dict(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - t0
    content = resp.choices[0].message.content
    tokens_in  = resp.usage.prompt_tokens
    tokens_out = resp.usage.completion_tokens
    return content, elapsed, tokens_in, tokens_out


print(f"\n{'='*72}")
print("ERGO RAG — Groq model benchmark")
print(f"{'='*72}\n")

results = {}

for step, cfg in STEPS.items():
    print(f"\n{SEP}")
    print(f"STEP: {step}")
    print(SEP)
    results[step] = []

    for model in MODELS:
        try:
            content, elapsed, t_in, t_out = call(
                model, cfg["system"], cfg["user"], cfg["json"]
            )
            tps = t_out / elapsed if elapsed > 0 else 0
            print(f"\n  [{model}]")
            print(f"  latency={elapsed:.2f}s  in={t_in}tok  out={t_out}tok  {tps:.0f}tok/s")
            preview = content[:200].replace('\n', ' ')
            print(f"  output: {preview}")
            results[step].append({
                "model": model, "latency": round(elapsed, 2),
                "tokens_in": t_in, "tokens_out": t_out, "tps": round(tps, 1),
                "ok": True
            })
        except Exception as e:
            print(f"\n  [{model}] ERROR: {e}")
            results[step].append({"model": model, "ok": False, "error": str(e)})

print(f"\n\n{'='*72}")
print("SUMMARY (latency, tok/s)")
print('='*72)
print(f"{'step':<25} {'model':<35} {'lat':>6} {'tok/s':>7}")
print("-"*72)
for step, rows in results.items():
    for r in rows:
        if r["ok"]:
            print(f"{step:<25} {r['model']:<35} {r['latency']:>5.2f}s {r['tps']:>6.0f}")
        else:
            print(f"{step:<25} {r['model']:<35}  ERROR")

print(f"\n{'='*72}\n")
