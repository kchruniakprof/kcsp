"""Standalone timing analysis — reads eval_te3small_stability_out.log"""
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

LOG = Path("eval_te3small_stability_out.log")

lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
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

pipelines = []
current = None
for ev in events:
    if ev.get("event") == "pipeline_start":
        if current:
            pipelines.append(current)
        current = {"query": ev.get("query", ""), "start": ev["_ts"], "steps": []}
    elif current is not None:
        current["steps"].append(ev)
if current:
    pipelines.append(current)

stats = []
for p in pipelines:
    rec = {"intent": None, "t_expansion": None, "t_retriever": None,
           "t_generator": None, "t_critic": None, "t_total": None, "verdict": None}
    prev = p["start"]
    for ev in p["steps"]:
        if ev.get("event") == "step_done":
            step = ev.get("step")
            dt = (ev["_ts"] - prev).total_seconds()
            prev = ev["_ts"]
            if step == "query_expansion":
                rec["intent"] = ev.get("intent")
                rec["t_expansion"] = dt
            elif step == "retriever":
                rec["t_retriever"] = dt
            elif step == "generator":
                rec["t_generator"] = dt
            elif step == "critic":
                rec["t_critic"] = dt
                if rec["verdict"] is None:
                    rec["verdict"] = ev.get("verdict")
    if p["steps"]:
        rec["t_total"] = (p["steps"][-1]["_ts"] - p["start"]).total_seconds()
    stats.append(rec)

by_intent = defaultdict(list)
for s in stats:
    by_intent[s["intent"] or "UNKNOWN"].append(s)

def avg(lst, k):
    v = [x[k] for x in lst if x[k]]
    return round(sum(v)/len(v), 1) if v else "-"

print(f"Parsed: {len(stats)} pytań\n")
print(f"{'Intent':<24} {'N':>3} {'Pass%':>6} {'expand':>7} {'retriev':>8} {'gen':>6} {'critic':>7} {'total':>7}")
print("-"*72)
for intent, recs in sorted(by_intent.items()):
    passed = sum(1 for r in recs if r["verdict"] == "PASS")
    pct = round(passed/len(recs)*100) if recs else 0
    print(f"{intent:<24} {len(recs):>3} {pct:>5}% "
          f" {str(avg(recs,'t_expansion'))+'s':>7}"
          f" {str(avg(recs,'t_retriever'))+'s':>8}"
          f" {str(avg(recs,'t_generator'))+'s':>6}"
          f" {str(avg(recs,'t_critic'))+'s':>7}"
          f" {str(avg(recs,'t_total'))+'s':>7}")
print("-"*72)
all_passed = sum(1 for r in stats if r["verdict"] == "PASS")
print(f"{'TOTAL':<24} {len(stats):>3} {round(all_passed/len(stats)*100):>5}% "
      f" {str(avg(stats,'t_expansion'))+'s':>7}"
      f" {str(avg(stats,'t_retriever'))+'s':>8}"
      f" {str(avg(stats,'t_generator'))+'s':>6}"
      f" {str(avg(stats,'t_critic'))+'s':>7}"
      f" {str(avg(stats,'t_total'))+'s':>7}")

fails = [(r["intent"] or "?", p["query"][:80]) for r, p in zip(stats, pipelines) if r["verdict"] != "PASS"]
if fails:
    print("\nFAILS:")
    for intent, q in fails:
        print(f"  [{intent}] {q}")
