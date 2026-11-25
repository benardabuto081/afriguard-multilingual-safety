#!/usr/bin/env python3
"""Summarise E1 runs (full-dev, ten-language) into tables."""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/baselines")
rows, long = [], []
for p in sorted(root.glob("*/metrics.json")):
    m = json.loads(p.read_text())
    c = m["config"]
    if c.get("limit"):
        continue
    row = {"model": c["model"], "mode": c["mode"], "shots": c["shots"],
           "input": "prompt" if c["prompt_only"] else "both", "split": c["split"],
           "params_B": round(m["n_params"] / 1e9, 2) if m.get("n_params") else None,
           "minutes": round(m["total_seconds"] / 60, 1), "run_id": c["run_id"]}
    for t in ("A", "B"):
        v = m["tasks"].get(t)
        if v and v["n_languages"] == 10:
            row[f"{t}_wf1"] = round(v["avg_weighted_f1"], 4)
            row[f"{t}_unparsed"] = round(sum(r["unparsed_rate"] for r in v["per_language"].values()) / 10, 4)
            for lang, r in v["per_language"].items():
                long.append({"run_id": c["run_id"], "model": c["model"], "mode": c["mode"],
                             "shots": c["shots"], "task": t, "language": lang,
                             "weighted_f1": r["weighted_f1"], "macro_f1": r["macro_f1"],
                             "unparsed": r["unparsed_rate"]})
    rows.append(row)
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit(f"No full runs found under {root}")
df = df.sort_values(["model", "shots", "mode"])
Path("results/tables").mkdir(parents=True, exist_ok=True)
df.to_csv("results/tables/e1_baselines.csv", index=False)
pd.DataFrame(long).to_csv("results/tables/e1_per_language.csv", index=False)
pd.set_option("display.width", 200)
print(df.drop(columns=["run_id", "split"]).to_string(index=False))
lg = pd.DataFrame(long)
for t in ("A", "B"):
    pv = lg[lg.task == t].pivot_table(index="language", columns=["model", "shots", "mode"], values="weighted_f1")
    print(f"\nTask {t} per-language weighted F1")
    print(pv.round(3).to_string())
