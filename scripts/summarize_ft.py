#!/usr/bin/env python3
"""Summarise fine-tuning runs (experiments/multitask/*/eval_dev/metrics.json)."""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/multitask")
rows, per = [], {}
for p in sorted(root.glob("*/eval_dev/metrics.json")):
    run, m = p.parent.parent.name, json.loads(p.read_text())
    if m["config"].get("limit"):
        continue
    cp = p.parent.parent / "config.json"
    c = json.loads(cp.read_text()) if cp.exists() else {}
    row = {"run": run, "n_train": c.get("n_examples"), "tok_mean": round(c["tokens_mean"]) if c.get("tokens_mean") else None,
           "train_min": c.get("train_minutes"), "peak_gb": c.get("peak_gpu_gb"), "eval_min": round(m["total_seconds"] / 60, 1)}
    for t, v in m["tasks"].items():
        langs = {k: x for k, x in v.items() if not k.startswith("_")}
        if len(langs) != 10:
            continue
        row[f"{t}_wf1"] = round(v["_avg_weighted_f1"], 4)
        row[f"{t}_unparsed"] = round(sum(x["unparsed_rate"] for x in langs.values()) / 10, 4)
        for k, x in langs.items():
            per.setdefault((t, k), {})[run] = round(x["weighted_f1"], 3)
    rows.append(row)
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no runs found")
Path("results/tables").mkdir(parents=True, exist_ok=True)
df.to_csv("results/tables/ft_runs.csv", index=False)
pd.set_option("display.width", 220)
print(df.to_string(index=False))
pl = pd.DataFrame(per).T
pl.index.names = ["task", "language"]
pl.to_csv("results/tables/ft_per_language.csv")
for t in ("A", "B"):
    if t in pl.index.get_level_values(0):
        print(f"\nTask {t} per-language weighted F1")
        print(pl.loc[t].to_string())
