#!/usr/bin/env python3
"""Summarise label-log-likelihood evaluations (experiments/multitask/*/eval_lbl/metrics.json)."""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/multitask")
rows = []
for p in sorted(root.glob("*/eval_lbl/metrics.json")):
    run, m = p.parent.parent.name, json.loads(p.read_text())
    for t, r in m["templates"].items():
        for task in ("A", "B"):
            if task in r:
                rows.append({"run": run, "template": t, "task": task, "wf1": r[task]["_avg"],
                             "wf1_lenorm": r[f"{task}_mean"]["_avg"]})
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no label-evaluation results found")
Path("results/tables").mkdir(parents=True, exist_ok=True)
df.to_csv("results/tables/lbl_eval.csv", index=False)
pd.set_option("display.width", 220)
for task in ("A", "B"):
    d = df[df.task == task]
    pv = d.pivot(index="run", columns="template", values="wf1")
    held = [c for c in pv.columns if c.startswith("h")]
    if held:
        pv["held_mean"], pv["held_min"] = pv[held].mean(axis=1), pv[held].min(axis=1)
    probes = [c for c in pv.columns if c.startswith("s")]
    if probes:
        pv["probe_mean"] = pv[probes].mean(axis=1)
    print(f"\n=== Task {task}: 10-language weighted F1 by template (sum of label log-probs) ===")
    print(pv.round(4).to_string())
