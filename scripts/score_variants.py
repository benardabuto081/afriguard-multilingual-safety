#!/usr/bin/env python3
"""Exposure of a run to alternative scoring rules: summed log-prob vs per-token mean (from metrics.json)
vs per-character normalisation (lm-eval acc_norm style, assumed). Also the share of Task A items predicted unsafe."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from afriguard.data import load_split
from afriguard.labels import LANGUAGES, labels_for
from afriguard.templates import ALL, answers

run = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/multitask/e2p_full/eval_lbl")
m = json.loads((run / "metrics.json").read_text())
rows = []
for t in m["templates"]:
    z, tpl = np.load(run / f"lp_{t}.npz"), ALL[t]
    for task in ("A", "B"):
        labs = labels_for(task)
        chars = np.array([len(x) for x in answers(tpl, task)], dtype=float)
        ws, wc, us, uc = [], [], [], []
        for lang in LANGUAGES:
            ids, S = z[f"{lang}_{task}_ids"], z[f"{lang}_{task}"]
            gold = load_split("data/raw", task, lang, "dev").set_index("id").loc[ids, "label"].tolist()
            ps, pc = S.argmax(1), (S / chars).argmax(1)
            ws.append(f1_score(gold, [labs[i] for i in ps], average="weighted", zero_division=0))
            wc.append(f1_score(gold, [labs[i] for i in pc], average="weighted", zero_division=0))
            us.append(float((ps == 1).mean()))
            uc.append(float((pc == 1).mean()))
        row = {"template": t, "task": task, "sum": np.mean(ws), "per_char": np.mean(wc),
               "per_token": m["templates"][t].get(f"{task}_mean", {}).get("_avg", np.nan)}
        if task == "A":
            row["unsafe_share_sum"], row["unsafe_share_char"] = np.mean(us), np.mean(uc)
        rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"run: {run}   (gold unsafe share on dev is about 0.50)")
print(df.round(4).to_string(index=False))
Path("results/reports").mkdir(parents=True, exist_ok=True)
df.to_csv(f"results/reports/score_variants_{run.parent.name}.csv", index=False)
