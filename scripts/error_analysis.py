#!/usr/bin/env python3
"""Error analysis for a fine-tuned run's dev predictions. Statistics only; no prompt text."""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from afriguard.labels import LANGUAGES, TASK_A_LABELS, TASK_B_LABELS

SHORT = ["Crim", "Poli", "Viol", "Syst", "Sexu", "Advi", "Priv", "Hate", "Ethi", "IP"]
# Dev-phase leaderboard rank 1 (sabio, afq_lora), per-language weighted F1, as of 2026-11-26.
LEADER_A = dict(amharic=0.9165, hausa=0.9583, igbo=0.9499, oromo=0.9416, shona=0.8917,
                swahili=0.9332, twi=0.9666, wolof=0.9499, yoruba=0.9249, zulu=0.9583)
LEADER_B = dict(amharic=0.5986, hausa=0.7812, igbo=0.7577, oromo=0.6691, shona=0.6579,
                swahili=0.6419, twi=0.7059, wolof=0.6053, yoruba=0.6903, zulu=0.7292)

run = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/multitask/e2_tinyaya_joint/eval_dev")
lines = []


def emit(s=""):
    lines.append(str(s))
    print(s)


frames = []
for lang in LANGUAGES:
    pr = pd.read_csv(run / "predictions" / f"{lang}.csv", dtype=str, keep_default_na=False)
    ga = pd.read_csv(f"data/raw/task_a_safety/{lang}/dev.csv", dtype=str, keep_default_na=False)[["id", "label"]].rename(columns={"label": "gold_a"})
    gb = pd.read_csv(f"data/raw/task_b_category/{lang}/dev.csv", dtype=str, keep_default_na=False)[["id", "label"]].rename(columns={"label": "gold_b"})
    d = pr[["id", "label_a", "label_b"]].merge(ga, on="id", validate="one_to_one").merge(gb, on="id", validate="one_to_one")
    d["language"] = lang
    frames.append(d)
D = pd.concat(frames, ignore_index=True)
D["a_err"] = D.gold_a != D.label_a
D["b_err"] = D.gold_b != D.label_b
emit(f"run: {run}   items: {len(D)}   unparsed A: {(D.label_a == '__unparsed__').sum()}   unparsed B: {(D.label_b == '__unparsed__').sum()}")

p, r, f, s = precision_recall_fscore_support(D.gold_a, D.label_a, labels=TASK_A_LABELS, zero_division=0)
emit("\n== Task A, pooled per-class ==")
emit(pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=TASK_A_LABELS).round(3).to_string())

safe = D[D.gold_a == "safe"].groupby("gold_b").agg(n_safe=("a_err", "size"), safe_called_unsafe=("a_err", "mean"))
uns = D[D.gold_a == "unsafe"].groupby("gold_b").agg(n_unsafe=("a_err", "size"), unsafe_called_safe=("a_err", "mean"))
g = safe.join(uns, how="outer").reindex(TASK_B_LABELS)
g.index = SHORT
emit("\n== Task A error rate by gold category (false refusals on safe / misses on unsafe) ==")
emit(g.round(3).to_string())

p, r, f, s = precision_recall_fscore_support(D.gold_b, D.label_b, labels=TASK_B_LABELS, zero_division=0)
pc = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=SHORT)
pc["gold_share"] = (pc.support / pc.support.sum()).round(3)
pc["pred_share"] = [round(float((D.label_b == lab).mean()), 3) for lab in TASK_B_LABELS]
emit("\n== Task B, pooled per-category ==")
emit(pc.round(3).to_string())

cm = confusion_matrix(D.gold_b, D.label_b, labels=TASK_B_LABELS)
emit("\n== Task B confusion (rows = gold, columns = predicted; pooled over languages) ==")
emit(pd.DataFrame(cm, index=SHORT, columns=SHORT).to_string())
pairs = sorted(((int(cm[i, j]), SHORT[i], SHORT[j]) for i in range(10) for j in range(10) if i != j), reverse=True)[:10]
emit("\ntop confusions (count, gold -> predicted): " + "; ".join(f"{c} {a}->{b}" for c, a, b in pairs))

emit("\n== Task B error rate when Task A is right vs wrong ==")
emit(D.groupby("a_err")["b_err"].agg(["mean", "size"]).rename(index={False: "A correct", True: "A wrong"}).round(3).to_string())

ref = pd.read_csv("results/tables/reference_baseline.csv")
ref = ref[ref["input"] == "both"].pivot(index="language", columns="task", values="dev_f1")
m = json.loads((run / "metrics.json").read_text())["tasks"]
tab = pd.DataFrame({
    "A": {lang: m["A"][lang]["weighted_f1"] for lang in LANGUAGES},
    "A_leader": pd.Series(LEADER_A), "A_tfidf": ref["A"],
    "B": {lang: m["B"][lang]["weighted_f1"] for lang in LANGUAGES},
    "B_leader": pd.Series(LEADER_B), "B_tfidf": ref["B"]}).loc[LANGUAGES]
tab["B_minus_leader"] = tab.B - tab.B_leader
tab["B_minus_tfidf"] = tab.B - tab.B_tfidf
tab.loc["MEAN"] = tab.mean()
emit("\n== Per-language weighted F1: this run vs dev leader vs TF-IDF diagnostic ==")
emit(tab.round(3).to_string())

Path("results/reports").mkdir(parents=True, exist_ok=True)
(Path("results/reports") / f"error_analysis_{run.parent.name}.txt").write_text("\n".join(lines) + "\n")
