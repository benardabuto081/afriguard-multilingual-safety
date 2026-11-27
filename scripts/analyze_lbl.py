#!/usr/bin/env python3
"""Analysis of a label-log-likelihood evaluation run (needs lp_*.npz + metrics.json). No prompt text printed."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from afriguard.data import load_split
from afriguard.labels import LANGUAGES, TASK_A_LABELS, TASK_B_LABELS

SHORT = ["Crim", "Poli", "Viol", "Syst", "Sexu", "Advi", "Priv", "Hate", "Ethi", "IP"]
LABELS = {"A": TASK_A_LABELS, "B": TASK_B_LABELS}
run = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/multitask/e2p_full/eval_lbl")
lines = []


def emit(s=""):
    lines.append(str(s))
    print(s)


def softmax(s):
    z = s - s.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


m = json.loads((run / "metrics.json").read_text())
tn = list(m["templates"])
held = [t for t in tn if t.startswith("h")] or tn
hi = [tn.index(t) for t in held]
pd.set_option("display.width", 200)

# 1. per-language vs prompt-only TF-IDF reference
ref = pd.read_csv("results/tables/reference_baseline.csv")
ref = ref[ref["input"] == "prompt"].pivot(index="language", columns="task", values="dev_f1")
tab = pd.DataFrame({lang: {
    "A": np.mean([m["templates"][t]["A"][lang] for t in held]), "A_tfidf": ref.loc[lang, "A"],
    "B": np.mean([m["templates"][t]["B"][lang] for t in held]), "B_tfidf": ref.loc[lang, "B"]}
    for lang in LANGUAGES}).T
tab["A_gain"], tab["B_gain"] = tab.A - tab.A_tfidf, tab.B - tab.B_tfidf
tab.loc["MEAN"] = tab.mean()
emit(f"run: {run}   held-out templates: {held}   all templates: {tn}")
emit("\n== Per-language weighted F1 (mean over held-out templates) vs prompt-only TF-IDF ==")
emit(tab.round(3).to_string())

# 2. load log-probs and gold
arrs = {t: np.load(run / f"lp_{t}.npz") for t in tn}
data, cat = {"A": {}, "B": {}}, {}
for lang in LANGUAGES:
    for task in ("A", "B"):
        ids = arrs[tn[0]][f"{lang}_{task}_ids"]
        S = np.stack([arrs[t][f"{lang}_{task}"] for t in tn])
        assert all((arrs[t][f"{lang}_{task}_ids"] == ids).all() for t in tn)
        df = load_split("data/raw", task, lang, "dev").set_index("id").loc[ids]
        data[task][lang] = (ids, S, np.array([LABELS[task].index(x) for x in df["label"]]))
    cat[lang] = load_split("data/raw", "B", lang, "dev").set_index("id").loc[data["A"][lang][0], "label"].values

# 3. Task A errors by gold category, and confident disagreements (label-noise candidates)
rows = []
for lang in LANGUAGES:
    _, S, y = data["A"][lang]
    P = softmax(S[hi]).mean(0)
    pred = P.argmax(1)
    rows.append(pd.DataFrame({"cat": cat[lang], "gold": y, "err": pred != y, "conf": P.max(1), "lang": lang}))
A = pd.concat(rows)
g = A.groupby(["cat", "gold"])["err"].mean().unstack().reindex(TASK_B_LABELS)
g.index, g.columns = SHORT, ["safe_called_unsafe", "unsafe_called_safe"]
emit("\n== Task A error rate by gold category (probabilities averaged over held-out templates) ==")
emit(g.round(3).to_string())
ce = pd.DataFrame({"errors": A[A.err].groupby("lang").size(),
                   "confident_errors_p>=0.9": A[A.err & (A.conf >= 0.9)].groupby("lang").size()}
                  ).reindex(LANGUAGES).fillna(0).astype(int)
ce.loc["TOTAL"] = ce.sum()
emit("\n== Task A: errors and confident errors per language (label-noise candidates; 120 items each) ==")
emit(ce.to_string())

# 4. Task B pooled per-category and confusion
Y, PR = [], []
for lang in LANGUAGES:
    _, S, y = data["B"][lang]
    Y.append(y)
    PR.append(softmax(S[hi]).mean(0).argmax(1))
Y, PR = np.concatenate(Y), np.concatenate(PR)
p, r, f, s = precision_recall_fscore_support(Y, PR, labels=list(range(10)), zero_division=0)
emit("\n== Task B pooled per-category ==")
emit(pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=SHORT).round(3).to_string())
cm = pd.crosstab(Y, PR).reindex(index=range(10), columns=range(10), fill_value=0)
cm.index, cm.columns = SHORT, SHORT
emit("\n== Task B confusion (rows gold, columns predicted) ==")
emit(cm.to_string())
top = sorted(((int(cm.iloc[i, j]), SHORT[i], SHORT[j]) for i in range(10) for j in range(10) if i != j), reverse=True)[:8]
emit("top confusions: " + "; ".join(f"{c} {a}->{b}" for c, a, b in top))


# 5. calibration upper bound: class-bias correction fitted with 5-fold CV on dev (not deployable by us)
def fit_bias(S, y, K, l2=1e-2):
    oh = np.eye(K)[y]

    def fg(b):
        z = S + b
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        p_ = e / e.sum(1, keepdims=True)
        return -np.log(p_[np.arange(len(y)), y] + 1e-12).mean() + l2 * (b ** 2).sum(), (p_ - oh).mean(0) + 2 * l2 * b
    return minimize(fg, np.zeros(K), jac=True, method="L-BFGS-B").x


def avg_wf1(y, pred, lg):
    return np.mean([f1_score(y[lg == lang], pred[lg == lang], average="weighted", zero_division=0) for lang in LANGUAGES])


emit("\n== Calibration upper bound (class-bias, 5-fold CV on dev; the organizers' harness would not apply it) ==")
res = []
for task in ("A", "B"):
    K = len(LABELS[task])
    y = np.concatenate([data[task][lang][2] for lang in LANGUAGES])
    lg = np.array([lang for lang in LANGUAGES for _ in range(len(data[task][lang][2]))])
    for ti, t in enumerate(tn):
        S = np.concatenate([data[task][lang][1][ti] for lang in LANGUAGES])
        base, cal = S.argmax(1), np.zeros(len(y), dtype=int)
        for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=0).split(S, y):
            cal[b_] = (S[b_] + fit_bias(S[a_], y[a_], K)).argmax(1)
        res.append({"task": task, "template": t, "argmax": avg_wf1(y, base, lg), "bias_corrected": avg_wf1(y, cal, lg)})
cr = pd.DataFrame(res)
cr["gain"] = cr.bias_corrected - cr.argmax
emit(cr.round(4).to_string(index=False))
emit("mean gain: " + str(cr.groupby("task")["gain"].mean().round(4).to_dict()))

Path("results/reports").mkdir(parents=True, exist_ok=True)
(Path("results/reports") / f"analysis_lbl_{run.parent.name}.txt").write_text("\n".join(lines) + "\n")
