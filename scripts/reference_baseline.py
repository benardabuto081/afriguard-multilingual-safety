#!/usr/bin/env python3
"""CPU diagnostic: char n-gram TF-IDF + linear SVM per language/task. Not a submission model."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

LANGS = ["amharic", "hausa", "igbo", "oromo", "shona", "swahili", "twi", "wolof", "yoruba", "zulu"]
TASKS = {"A": "task_a_safety", "B": "task_b_category"}
MODES = ["prompt", "response", "both", "majority"]


def feats(df, mode):
    if mode == "prompt":
        return df["prompt"]
    if mode == "response":
        return df["response"]
    return df["prompt"] + " [SEP] " + df["response"]


def mk():
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2,
                        max_features=300000),
        LinearSVC(C=0.5))


def wf1(y, p):
    return float(f1_score(y, p, average="weighted", zero_division=0))


def run(task, lang, mode, root, folds, seed):
    d = Path(root) / TASKS[task] / lang
    tr = pd.read_csv(d / "train.csv", dtype=str, keep_default_na=False)
    dv = pd.read_csv(d / "dev.csv", dtype=str, keep_default_na=False)
    if mode == "majority":
        maj = tr["label"].mode()[0]
        return dict(task=task, language=lang, input=mode, cv_f1=np.nan,
                    dev_f1=wf1(dv["label"], [maj] * len(dv)))
    x, y = feats(tr, mode), tr["label"].values
    oof = np.empty(len(tr), dtype=object)
    for a, b in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y):
        oof[b] = mk().fit(x.iloc[a], y[a]).predict(x.iloc[b])
    pred = mk().fit(x, y).predict(feats(dv, mode))
    return dict(task=task, language=lang, input=mode, cv_f1=wf1(y, oof), dev_f1=wf1(dv["label"], pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="results/tables/reference_baseline.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    rows = Parallel(n_jobs=a.jobs)(
        delayed(run)(t, l, m, a.data_dir, a.folds, a.seed) for t in TASKS for l in LANGS for m in MODES)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    for t in TASKS:
        pv = df[df.task == t].pivot_table(index="language", columns="input", values=["cv_f1", "dev_f1"])
        pv.loc["MEAN"] = pv.mean()
        print(f"\n=== Task {t}: weighted F1 (cv = {a.folds}-fold on train, dev = train->dev) ===")
        print(pv.round(3).to_string())


if __name__ == "__main__":
    main()
