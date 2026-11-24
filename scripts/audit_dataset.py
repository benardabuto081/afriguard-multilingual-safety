#!/usr/bin/env python3
"""Audit for the task/language/split.csv layout. Emits statistics only, never prompt text."""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

EXPECTED = ["id", "language", "prompt", "response", "label"]
SPLITS = ("train", "dev", "test")


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(out)


def wstats(s: pd.Series) -> dict:
    w = s.str.split().str.len()
    if not len(w):
        return {}
    return {"mean": round(float(w.mean()), 1), "median": float(w.median()), "max": int(w.max())}


def file_stats(df: pd.DataFrame, split: str) -> dict:
    cols = list(df.columns)
    need = [c for c in EXPECTED if not (c == "label" and split == "test")]
    r = {"rows": len(df), "columns": cols,
         "missing_columns": [c for c in need if c not in cols],
         "extra_columns": [c for c in cols if c not in EXPECTED],
         "blank_by_column": {c: int((df[c].str.strip() == "").sum()) for c in cols}}
    if "id" in cols:
        r["duplicate_ids"] = int(df["id"].duplicated().sum())
        r["id_patterns"] = df["id"].str.replace(r"\d+", "#", regex=True).value_counts().head(3).to_dict()
    if "language" in cols:
        r["language_values"] = df["language"].value_counts().to_dict()
    if "prompt" in cols:
        pn = df["prompt"].map(norm)
        r["duplicate_prompts_normalised"] = int(pn.duplicated().sum())
        r["prompt_words"] = wstats(df["prompt"])
        if "response" in cols:
            r["duplicate_prompt_response_pairs"] = int(
                (pn + "||" + df["response"].map(norm)).duplicated().sum())
    if "response" in cols:
        r["response_words"] = wstats(df["response"])
    if "label" in cols:
        r["label_distribution"] = df["label"].value_counts().to_dict()
    return r


def neighbour_analysis(d: dict[str, pd.DataFrame]) -> dict:
    tr = d.get("train")
    if tr is None or "label" not in tr:
        return {}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    xtr = vec.fit_transform(tr["prompt"].map(norm))
    out: dict = {}
    for sp in ("dev", "test"):
        if sp in d:
            x = vec.transform(d[sp]["prompt"].map(norm))
            mx = np.asarray((x @ xtr.T).max(axis=1).todense()).ravel()
            out[f"{sp}_to_train"] = {"median_max_cos": round(float(np.median(mx)), 3),
                                     "share_ge_0.8": round(float((mx >= 0.8).mean()), 3),
                                     "share_ge_0.9": round(float((mx >= 0.9).mean()), 3)}
    s = (xtr @ xtr.T).toarray()
    np.fill_diagonal(s, 0)
    mx = s.max(axis=1)
    out["train_internal"] = {"median_nn_cos": round(float(np.median(mx)), 3),
                             "share_ge_0.8": round(float((mx >= 0.8).mean()), 3)}
    lab = tr["label"].str.lower().values
    for thr in (0.5, 0.7, 0.85):
        n, comp = connected_components(csr_matrix(s >= thr), directed=False)
        sizes = np.bincount(comp)
        mixed = sum(1 for c in range(n) if sizes[c] > 1 and len(set(lab[comp == c])) > 1)
        out[f"clusters_cos_{thr}"] = {"n_clusters": int(n), "singletons": int((sizes == 1).sum()),
                                      "max_size": int(sizes.max()), "clusters_with_both_labels": mixed}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--output-dir", default="results/reports")
    ap.add_argument("--tables-dir", default="results/tables")
    ap.add_argument("--docs-dir", default="docs")
    a = ap.parse_args()

    root = Path(a.data_dir)
    frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    for p in sorted(root.rglob("*.csv")):
        if len(p.relative_to(root).parts) >= 3:
            frames[(p.parent.parent.name, p.parent.name, p.stem)] = pd.read_csv(
                p, dtype=str, keep_default_na=False)
    if not frames:
        raise SystemExit(f"No task/language/split.csv files found under {root}")
    tasks = sorted({k[0] for k in frames})
    combined_check = {}
    for (t, lang, sp), df in list(frames.items()):
        if lang == "multilingual":
            if sp in ("train", "dev"):
                parts = [frames[k] for k in frames if k[0] == t and k[2] == sp and k[1] != "multilingual"]
                key = ["id"] + [c for c in ("prompt", "response", "label") if c in df.columns]
                u = pd.concat(parts, ignore_index=True)[key].sort_values("id").reset_index(drop=True)
                m = df[key].sort_values("id").reset_index(drop=True)
                combined_check[f"{t}/{sp}"] = bool(u.equals(m))
            del frames[(t, lang, sp)]
    langs = sorted({k[1] for k in frames})

    binary = None
    for t in tasks:
        vals: set = set()
        for (tt, _, sp), df in frames.items():
            if tt == t and sp in ("train", "dev") and "label" in df:
                vals |= set(df["label"].str.lower().unique())
        if vals and vals <= {"safe", "unsafe"}:
            binary = t
    other = [t for t in tasks if t != binary]

    rep: dict = {"tasks": tasks, "languages": langs, "binary_task_detected": binary, "files": {}}
    for (t, lang, sp), df in frames.items():
        rep["files"][f"{t}/{lang}/{sp}"] = file_stats(df, sp)

    tdir = Path(a.tables_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    md = ["# Dataset audit", "", "Generated by `scripts/audit_dataset.py`. Statistics only; no prompt text.", "",
          f"- tasks: {tasks}", f"- languages: {langs}", f"- binary task detected: {binary}",
          f"- multilingual folder identical to union of language folders: {combined_check}", ""]

    # sizes
    base = binary or tasks[0]
    sizes = pd.DataFrame({sp: {lang: len(frames[(base, lang, sp)]) for lang in langs
                               if (base, lang, sp) in frames} for sp in sorted({k[2] for k in frames})})
    sizes.to_csv(tdir / "language_distribution.csv")
    md += ["## Rows per language and split (" + base + ")", "", md_table(sizes), ""]

    # labels per task
    rep["label_tables"] = {}
    for t in tasks:
        rows = []
        for (tt, lang, sp), df in frames.items():
            if tt == t and sp in ("train", "dev") and "label" in df:
                rows.append(df.assign(_lang=lang, _split=sp)[["_lang", "_split", "label"]])
        if rows:
            all_ = pd.concat(rows)
            ct = pd.crosstab(all_["_lang"], all_["label"])
            ct.to_csv(tdir / f"class_distribution_{t}.csv")
            tot = all_["label"].value_counts()
            rep["label_tables"][t] = {"total": tot.to_dict(), "by_language": ct.to_dict("index")}
            md += [f"## Labels, {t} (train+dev)", "", md_table(tot.to_frame("count")), "", md_table(ct), ""]

    # cross-task consistency and label relationship
    cons, pairs = {}, []
    if binary and other:
        for o in other:
            for lang in langs:
                for sp in SPLITS:
                    ka, kb = (binary, lang, sp), (o, lang, sp)
                    if ka in frames and kb in frames:
                        fa = frames[ka].set_index("id").sort_index()
                        fb = frames[kb].set_index("id").sort_index()
                        same_ids = list(fa.index) == list(fb.index)
                        same_txt = same_ids and fa[["prompt", "response"]].equals(fb[["prompt", "response"]])
                        cons[f"{lang}/{sp}"] = {"same_ids": same_ids, "same_prompt_response": bool(same_txt)}
                        if sp != "test" and same_ids:
                            pairs.append(pd.DataFrame({"A": fa["label"].str.lower(), "B": fb["label"]}))
        rep["cross_task_consistency"] = cons
        bad = {k: v for k, v in cons.items() if not (v["same_ids"] and v["same_prompt_response"])}
        md += ["## Cross-task consistency", "", f"- checked: {len(cons)}; inconsistent: {len(bad)}"]
        if bad:
            md += [f"- {k}: {v}" for k, v in list(bad.items())[:20]]
        if pairs:
            cat = pd.concat(pairs, ignore_index=True)
            rel = pd.crosstab(cat["A"], cat["B"])
            rel.to_csv(tdir / "task_a_vs_task_b_labels.csv")
            rep["task_a_vs_b"] = rel.to_dict("index")
            md += ["", "### Task A label vs Task B label (train+dev, all languages)", "", md_table(rel), ""]

    # split integrity + neighbours on the binary task
    rep["split_integrity"], rep["neighbours"] = {}, {}
    nb_rows, ov_rows = {}, []
    if binary:
        for lang in langs:
            d = {sp: frames[(binary, lang, sp)] for sp in SPLITS if (binary, lang, sp) in frames}
            for x, y in combinations(d, 2):
                ov = {"id": len(set(d[x]["id"]) & set(d[y]["id"])),
                      "prompt": len(set(d[x]["prompt"].map(norm)) & set(d[y]["prompt"].map(norm))),
                      "response": len(set(d[x]["response"].map(norm)) & set(d[y]["response"].map(norm)))}
                rep["split_integrity"][f"{lang}/{x}&{y}"] = ov
                ov_rows.append({"language": lang, "pair": f"{x}&{y}", **ov})
            nb = neighbour_analysis(d)
            rep["neighbours"][lang] = nb
            nb_rows[lang] = pd.json_normalize(nb, sep=".").iloc[0].to_dict() if nb else {}
        ov_df = pd.DataFrame(ov_rows)
        ov_df.to_csv(tdir / "split_overlap.csv", index=False)
        nb_df = pd.DataFrame(nb_rows).T
        nb_df.to_csv(tdir / "neighbour_analysis.csv")
        md += ["## Split overlap (counts of shared ids / normalised prompts / responses)", "",
               md_table(ov_df.set_index("language")), "",
               "## Prompt similarity (char n-gram TF-IDF cosine) and estimated scenario clusters", "",
               md_table(nb_df), ""]

    # sample submission check
    sub = {}
    for (t, lang, sp), df in frames.items():
        if sp == "sample_submission" and (t, lang, "test") in frames:
            sub[f"{t}/{lang}"] = {"columns": list(df.columns),
                                  "ids_match_test": set(df["id"]) == set(frames[(t, lang, "test")]["id"]),
                                  "rows": len(df)}
    rep["sample_submission"] = sub
    bad = {k: v for k, v in sub.items() if not v["ids_match_test"]}
    md += ["## Sample submissions", "", f"- checked: {len(sub)}; ids not matching test: {len(bad)}", ""]

    # per-file quality summary
    q = []
    for k, v in rep["files"].items():
        if k.split("/")[-1] in SPLITS:
            q.append({"file": k, "rows": v["rows"], "missing_cols": v["missing_columns"],
                      "extra_cols": v["extra_columns"], "dup_ids": v.get("duplicate_ids"),
                      "dup_prompts": v.get("duplicate_prompts_normalised"),
                      "dup_pairs": v.get("duplicate_prompt_response_pairs"),
                      "blank_prompt": v["blank_by_column"].get("prompt"),
                      "blank_response": v["blank_by_column"].get("response"),
                      "blank_label": v["blank_by_column"].get("label"),
                      "lang_values": v.get("language_values"),
                      "prompt_words_mean": v.get("prompt_words", {}).get("mean"),
                      "response_words_mean": v.get("response_words", {}).get("mean")})
    qdf = pd.DataFrame(q).set_index("file")
    qdf.to_csv(tdir / "data_quality.csv")
    md += ["## Per-file data quality", "", md_table(qdf), ""]

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_audit.json").write_text(json.dumps(rep, indent=2, default=str, ensure_ascii=False))
    Path(a.docs_dir).mkdir(exist_ok=True)
    (Path(a.docs_dir) / "dataset_audit.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
