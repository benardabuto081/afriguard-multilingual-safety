#!/usr/bin/env python3
"""Numeric spot check of a merged checkpoint against saved adapter-model label log-probs (works on CPU)."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from afriguard.data import load_split
from afriguard.templates import ALL, answers, encode_pair, render_prefix

spec = importlib.util.spec_from_file_location("eval_lbl", Path(__file__).parent / "eval_lbl.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--run", required=True, help="eval_lbl directory holding lp_<template>.npz from the adapter model")
ap.add_argument("--template", default="t1")
ap.add_argument("--family", default="aya")
ap.add_argument("--languages", nargs="+", default=["swahili"])
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
dt = torch.float32 if a.device == "cpu" else torch.float16
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, device_map={"": 0} if a.device == "cuda" else None).eval()
z, tpl = np.load(Path(a.run) / f"lp_{a.template}.npz"), ALL[a.template]
agree, diffs = [], []
for lang in a.languages:
    for task in ("A", "B"):
        df = load_split("data/raw", task, lang, "dev").head(a.n)
        assert list(z[f"{lang}_{task}_ids"][:a.n]) == df["id"].tolist()
        ref = z[f"{lang}_{task}"][:a.n]
        prefixes = [render_prefix(tok, a.family, tpl, task, p) for p in df["prompt"]]
        pre = [tok(x, add_special_tokens=False)["input_ids"] for x in prefixes]
        cand = [encode_pair(tok, prefixes[0], s)[1] for s in answers(tpl, task)]
        sums, _ = ev.score_candidates(model, tok.pad_token_id, pre, cand, 4)
        agree += list(sums.argmax(1) == ref.argmax(1))
        diffs += list(np.abs(sums - ref).ravel())
        print(lang, task, "argmax agreement", float(np.mean(sums.argmax(1) == ref.argmax(1))),
              "median |diff|", round(float(np.median(np.abs(sums - ref))), 4), flush=True)
print("OVERALL argmax agreement", round(float(np.mean(agree)), 3), "| median |diff|", round(float(np.median(diffs)), 4),
      "| max |diff|", round(float(np.max(diffs)), 3))
print("VERDICT:", "OK" if np.mean(agree) >= 0.9 and np.median(diffs) < 0.3 else "INVESTIGATE")
