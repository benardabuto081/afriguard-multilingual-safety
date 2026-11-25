#!/usr/bin/env python3
"""Tokens per language for candidate tokenizers (fine-tuning input text, first 200 train rows each)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from afriguard.data import load_split
from afriguard.ftformat import build_user_text
from afriguard.labels import LANGUAGES

models = sys.argv[1:] or ["Qwen/Qwen3-4B", "Qwen/Qwen2.5-3B-Instruct", "google/gemma-3-4b-it"]
rows = []
for m in models:
    tok = AutoTokenizer.from_pretrained(m)
    for lang in LANGUAGES:
        df = load_split("data/raw", "A", lang, "train").head(200)
        lens = [len(tok(build_user_text(r.prompt, r.response), add_special_tokens=False)["input_ids"])
                for r in df.itertuples()]
        rows.append({"model": m, "language": lang, "mean": float(np.mean(lens)),
                     "p95": float(np.percentile(lens, 95)), "max": int(max(lens))})
df = pd.DataFrame(rows)
Path("results/tables").mkdir(parents=True, exist_ok=True)
df.to_csv("results/tables/token_stats.csv", index=False)
print(df.pivot(index="language", columns="model", values="mean").round(0).to_string())
print("\np95 overall:", df.groupby("model")["p95"].max().to_dict())
print("max overall:", df.groupby("model")["max"].max().to_dict())
