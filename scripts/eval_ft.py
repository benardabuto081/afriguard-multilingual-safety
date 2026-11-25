#!/usr/bin/env python3
"""Evaluate a fine-tuned model (base + LoRA adapter): metrics + official-format submission zip."""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from afriguard.data import load_split
from afriguard.evaluate import full_metrics
from afriguard.ftformat import build_user_text, parse_answer, render_prompt
from afriguard.labels import LANGUAGES, TASK_A_LABELS, TASK_B_LABELS, UNPARSED

FALLBACK = {"A": "unsafe", "B": "Criminal Activity & Exploitation"}
LABELS = {"A": TASK_A_LABELS, "B": TASK_B_LABELS}
DIRS = {"A": "task_a_safety", "B": "task_b_category"}


def generate(model, tok, texts, bs, max_new):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    res, i = [None] * len(texts), 0
    while i < len(order):
        idx = order[i:i + bs]
        try:
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            dec = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            continue
        for j, d in zip(idx, dec):
            res[j] = d
        i += len(idx)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--task-mode", default="joint", choices=["joint", "A", "B"])
    ap.add_argument("--template", default="default", choices=["default", "compact"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--languages", nargs="+", default=LANGUAGES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--prompt-only", action="store_true")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    tasks = ("A", "B") if a.task_mode == "joint" else (a.task_mode,)
    cdt = torch.float16 if a.dtype == "float16" else torch.bfloat16
    out = Path(a.out_dir) if a.out_dir else (Path(a.adapter).parent if a.adapter else Path("experiments/ft")) / f"eval_{a.split}"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    has_tok = a.adapter and (Path(a.adapter) / "tokenizer_config.json").exists()
    tok = AutoTokenizer.from_pretrained(a.adapter if has_tok else a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {} if a.no_4bit else {"quantization_config": BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cdt)}
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, device_map={"": 0}, dtype=cdt, **kw)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    t_all, metrics, sub = time.time(), {"config": vars(a), "tasks": {t: {} for t in tasks}}, {}
    for lang in a.languages:
        t0 = time.time()
        A = load_split(a.data_dir, "A", lang, a.split)
        B = load_split(a.data_dir, "B", lang, a.split)
        B = B.set_index("id").loc[A["id"]].reset_index()  # align Task B rows to Task A order by id
        if a.limit:
            A, B = A.head(a.limit), B.head(a.limit)
        texts = [render_prompt(tok, a.model, build_user_text(r.prompt, r.response, not a.prompt_only,
                                                             a.max_chars, a.task_mode), a.template) for r in A.itertuples()]
        raw = generate(model, tok, texts, a.batch_size, a.max_new_tokens)
        parsed = [parse_answer(x) for x in raw]
        preds = {"A": [p[0] or UNPARSED for p in parsed], "B": [p[1] or UNPARSED for p in parsed]}
        pd.DataFrame({"id": A["id"], "label_a": preds["A"], "label_b": preds["B"], "raw": raw}).to_csv(
            out / "predictions" / f"{lang}.csv", index=False)
        sub[lang] = (A["id"].tolist(), {t: [x if x != UNPARSED else FALLBACK[t] for x in preds[t]] for t in tasks})
        sec = round(time.time() - t0, 1)
        line = f"{lang:8s} n={len(A)} {sec}s " + " ".join(
            f"unparsed {t}={preds[t].count(UNPARSED)/len(A):.1%}" for t in tasks)
        if "label" in A.columns:
            gold = {"A": A["label"], "B": B["label"]}
            for t in tasks:
                m = full_metrics(gold[t], preds[t], LABELS[t])
                m.update(n=len(A), seconds=sec, unparsed_rate=preds[t].count(UNPARSED) / len(A))
                metrics["tasks"][t][lang] = m
                line += f" wF1 {t}={m['weighted_f1']:.3f}"
        print(line, flush=True)
    for t in tasks:
        s = [v["weighted_f1"] for k, v in metrics["tasks"][t].items() if not k.startswith("_")]
        metrics["tasks"][t]["_avg_weighted_f1"] = sum(s) / len(s) if s else None
    print("AVG weighted F1:", {t: metrics["tasks"][t]["_avg_weighted_f1"] for t in tasks})
    metrics["total_seconds"] = round(time.time() - t_all, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for lang, (ids, labs) in sub.items():
            for t in tasks:
                buf = io.StringIO()
                buf.write("id,label\n")
                for i, lab in zip(ids, labs[t]):
                    buf.write(f'{i},"{lab}"\n')
                z.writestr(f"{DIRS[t]}/{lang}.csv", buf.getvalue())
    print("Wrote", out)


if __name__ == "__main__":
    main()
