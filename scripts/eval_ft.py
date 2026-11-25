#!/usr/bin/env python3
"""Evaluate a fine-tuned unified model (base + LoRA adapter): metrics + official-format submission zip."""
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

    cdt = torch.float16 if a.dtype == "float16" else torch.bfloat16
    out = Path(a.out_dir) if a.out_dir else (Path(a.adapter).parent if a.adapter else Path("experiments/ft")) / f"eval_{a.split}"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.adapter if a.adapter else a.model, revision=None if a.adapter else a.revision)
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

    t_all, metrics, sub = time.time(), {"config": vars(a), "tasks": {"A": {}, "B": {}}}, {}
    for lang in a.languages:
        t0 = time.time()
        A = load_split(a.data_dir, "A", lang, a.split)
        B = load_split(a.data_dir, "B", lang, a.split)
        if a.limit:
            A, B = A.head(a.limit), B.head(a.limit)
        texts = [render_prompt(tok, a.model, build_user_text(r.prompt, r.response, not a.prompt_only, a.max_chars))
                 for r in A.itertuples()]
        raw = generate(model, tok, texts, a.batch_size, a.max_new_tokens)
        parsed = [parse_answer(x) for x in raw]
        pa = [p[0] or UNPARSED for p in parsed]
        pb = [p[1] or UNPARSED for p in parsed]
        pd.DataFrame({"id": A["id"], "label_a": pa, "label_b": pb, "raw": raw}).to_csv(
            out / "predictions" / f"{lang}.csv", index=False)
        sub[lang] = (A["id"].tolist(), [x if x != UNPARSED else FALLBACK["A"] for x in pa],
                     [x if x != UNPARSED else FALLBACK["B"] for x in pb])
        sec = round(time.time() - t0, 1)
        line = f"{lang:8s} n={len(A)} {sec}s unparsed A={pa.count(UNPARSED)/len(pa):.1%} B={pb.count(UNPARSED)/len(pb):.1%}"
        if "label" in A.columns:
            for t, preds, gold, labels in (("A", pa, A["label"], TASK_A_LABELS), ("B", pb, B["label"], TASK_B_LABELS)):
                m = full_metrics(gold, preds, labels)
                m.update(n=len(A), seconds=sec, unparsed_rate=preds.count(UNPARSED) / len(preds))
                metrics["tasks"][t][lang] = m
            line += f" wF1 A={metrics['tasks']['A'][lang]['weighted_f1']:.3f} B={metrics['tasks']['B'][lang]['weighted_f1']:.3f}"
        print(line, flush=True)
    for t in ("A", "B"):
        s = [v["weighted_f1"] for v in metrics["tasks"][t].values()]
        metrics["tasks"][t]["_avg_weighted_f1"] = sum(s) / len(s) if s else None
    print("AVG weighted F1: A=%s B=%s" % (metrics["tasks"]["A"].get("_avg_weighted_f1"), metrics["tasks"]["B"].get("_avg_weighted_f1")))
    metrics["total_seconds"] = round(time.time() - t_all, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for lang, (ids, la, lb) in sub.items():
            for d, labs in (("task_a_safety", la), ("task_b_category", lb)):
                buf = io.StringIO()
                buf.write("id,label\n")
                for i, lab in zip(ids, labs):
                    buf.write(f'{i},"{lab}"\n')
                z.writestr(f"{d}/{lang}.csv", buf.getvalue())
    print("Wrote", out)


if __name__ == "__main__":
    main()
