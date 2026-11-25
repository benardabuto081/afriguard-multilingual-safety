#!/usr/bin/env python3
"""Teacher-forced label scoring for a fine-tuned joint model (for calibration, E5).
Per item: logits over the Task A labels, and over the Task B categories given each possible Task A label,
read at the first answer token of each candidate. No generation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from afriguard.data import load_split
from afriguard.evaluate import weighted_f1
from afriguard.ftformat import build_user_text, render_prompt
from afriguard.labels import LANGUAGES, TASK_A_LABELS, TASK_B_LABELS


def first_tokens(tok, prefix, names):
    base = tok(prefix, add_special_tokens=False)["input_ids"]
    out = []
    for n in names:
        full = tok(prefix + " " + n, add_special_tokens=False)["input_ids"]
        if full[:len(base)] != base or len(full) <= len(base):
            raise SystemExit(f"{prefix!r} is not a token prefix of {prefix + ' ' + n!r}")
        out.append(full[len(base)])
    if len(set(out)) != len(out):
        raise SystemExit(f"first-token collision for prefix {prefix!r}: {out}")
    return out


@torch.no_grad()
def next_logits(model, tok, seqs, cand, bs):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = np.zeros((len(seqs), len(cand)), dtype=np.float32)
    i = 0
    while i < len(order):
        idx = order[i:i + bs]
        width = max(len(seqs[j]) for j in idx)
        ids = torch.full((len(idx), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), width), dtype=torch.long)
        for r, j in enumerate(idx):
            s = seqs[j]
            ids[r, width - len(s):] = torch.tensor(s)
            att[r, width - len(s):] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        kw = {"input_ids": ids.to(model.device), "attention_mask": att.to(model.device),
              "position_ids": pos.to(model.device)}
        try:
            res = model(**kw, logits_to_keep=1)
        except TypeError:
            res = model(**kw)
        out[idx] = res.logits[:, -1, :].float()[:, cand].cpu().numpy()
        i += len(idx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--template", default="default", choices=["default", "compact"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--languages", nargs="+", default=LANGUAGES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--prompt-only", action="store_true")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--compare-dir", default=None, help="eval_dev dir with generation predictions to compare")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    cdt = torch.float16 if a.dtype == "float16" else torch.bfloat16
    out = Path(a.out_dir) if a.out_dir else (Path(a.adapter).parent if a.adapter else Path("experiments/ft")) / f"score_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    has_tok = a.adapter and (Path(a.adapter) / "tokenizer_config.json").exists()
    tok = AutoTokenizer.from_pretrained(a.adapter if has_tok else a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=cdt)
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map={"": 0}, dtype=cdt, quantization_config=bnb)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    a_first = first_tokens(tok, "safety:", TASK_A_LABELS)
    b_first = {x: first_tokens(tok, f"safety: {x}\ncategory:", TASK_B_LABELS) for x in TASK_A_LABELS}
    suf_a = tok("safety:", add_special_tokens=False)["input_ids"]
    suf_b = {x: tok(f"safety: {x}\ncategory:", add_special_tokens=False)["input_ids"] for x in TASK_A_LABELS}

    metrics, t_all = {"config": vars(a), "languages": {}}, time.time()
    for lang in a.languages:
        t0 = time.time()
        A = load_split(a.data_dir, "A", lang, a.split)
        B = load_split(a.data_dir, "B", lang, a.split)
        B = B.set_index("id").loc[A["id"]].reset_index()
        if a.limit:
            A, B = A.head(a.limit), B.head(a.limit)
        texts = [render_prompt(tok, a.model, build_user_text(r.prompt, r.response, not a.prompt_only,
                                                             a.max_chars, "joint"), a.template) for r in A.itertuples()]
        p_ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        la = next_logits(model, tok, [p + suf_a for p in p_ids], a_first, a.batch_size)
        lb = {x: next_logits(model, tok, [p + suf_b[x] for p in p_ids], b_first[x], a.batch_size)
              for x in TASK_A_LABELS}
        a_idx = la.argmax(1)
        pred_a = [TASK_A_LABELS[i] for i in a_idx]
        lb_sel = np.where(a_idx[:, None] == 0, lb["safe"], lb["unsafe"])
        pred_b = [TASK_B_LABELS[i] for i in lb_sel.argmax(1)]
        np.savez(out / f"{lang}.npz", ids=A["id"].values.astype(str), A=la, B_safe=lb["safe"], B_unsafe=lb["unsafe"])
        rec = {"n": len(A), "seconds": round(time.time() - t0, 1)}
        if "label" in A.columns:
            rec["wf1_A"], rec["wf1_B"] = weighted_f1(A["label"], pred_a), weighted_f1(B["label"], pred_b)
        if a.compare_dir:
            cp = Path(a.compare_dir) / "predictions" / f"{lang}.csv"
            if cp.exists():
                g = pd.read_csv(cp, dtype=str, keep_default_na=False).set_index("id").loc[A["id"]]
                rec["agree_A"] = float((g["label_a"].values == np.array(pred_a)).mean())
                rec["agree_B"] = float((g["label_b"].values == np.array(pred_b)).mean())
        metrics["languages"][lang] = rec
        print(lang, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    for k in ("wf1_A", "wf1_B", "agree_A", "agree_B"):
        v = [r[k] for r in metrics["languages"].values() if k in r]
        if v:
            metrics[f"avg_{k}"] = sum(v) / len(v)
    print("AVG", {k: round(v, 4) for k, v in metrics.items() if k.startswith("avg_")})
    metrics["total_seconds"] = round(time.time() - t_all, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
