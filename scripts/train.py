#!/usr/bin/env python3
"""E2: unified multi-task QLoRA fine-tuning (one input, both labels in one structured output)."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from afriguard.data import load_split
from afriguard.ftformat import build_user_text, end_token, format_answer, render_prompt
from afriguard.labels import LANGUAGES
from afriguard.utils import set_seed

TARGETS = r"(?!.*(vision|multi_modal)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load_joint(data_dir, langs, split):
    frames = []
    for lang in langs:
        a = load_split(data_dir, "A", lang, split)
        b = load_split(data_dir, "B", lang, split)
        m = a.merge(b[["id", "label"]], on="id", suffixes=("_a", "_b"), validate="one_to_one")
        m["lang"] = lang
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def encode(tok, model_id, row, max_len, end, use_response):
    ai = tok(format_answer(row.label_a, row.label_b) + end, add_special_tokens=False)["input_ids"]
    for clip in (1500, 1000, 600, 300):
        p = render_prompt(tok, model_id, build_user_text(row.prompt, row.response, use_response, clip))
        pi = tok(p, add_special_tokens=False)["input_ids"]
        if len(pi) + len(ai) <= max_len:
            break
    return pi + ai, [-100] * len(pi) + ai


def make_batches(lengths, bs, rng):
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    out = []
    for i in range(0, len(idx), bs * 64):
        c = sorted(idx[i:i + bs * 64], key=lambda j: lengths[j])
        out += [c[k:k + bs] for k in range(0, len(c), bs)]
    rng.shuffle(out)
    return out


def collate(items, pad_id):
    n, width = len(items), max(len(x[0]) for x in items)
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    lab = torch.full((n, width), -100, dtype=torch.long)
    att = torch.zeros((n, width), dtype=torch.long)
    for i, (a, b) in enumerate(items):
        ids[i, :len(a)], lab[i, :len(b)], att[i, :len(a)] = torch.tensor(a), torch.tensor(b), 1
    return ids, att, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--languages", nargs="+", default=LANGUAGES)
    ap.add_argument("--prompt-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-language", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1100)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    set_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cdt = torch.float16 if a.dtype == "float16" else torch.bfloat16
    started = datetime.now(timezone.utc).isoformat()

    df = pd.concat([load_joint(a.data_dir, a.languages, s) for s in a.splits], ignore_index=True)
    if a.per_language:
        df = df.groupby("lang").sample(n=a.per_language, random_state=a.seed).reset_index(drop=True)
    if a.limit:
        df = df.sample(n=min(a.limit, len(df)), random_state=a.seed).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    end = end_token(tok)
    enc = [encode(tok, a.model, r, a.max_len, end, not a.prompt_only) for r in df.itertuples()]
    lengths = [len(x[0]) for x in enc]
    over = sum(n > a.max_len for n in lengths)
    print(f"examples={len(enc)} tokens mean={sum(lengths)/len(lengths):.0f} max={max(lengths)} "
          f"over_max_len={over} end_token={end!r}", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cdt)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, quantization_config=bnb,
                                                 device_map={"": 0}, dtype=cdt)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                             bias="none", task_type="CAUSAL_LM", target_modules=TARGETS))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    per_epoch = math.ceil(len(enc) / a.bs)
    total = a.max_steps if a.max_steps > 0 else max(1, (per_epoch * a.epochs) // a.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total)), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(cdt == torch.float16))

    config = {**vars(a), "started": started, "n_examples": len(enc), "total_updates": total,
              "tokens_mean": sum(lengths) / len(lengths), "tokens_max": max(lengths),
              "end_token": end, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    model.train()
    step = micro = tokens = bad = 0
    run_loss, t0, logf = 0.0, time.time(), open(out / "train_log.jsonl", "w")
    done = False
    for _ in range(a.epochs):
        for bidx in make_batches(lengths, a.bs, rng):
            ids, att, lab = (t.cuda() for t in collate([enc[j] for j in bidx], tok.pad_token_id))
            with torch.autocast("cuda", dtype=cdt):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.grad_accum
            if not torch.isfinite(loss):
                bad += 1
                opt.zero_grad(set_to_none=True)
                if bad > 5:
                    raise SystemExit("Non-finite loss repeatedly; try --dtype bfloat16 or float32 compute.")
                continue
            scaler.scale(loss).backward()
            run_loss += loss.item()
            tokens += int(att.sum())
            micro += 1
            if micro % a.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0 or step == 1:
                    el = time.time() - t0
                    rec = {"step": step, "of": total, "loss": round(run_loss, 4), "lr": sched.get_last_lr()[0],
                           "tok_per_s": round(tokens / el), "elapsed_min": round(el / 60, 1),
                           "eta_min": round(el / step * (total - step) / 60, 1)}
                    print(rec, flush=True)
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                run_loss = 0.0
                if step % a.save_every == 0:
                    model.save_pretrained(out / "adapter")
                if step >= total:
                    done = True
                    break
        if done:
            break
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    config.update(finished=datetime.now(timezone.utc).isoformat(), updates_done=step,
                  train_minutes=round((time.time() - t0) / 60, 1),
                  peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), nonfinite_skipped=bad)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print("DONE", {k: config[k] for k in ("updates_done", "train_minutes", "peak_gpu_gb", "nonfinite_skipped")})


if __name__ == "__main__":
    main()
