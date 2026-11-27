#!/usr/bin/env python3
"""Prompt-only QLoRA training with a template bank; one sequence per (example, task), label-only answers.
Resume-safe: checkpoints adapter + optimizer state; rerun with --resume."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from afriguard.data import load_joint
from afriguard.ftformat import end_token
from afriguard.labels import LANGUAGES
from afriguard.sampling import build_rules, repeat_count
from afriguard.templates import BANKS, answers, build_train_item, family_of, render_prefix
from afriguard.utils import set_seed

TARGETS = r"(?!.*(vision|multi_modal)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def build_items(df, bank, tasks, seed, compact_p, tok, family, max_len, end_chat, end_raw, rules=None):
    rng = random.Random(seed)
    items = []
    for r in df.itertuples():
        reps = 1 if not rules else repeat_count(rules, r, rng)
        for task in list(tasks) * reps:
            tpl = bank[rng.randrange(len(bank))]
            u = rng.random()
            render = "compact" if (family == "aya" and u < compact_p) else "native"
            label = r.label_a if task == "A" else r.label_b
            end = end_chat if tpl.style == "chat" else end_raw
            items.append(build_train_item(tok, family, tpl, task, r.prompt, label, render, max_len, end))
    return items


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


def save_ckpt(model, opt, sched, scaler, out, st):
    model.save_pretrained(out / "adapter")
    tmp = out / "state.pt.tmp"
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), **st}, tmp)
    os.replace(tmp, out / "state.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bank", default="train", choices=sorted(BANKS))
    ap.add_argument("--tasks", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--languages", nargs="+", default=LANGUAGES)
    ap.add_argument("--per-language", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--compact-p", type=float, default=0.8)
    ap.add_argument("--upsample-cells", default="", help="e.g. 8:safe=2,9:unsafe=2 (category number 1-10 : Task A label = factor)")
    ap.add_argument("--upsample-langs", default="", help="e.g. oromo=2,twi=2")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    set_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cdt = torch.float16 if a.dtype == "float16" else torch.bfloat16
    family, bank = family_of(a.model), BANKS[a.bank]
    rules = build_rules(a.upsample_cells, a.upsample_langs)
    started = datetime.now(timezone.utc).isoformat()

    import pandas as pd
    df = pd.concat([load_joint(a.data_dir, a.languages, s) for s in a.splits], ignore_index=True)
    if a.per_language:
        df = df.groupby("lang").sample(n=a.per_language, random_state=a.seed).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    end_chat, end_raw = end_token(tok), (tok.eos_token or end_token(tok))

    print("TEMPLATE SKELETONS (what the model sees; <PROMPT> is a placeholder):", flush=True)
    for tpl in bank:
        for task in a.tasks:
            print(f"  [{tpl.name}/{task}] {render_prefix(tok, family, tpl, task, '<PROMPT>', 'native')!r}"
                  f" -> {answers(tpl, task)[0]!r} ...", flush=True)

    enc, batches, offset, epoch_batches = [], [], 0, None
    for e in range(a.epochs):
        items = build_items(df, bank, a.tasks, a.seed + e, a.compact_p, tok, family, a.max_len, end_chat, end_raw, rules)
        lens = [len(x[0]) for x in items]
        batches += [[offset + j for j in b] for b in make_batches(lens, a.bs, rng)]
        enc += items
        offset += len(items)
        if e == 0:
            epoch_batches = len(batches)
    lengths = [len(x[0]) for x in enc]
    print(f"rows={len(df)} sequences={len(enc)} tokens mean={sum(lengths)/len(lengths):.0f} max={max(lengths)} "
          f"over_max_len={sum(n > a.max_len for n in lengths)} bank={a.bank} tasks={a.tasks} "
          f"end_chat={end_chat!r} end_raw={end_raw!r}", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cdt)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map={"": 0}, dtype=cdt)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                             bias="none", task_type="CAUSAL_LM", target_modules=TARGETS))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = a.max_steps if a.max_steps > 0 else max(1, len(batches) // a.grad_accum)
    epoch_steps = max(1, epoch_batches // a.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total)), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(cdt == torch.float16))

    step = micro = consumed = bad = tokens = 0
    elapsed0 = 0.0
    if a.resume and (out / "state.pt").exists():
        set_peft_model_state_dict(model, load_file(str(out / "adapter" / "adapter_model.safetensors")))
        st = torch.load(out / "state.pt", map_location="cpu")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        step, micro, consumed, bad, tokens, elapsed0 = (st[k] for k in ("step", "micro", "consumed", "bad", "tokens", "elapsed"))
        print(f"RESUMED at update {step} of {total}", flush=True)

    config = {**vars(a), "started": started, "n_sequences": len(enc), "total_updates": total,
              "tokens_mean": sum(lengths) / len(lengths), "tokens_max": max(lengths),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "resumed_from_update": step}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    model.train()
    t0 = time.time() - elapsed0
    run_loss = 0.0
    logf = open(out / "train_log.jsonl", "a")
    for bidx in batches[consumed:]:
        if step >= total:
            break
        consumed += 1
        ids, att, lab = (t.cuda() for t in collate([enc[j] for j in bidx], tok.pad_token_id))
        with torch.autocast("cuda", dtype=cdt):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.grad_accum
        if not torch.isfinite(loss):
            bad += 1
            opt.zero_grad(set_to_none=True)
            if bad > 5:
                raise SystemExit("Non-finite loss repeatedly; try --dtype bfloat16.")
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
            if step % a.save_every == 0 or step >= total:
                save_ckpt(model, opt, sched, scaler, out, {"step": step, "micro": micro, "consumed": consumed,
                                                           "bad": bad, "tokens": tokens, "elapsed": time.time() - t0})
            if step % epoch_steps == 0 and step < total:
                model.save_pretrained(out / f"adapter_epoch{step // epoch_steps}")
    logf.close()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    config.update(finished=datetime.now(timezone.utc).isoformat(), updates_done=step,
                  train_minutes=round((time.time() - t0) / 60, 1),
                  peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), nonfinite_skipped=bad)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print("DONE", {k: config[k] for k in ("updates_done", "train_minutes", "peak_gpu_gb", "nonfinite_skipped")})


if __name__ == "__main__":
    main()
