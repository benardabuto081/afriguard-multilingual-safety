#!/usr/bin/env python3
"""Score by label log-likelihood (as the organizers describe), prompt only, under several templates.
Works for adapters (4-bit) or merged standalone checkpoints (--no-4bit)."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from afriguard.data import load_split
from afriguard.evaluate import weighted_f1
from afriguard.labels import LANGUAGES, labels_for
from afriguard.templates import ALL, answers, encode_pair, family_of, render_prefix


def _last_logits(model, **kw):
    try:
        out = model(**kw, logits_to_keep=1)
    except TypeError:
        out = model(**kw)
    return out


@torch.no_grad()
def score_candidates(model, pad_id, prefix_ids, cand_ids, bs=8):
    """Sum of token log-probs of each candidate continuation given each prefix. Prefix is run once (KV cache)."""
    n, k_all, dev = len(prefix_ids), len(cand_ids), model.device
    sums = np.zeros((n, k_all), dtype=np.float32)
    lens = np.array([len(c) for c in cand_ids], dtype=np.float32)
    order = sorted(range(n), key=lambda i: len(prefix_ids[i]))
    for s in range(0, n, bs):
        idx = order[s:s + bs]
        b, width = len(idx), max(len(prefix_ids[j]) for j in idx)
        ids = torch.full((b, width), pad_id, dtype=torch.long)
        att = torch.zeros((b, width), dtype=torch.long)
        for r, j in enumerate(idx):
            p = prefix_ids[j]
            ids[r, width - len(p):] = torch.tensor(p)
            att[r, width - len(p):] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        ids, att, pos = ids.to(dev), att.to(dev), pos.to(dev)
        out = _last_logits(model, input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True)
        first = torch.log_softmax(out.logits[:, -1].float(), -1)
        plen = att.sum(-1, keepdim=True)
        for k, c in enumerate(cand_ids):
            lp = first[:, c[0]]
            if len(c) > 1:
                cache = copy.deepcopy(out.past_key_values)
                cin = torch.tensor(c[:-1], device=dev).unsqueeze(0).expand(b, -1)
                am = torch.cat([att, torch.ones((b, len(c) - 1), dtype=att.dtype, device=dev)], 1)
                cp = plen + torch.arange(len(c) - 1, device=dev).unsqueeze(0)
                o2 = model(input_ids=cin, attention_mask=am, position_ids=cp, past_key_values=cache, use_cache=True)
                lps = torch.log_softmax(o2.logits.float(), -1)
                tgt = torch.tensor(c[1:], device=dev).unsqueeze(0).expand(b, -1)
                lp = lp + lps.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(-1)
            sums[idx, k] = lp.float().cpu().numpy()
    return sums, lens


@torch.no_grad()
def reference_score(model, prefix, cand):
    ids = torch.tensor([prefix + cand], device=model.device)
    lp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
    return float(sum(lp[len(prefix) - 1 + j, cand[j]] for j in range(len(cand))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--family", default=None)
    ap.add_argument("--templates", nargs="+", default=["t1", "t3", "h1", "h2", "h3", "h4"])
    ap.add_argument("--render", default="native", choices=["native", "compact"])
    ap.add_argument("--tasks", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--languages", nargs="+", default=LANGUAGES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--check", action="store_true", help="verify cached scoring against a naive reference")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    cdt = torch.float16 if a.dtype == "float16" else torch.bfloat16
    family = a.family or family_of(a.model)
    out = Path(a.out_dir) if a.out_dir else Path("experiments/lbl_eval")
    out.mkdir(parents=True, exist_ok=True)
    has_tok = a.adapter and (Path(a.adapter) / "tokenizer_config.json").exists()
    tok = AutoTokenizer.from_pretrained(a.adapter if has_tok else a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {} if a.no_4bit else {"quantization_config": BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cdt)}
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map={"": 0}, dtype=cdt, **kw)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    metrics = {"config": vars(a), "templates": {}}
    checked, t_all = False, time.time()
    for tname in a.templates:
        tpl, t0 = ALL[tname], time.time()
        res = {t: {} for t in a.tasks}
        res.update({f"{t}_mean": {} for t in a.tasks})
        arrays = {}
        for task in a.tasks:
            labs = labels_for(task)
            for lang in a.languages:
                df = load_split(a.data_dir, task, lang, a.split)
                if a.limit:
                    df = df.head(a.limit)
                prefixes = [render_prefix(tok, family, tpl, task, p, a.render) for p in df["prompt"]]
                pre = [tok(x, add_special_tokens=False)["input_ids"] for x in prefixes]
                cand = [encode_pair(tok, prefixes[0], s)[1] for s in answers(tpl, task)]
                sums, lens = score_candidates(model, tok.pad_token_id, pre, cand, a.batch_size)
                if a.check and not checked:
                    diffs = [abs(sums[i, k] - reference_score(model, pre[i], cand[k]))
                             for i in range(min(4, len(pre))) for k in range(len(cand))]
                    print(f"CHECK cached-vs-reference max abs diff = {max(diffs):.4f} "
                          f"(template {tname}, task {task}, {lang})", flush=True)
                    assert max(diffs) < 0.25, "cached scoring disagrees with the reference implementation"
                    checked = True
                gold = df["label"].tolist()
                res[task][lang] = weighted_f1(gold, [labs[i] for i in sums.argmax(1)])
                res[f"{task}_mean"][lang] = weighted_f1(gold, [labs[i] for i in (sums / lens).argmax(1)])
                arrays[f"{lang}_{task}"] = sums
                arrays[f"{lang}_{task}_ids"] = df["id"].values.astype(str)
        for key in list(res):
            if res[key]:
                res[key]["_avg"] = float(np.mean(list(res[key].values())))
        np.savez_compressed(out / f"lp_{tname}.npz", **arrays)
        metrics["templates"][tname] = res
        print(f"template {tname:3s} ({tpl.style}) {round(time.time() - t0)}s  " + "  ".join(
            f"{t}={res[t]['_avg']:.4f} ({t}_mean={res[t + '_mean']['_avg']:.4f})" for t in a.tasks), flush=True)
    metrics["total_seconds"] = round(time.time() - t_all, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
