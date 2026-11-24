#!/usr/bin/env python3
"""E1: zero/few-shot baseline evaluation of candidate generative models."""
from __future__ import annotations

import argparse
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from afriguard.data import load_split
from afriguard.evaluate import full_metrics
from afriguard.labels import LANGUAGES, LETTERS, TASK_DIRS, UNPARSED, labels_for
from afriguard.parsing import parse_label
from afriguard.prompts import build_messages, select_shots
from afriguard.utils import set_seed


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        import transformers

        info.update(torch=torch.__version__, transformers=transformers.__version__,
                    cuda=torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:
        pass
    return info


class HFModel:
    def __init__(self, model_id, revision, dtype):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.model_id = torch, model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32, "auto": "auto"}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dt,
            device_map="auto" if torch.cuda.is_available() else None)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def render(self, messages):
        kw = {"enable_thinking": False} if "qwen3" in self.model_id.lower() else {}
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)

    def token_lengths(self, texts):
        return [len(x) for x in self.tok(texts, add_special_tokens=False)["input_ids"]]

    def letter_ids(self, n):
        ids = []
        for ch in LETTERS[:n]:
            t = self.tok.encode(ch, add_special_tokens=False)
            if len(t) != 1:
                raise ValueError(f"letter {ch!r} is not a single token: {t}")
            ids.append(t[0])
        return ids

    def _batch(self, texts, mode, max_new_tokens, letter_ids):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            if mode == "choice":
                try:
                    out = self.model(**enc, logits_to_keep=1)
                except TypeError:
                    out = self.model(**enc)
                return out.logits[:, -1, letter_ids].float().argmax(-1).tolist()
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            return self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def run(self, texts, mode, max_new_tokens, letter_ids, batch_size):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out, i = [None] * len(texts), 0
        while i < len(order):
            idx = order[i:i + batch_size]
            try:
                res = self._batch([texts[j] for j in idx], mode, max_new_tokens, letter_ids)
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                continue
            for j, r in zip(idx, res):
                out[j] = r
            i += len(idx)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, or dummy:constant for a pipeline test")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tasks", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--languages", nargs="+", default=LANGUAGES)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--mode", default="generate", choices=["generate", "choice"])
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--prompt-only", action="store_true")
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--shot-chars", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="experiments/baselines")
    a = ap.parse_args()

    set_seed(a.seed)
    use_response = not a.prompt_only
    fmt = "letter" if a.mode == "choice" else "label"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", a.model).strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{slug}_{a.mode}_{a.shots}shot_{'both' if use_response else 'prompt'}_{a.split}_{stamp}"
    out = Path(a.out_dir) / run_id
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    config = {**vars(a), "run_id": run_id, "use_response": use_response, "answer_format": fmt}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    dummy = a.model.startswith("dummy:")
    model = None if dummy else HFModel(a.model, a.revision, a.dtype)
    t_all = time.time()
    metrics = {"config": config, "env": env_info(), "tasks": {},
               "n_params": None if dummy else model.n_params}

    for task in a.tasks:
        labels = labels_for(task)
        letter_ids = model.letter_ids(len(labels)) if (model and a.mode == "choice") else None
        per_lang = {}
        for lang in a.languages:
            t0 = time.time()
            train = load_split(a.data_dir, task, lang, "train")
            df = load_split(a.data_dir, task, lang, a.split)
            if a.limit:
                df = df.head(a.limit)
            shots = select_shots(train, task, a.shots, a.seed)
            rows = df.to_dict("records")
            extra = {}
            if dummy:
                maj = train["label"].mode()[0]
                raw, preds = [maj] * len(rows), [maj] * len(rows)
            else:
                texts = [model.render(build_messages(task, r, shots, use_response, fmt,
                                                     a.max_chars, a.shot_chars)) for r in rows]
                lens = model.token_lengths(texts)
                extra = {"tokens_mean": sum(lens) / len(lens), "tokens_max": max(lens)}
                res = model.run(texts, a.mode, a.max_new_tokens, letter_ids, a.batch_size)
                if a.mode == "choice":
                    raw = [LETTERS[i] for i in res]
                    preds = [labels[i] for i in res]
                else:
                    raw = res
                    preds = [parse_label(t, task) or UNPARSED for t in res]
            pd.DataFrame({"id": df["id"], "label": preds, "raw": raw}).to_csv(
                out / "predictions" / f"{TASK_DIRS[task]}_{lang}.csv", index=False)
            rec = {"n": len(df), "unparsed_rate": sum(p == UNPARSED for p in preds) / max(len(preds), 1),
                   "seconds": round(time.time() - t0, 1), **extra}
            if "label" in df.columns:
                rec.update(full_metrics(df["label"], preds, labels))
            per_lang[lang] = rec
            print(f"{task} {lang:8s} wF1={rec.get('weighted_f1', float('nan')):.3f} "
                  f"unparsed={rec['unparsed_rate']:.1%} n={rec['n']} {rec['seconds']}s", flush=True)
        scores = [v["weighted_f1"] for v in per_lang.values() if "weighted_f1" in v]
        metrics["tasks"][task] = {"per_language": per_lang,
                                  "avg_weighted_f1": sum(scores) / len(scores) if scores else None,
                                  "n_languages": len(per_lang)}
        print(f"== Task {task} average weighted F1: {metrics['tasks'][task]['avg_weighted_f1']}", flush=True)
    metrics["total_seconds"] = round(time.time() - t_all, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
