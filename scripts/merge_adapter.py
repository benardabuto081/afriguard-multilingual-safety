#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model on CPU in fp32 (no precision loss), then save a standalone fp16 checkpoint."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload().to(torch.float16)
for k in ("dtype", "torch_dtype"):
    if hasattr(model.config, k):
        setattr(model.config, k, torch.float16)
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
print("merged", round(sum(p.numel() for p in model.parameters()) / 1e9, 3), "B parameters ->", a.out)
