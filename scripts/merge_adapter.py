#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and save a standalone checkpoint (fp16)."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
print("merged", sum(p.numel() for p in model.parameters()) / 1e9, "B parameters ->", a.out)
