"""Prompt/answer format for the fine-tuned model: joint (both labels) or single-task."""
from __future__ import annotations

import re

from .parsing import parse_label

BASE = "Classify this prompt and model response for AI safety."
INSTR = {
    "joint": BASE + " Give the safety label (safe or unsafe) and the safety category of the user prompt.",
    "A": BASE + " Give the safety label (safe or unsafe) of the user prompt.",
    "B": BASE + " Give the safety category of the user prompt.",
}
FORMAT = {
    "joint": ["safety: <safe|unsafe>", "category: <category name>"],
    "A": ["safety: <safe|unsafe>"],
    "B": ["category: <category name>"],
}
SYSTEM = "You are a content-safety classifier."


def _clip(text, n: int) -> str:
    s = (text or "").strip()
    return s if len(s) <= n else s[:n] + " ..."


def build_user_text(prompt, response, use_response: bool = True, max_chars: int = 1500,
                    mode: str = "joint") -> str:
    parts = [INSTR[mode], "", "User prompt:", _clip(prompt, max_chars)]
    if use_response:
        parts += ["", "Model response:", _clip(response, max_chars)]
    parts += ["", "Answer in the format:"] + FORMAT[mode]
    return "\n".join(parts)


def format_answer(label_a: str, label_b: str, mode: str = "joint") -> str:
    lines = {"A": f"safety: {label_a}", "B": f"category: {label_b}"}
    return "\n".join(lines[k] for k in ("A", "B") if mode in ("joint", k))


def parse_answer(text):
    """Return (task_a_label, task_b_label); either may be None."""
    t = text or ""
    ma = re.search(r"safety\s*:\s*([^\n]*)", t, re.IGNORECASE)
    mb = re.search(r"category\s*:\s*([^\n]*)", t, re.IGNORECASE)
    return (parse_label(ma.group(1), "A") if ma else None,
            parse_label(mb.group(1), "B") if mb else None)


def end_token(tok) -> str:
    vocab = tok.get_vocab()
    for t in ("<|im_end|>", "<end_of_turn>", "<|END_OF_TURN_TOKEN|>"):
        if t in vocab:
            return t
    return tok.eos_token


def render_prompt(tok, model_id: str, user_text: str) -> str:
    kw = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}
    msgs = [{"role": "user", "content": user_text}]
    if "aya" in model_id.lower():  # replaces the long default preamble in Cohere's chat template
        msgs.insert(0, {"role": "system", "content": SYSTEM})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
