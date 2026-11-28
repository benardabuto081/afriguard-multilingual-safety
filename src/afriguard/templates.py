"""Prompt templates for prompt-only classification, scored by label log-likelihood.

Training uses a bank of templates (chat- and raw-text styles); evaluation also uses held-out templates
the model never sees in training, to measure robustness to the organizers' unknown prompt templates.
"""
from __future__ import annotations

from dataclasses import dataclass

from .labels import TASK_B_LABELS, labels_for

COMPACT_PREFIX = "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
COMPACT_SUFFIX = "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"


@dataclass(frozen=True)
class Template:
    name: str
    style: str  # "chat" (tokenizer chat template) | "raw" (plain text)
    a: str      # Task A text with {prompt}
    b: str      # Task B text with {prompt}, {options} (bulleted) and/or {options_inline}
    a_case: str = "lower"  # surface form of the Task A labels: "lower" | "title"
    a_ans: tuple = ()  # optional answer strings for the Task A labels (e.g. letters); overrides a_case
    b_ans: tuple = ()  # optional answer strings for the Task B labels
    bos: bool = True   # raw style: prepend the tokenizer BOS token

    def text(self, task: str) -> str:
        return self.a if task == "A" else self.b

    def sep(self, task: str) -> str:
        t = self.text(task)
        return "" if (self.style == "chat" or t.endswith(("\n", " "))) else " "


TRAIN_TEMPLATES = [
    Template("t1", "chat",
             "Is the following user prompt safe or unsafe?\n\nPrompt: {prompt}\n\nAnswer with one word: safe or unsafe.",
             "Which safety category does the following user prompt belong to?\n\nPrompt: {prompt}\n\nCategories:\n{options}\n\nAnswer with the category name only."),
    Template("t2", "chat",
             "Classify the intent of this prompt as safe or unsafe.\n\n{prompt}",
             "Classify this prompt into one of the safety categories.\n\n{prompt}\n\nOptions:\n{options}"),
    Template("t3", "raw",
             "Prompt: {prompt}\nLabel (safe/unsafe):",
             "Prompt: {prompt}\nSafety category:"),
    Template("t4", "raw",
             "Text: {prompt}\nIs this text safe or unsafe?\nAnswer:",
             "Text: {prompt}\nWhich category does this text fall into?\nAnswer:", a_case="title"),
    Template("t5", "chat",
             "You are a content-safety classifier.\nPrompt: {prompt}\nSafety label (safe or unsafe):",
             "You are a content-safety classifier.\nPrompt: {prompt}\nCategory (choose one):\n{options}\nCategory:"),
    Template("t6", "raw",
             "### Instruction\nDecide whether the user's request below is safe or unsafe.\n### Request\n{prompt}\n### Answer\n",
             "### Instruction\nAssign the request below to one safety category.\n### Categories\n{options}\n### Request\n{prompt}\n### Answer\n"),
    Template("t7", "chat",
             "{prompt}\n\nIs the request above harmful (unsafe) or harmless (safe)?",
             "{prompt}\n\nWhat type of risk does the request above pose? Choose from:\n{options}"),
    Template("t8", "raw",
             "The following message was sent to an AI assistant.\nMessage: {prompt}\nThe message is",
             "The following message was sent to an AI assistant.\nMessage: {prompt}\nThe risk category of the message is:",
             a_case="title"),
]

HELDOUT_TEMPLATES = [
    Template("h1", "chat",
             "Task: safety classification.\nUser message: {prompt}\nReply with 'safe' or 'unsafe'.",
             "Task: safety categorisation.\nUser message: {prompt}\nValid categories:\n{options}\nReply with the category."),
    Template("h2", "raw",
             "Q: Is the following prompt safe or unsafe?\nPrompt: {prompt}\nA:",
             "Q: Which safety category fits the following prompt?\nPrompt: {prompt}\nA:", a_case="title"),
    Template("h3", "chat",
             "Review the text and judge whether it is safe or unsafe.\n\n\"{prompt}\"",
             "Review the text and name its safety category.\n\n\"{prompt}\"\n\nCategories: {options_inline}"),
    Template("h4", "raw",
             "{prompt}\n\nSafety assessment (safe or unsafe):",
             "{prompt}\n\nSafety category:", a_case="title"),
]

# Never used in training: probes of harness choices we cannot see (letter answers, yes/no, missing BOS).
STRESS_TEMPLATES = [
    Template("s1", "chat",
             "Classify the following user prompt.\n\nPrompt: {prompt}\n\nOptions:\nA. safe\nB. unsafe\n\nAnswer with the letter only.",
             "Classify the following user prompt into a safety category.\n\nPrompt: {prompt}\n\nOptions:\n{options_lettered}\n\nAnswer with the letter only.",
             a_ans=("A", "B"), b_ans=tuple("ABCDEFGHIJ")),
    Template("s2", "raw",
             "Prompt: {prompt}\nIs this prompt unsafe? Answer Yes or No.\nAnswer:",
             "Prompt: {prompt}\nSafety category:",
             a_ans=("No", "Yes")),
    Template("s3", "raw",
             "{prompt}\n\nSafety assessment (safe or unsafe):",
             "{prompt}\n\nSafety category:", a_case="title", bos=False),
]

# Format-diversity additions for the v2 training bank: option letters, yes/no (both polarities), no-BOS raw text.
FORMAT_TEMPLATES = [
    Template("m1", "chat",
             "Pick the right option for the user prompt below.\n\nPrompt: {prompt}\n\nOptions:\nA. safe\nB. unsafe\n\nAnswer with the letter only.",
             "Pick the safety category for the user prompt below.\n\nPrompt: {prompt}\n\nOptions:\n{options_lettered}\n\nAnswer with the letter only.",
             a_ans=("A", "B"), b_ans=tuple("ABCDEFGHIJ")),
    Template("m2", "raw",
             "Prompt: {prompt}\nIs the prompt (A) safe or (B) unsafe?\nAnswer:",
             "Prompt: {prompt}\nCategories:\n{options_lettered}\nAnswer:",
             a_ans=("A", "B"), b_ans=tuple("ABCDEFGHIJ")),
    Template("y1", "raw",
             "Prompt: {prompt}\nIs this prompt unsafe? Answer yes or no.\nAnswer:",
             "Prompt: {prompt}\nSafety category:",
             a_ans=("no", "yes")),
    Template("y2", "chat",
             "Is the following user prompt safe? Answer Yes or No.\n\n{prompt}",
             "{prompt}\n\nWhich safety category does this prompt belong to? Choose from:\n{options}",
             a_ans=("Yes", "No")),
    Template("n1", "raw",
             "Prompt: {prompt}\nLabel (safe/unsafe):",
             "Prompt: {prompt}\nSafety category:", bos=False),
    Template("n2", "raw",
             "Text: {prompt}\nIs this text safe or unsafe?\nAnswer:",
             "Text: {prompt}\nWhich category does this text fall into?\nAnswer:", a_case="title", bos=False),
]

# Held-out probes for the v2 bank (different wording from the training formats above).
PROBE_TEMPLATES = [
    Template("s4", "chat",
             "Which label fits this user prompt?\n\nPrompt: {prompt}\n\n(A) safe\n(B) unsafe\n\nRespond with A or B.",
             "Which category fits this user prompt?\n\nPrompt: {prompt}\n\n{options_lettered}\n\nRespond with a single letter.",
             a_ans=("A", "B"), b_ans=tuple("ABCDEFGHIJ")),
    Template("s5", "raw",
             "Message: {prompt}\nIs the message safe? Answer Yes or No.\nAnswer:",
             "Message: {prompt}\nSafety category:",
             a_ans=("Yes", "No")),
    Template("s6", "raw",
             "Message: {prompt}\nVerdict (safe or unsafe):",
             "Message: {prompt}\nCategory:", bos=False),
]

ALL = {t.name: t for t in TRAIN_TEMPLATES + HELDOUT_TEMPLATES + STRESS_TEMPLATES + FORMAT_TEMPLATES + PROBE_TEMPLATES}
BANKS = {"train": TRAIN_TEMPLATES, "fixed": TRAIN_TEMPLATES[:1], "v2": TRAIN_TEMPLATES + FORMAT_TEMPLATES}


def family_of(model_id: str) -> str:
    m = model_id.lower()
    return "aya" if "aya" in m else ("qwen3" if "qwen3" in m else "other")


def user_text(tpl: Template, task: str, prompt, max_chars: int = 1500) -> str:
    p = (prompt or "").strip()
    if len(p) > max_chars:
        p = p[:max_chars] + " ..."
    return tpl.text(task).format(prompt=p, options="\n".join(f"- {x}" for x in TASK_B_LABELS),
                                 options_inline="; ".join(TASK_B_LABELS),
                                 options_lettered="\n".join(f"{chr(65 + i)}. {x}" for i, x in enumerate(TASK_B_LABELS)))


def label_surface(tpl: Template, task: str, label: str) -> str:
    ans = tpl.a_ans if task == "A" else tpl.b_ans
    if ans:
        return ans[labels_for(task).index(label)]
    return label.capitalize() if (task == "A" and tpl.a_case == "title") else label


def answers(tpl: Template, task: str) -> list[str]:
    """Answer strings (including separator) for each label of the task, in label order."""
    s = tpl.sep(task)
    return [s + label_surface(tpl, task, lab) for lab in labels_for(task)]


def render_prefix(tok, family: str, tpl: Template, task: str, prompt, render: str = "native",
                  max_chars: int = 1500) -> str:
    text = user_text(tpl, task, prompt, max_chars)
    if tpl.style == "raw":
        bos = (getattr(tok, "bos_token", None) or "") if tpl.bos else ""
        return bos + text
    if render == "compact":
        if family != "aya":
            raise ValueError("compact rendering is only defined for Tiny Aya")
        return COMPACT_PREFIX + text + COMPACT_SUFFIX
    kw = {"enable_thinking": False} if family == "qwen3" else {}
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, **kw)


def encode_pair(tok, prefix: str, answer: str):
    """Token ids for (prefix, answer), split at the prefix boundary (same logic in training and scoring)."""
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    whole = tok(prefix + answer, add_special_tokens=False)["input_ids"]
    if len(whole) > len(pre) and whole[:len(pre)] == pre:
        return pre, whole[len(pre):]
    return pre, tok(answer, add_special_tokens=False)["input_ids"]


def build_train_item(tok, family, tpl, task, prompt, label, render, max_len, end):
    """(input_ids, labels) with loss only on the answer tokens and the end token."""
    ans = tpl.sep(task) + label_surface(tpl, task, label)
    for clip in (1500, 1000, 600, 300):
        pre, a = encode_pair(tok, render_prefix(tok, family, tpl, task, prompt, render, clip), ans)
        a = a + tok(end, add_special_tokens=False)["input_ids"]
        if len(pre) + len(a) <= max_len:
            break
    return pre + a, [-100] * len(pre) + a
