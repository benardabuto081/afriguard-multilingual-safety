"""Parse generated text into canonical labels."""
from __future__ import annotations

import re

from .labels import labels_for, normalize_key

_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_label(text, task: str):
    """Return the canonical label, or None if the text cannot be parsed."""
    t = _THINK.sub("", text or "").strip()
    key = normalize_key(t)
    if not key:
        return None
    labels = labels_for(task)
    lk = {normalize_key(lab): lab for lab in labels}
    if key in lk:
        return lk[key]
    if task == "A":
        m = re.search(r"\b(unsafe|safe)\b", key)
        return m.group(1) if m else None
    m = re.match(r"(\d{1,2})\b", key)
    if m and 1 <= int(m.group(1)) <= len(labels):
        return labels[int(m.group(1)) - 1]
    hits = [(key.find(k), -len(k), lab) for k, lab in lk.items() if k in key]
    if hits:
        return min(hits)[2]
    for k, lab in lk.items():
        if key.startswith(" ".join(k.split()[:2])):
            return lab
    return None
