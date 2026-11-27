"""Language sampling weights."""
from __future__ import annotations


def language_weights(counts: dict[str, int], alpha: float) -> dict[str, float]:
    """Temperature sampling: p(l) proportional to n_l ** alpha (alpha=0 uniform, alpha=1 proportional)."""
    w = {lang: float(n) ** alpha for lang, n in counts.items()}
    s = sum(w.values())
    return {lang: v / s for lang, v in w.items()}


def build_rules(cells: str = "", langs: str = ""):
    """Upsampling rules. cells: '8:safe=2,9:unsafe=1.5' (category number 1-10 : Task A label = factor);
    langs: 'oromo=2,twi=2'. Returns None when no rule is given."""
    cell, lang = {}, {}
    for part in filter(None, (cells or "").replace(" ", "").split(",")):
        k, v = part.split("=")
        c, lab = k.split(":")
        cell[(int(c), lab.lower())] = float(v)
    for part in filter(None, (langs or "").replace(" ", "").split(",")):
        k, v = part.split("=")
        lang[k.lower()] = float(v)
    return {"cells": cell, "langs": lang} if (cell or lang) else None


def repeat_count(rules, row, rng) -> int:
    """Repeat count for a training row (needs .lang, .label_a, .label_b) with stochastic rounding of fractions."""
    import math

    from .labels import TASK_B_LABELS
    cat = TASK_B_LABELS.index(row.label_b) + 1
    f = max(rules["langs"].get(row.lang, 1.0), rules["cells"].get((cat, row.label_a), 1.0))
    n = math.floor(f)
    return n + (1 if rng.random() < f - n else 0)
