"""Language sampling weights."""
from __future__ import annotations


def language_weights(counts: dict[str, int], alpha: float) -> dict[str, float]:
    """Temperature sampling: p(l) proportional to n_l ** alpha (alpha=0 uniform, alpha=1 proportional)."""
    w = {lang: float(n) ** alpha for lang, n in counts.items()}
    s = sum(w.values())
    return {lang: v / s for lang, v in w.items()}
