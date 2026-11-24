"""Metric skeleton. Must be verified against the official evaluation code before use."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from sklearn.metrics import f1_score


def weighted_f1(y_true: Sequence, y_pred: Sequence) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def per_language_weighted_f1(
    y_true: Sequence, y_pred: Sequence, langs: Sequence[str]
) -> dict[str, float]:
    groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for t, p, lang in zip(y_true, y_pred, langs):
        groups[lang][0].append(t)
        groups[lang][1].append(p)
    return {lang: weighted_f1(t, p) for lang, (t, p) in sorted(groups.items())}


def average_over_languages(scores: dict[str, float]) -> float:
    return sum(scores.values()) / len(scores) if scores else 0.0
