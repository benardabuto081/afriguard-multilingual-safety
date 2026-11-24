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


def full_metrics(y_true, y_pred, labels) -> dict:
    """Weighted/macro F1, accuracy, per-class P/R/F1 and a confusion matrix."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    y_true, y_pred, labels = list(y_true), list(y_pred), list(labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    extra = sorted(set(y_pred) - set(labels))
    all_labels = labels + extra
    return {
        "weighted_f1": weighted_f1(y_true, y_pred),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                            "support": int(s[i])} for i, lab in enumerate(labels)},
        "confusion": {"labels": all_labels,
                      "matrix": confusion_matrix(y_true, y_pred, labels=all_labels).tolist()},
    }
