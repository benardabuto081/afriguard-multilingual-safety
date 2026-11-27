import random
from types import SimpleNamespace

from afriguard.labels import TASK_B_LABELS
from afriguard.sampling import build_rules, repeat_count


def test_parse():
    assert build_rules("", "") is None
    r = build_rules("8:safe=2, 9:unsafe=1.5", "oromo=3")
    assert r["cells"][(8, "safe")] == 2.0 and r["cells"][(9, "unsafe")] == 1.5 and r["langs"]["oromo"] == 3.0


def test_repeat_counts():
    rules, rng = build_rules("8:safe=2", "oromo=3"), random.Random(0)
    hate_safe = SimpleNamespace(lang="hausa", label_a="safe", label_b=TASK_B_LABELS[7])
    assert repeat_count(rules, hate_safe, rng) == 2
    assert repeat_count(rules, SimpleNamespace(lang="oromo", label_a="unsafe", label_b=TASK_B_LABELS[0]), rng) == 3
    assert repeat_count(rules, SimpleNamespace(lang="hausa", label_a="unsafe", label_b=TASK_B_LABELS[0]), rng) == 1


def test_fractional_mean():
    rules, rng = build_rules("", "twi=1.5"), random.Random(1)
    row = SimpleNamespace(lang="twi", label_a="safe", label_b=TASK_B_LABELS[0])
    assert abs(sum(repeat_count(rules, row, rng) for _ in range(4000)) / 4000 - 1.5) < 0.05
