from afriguard.labels import TASK_B_LABELS
from afriguard.templates import ALL, answers, render_prefix


class Tok:
    bos_token = "<B>"


def test_stress_answers():
    assert answers(ALL["s1"], "A") == ["A", "B"]
    assert answers(ALL["s1"], "B") == list("ABCDEFGHIJ")
    assert answers(ALL["s2"], "A") == [" No", " Yes"]
    assert [a.strip() for a in answers(ALL["s2"], "B")] == TASK_B_LABELS


def test_bos_flag():
    assert render_prefix(Tok(), "other", ALL["h4"], "A", "p").startswith("<B>")
    assert not render_prefix(Tok(), "other", ALL["s3"], "A", "p").startswith("<B>")
