import pytest

from afriguard.labels import TASK_A_LABELS, TASK_B_LABELS
from afriguard.templates import (
    ALL,
    BANKS,
    answers,
    build_train_item,
    encode_pair,
    render_prefix,
    user_text,
)


class FakeTok:
    bos_token = "<B>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "<U>" + msgs[-1]["content"] + "<A>"


@pytest.mark.parametrize("name", sorted(ALL))
def test_templates_format_cleanly(name):
    t = ALL[name]
    for task in ("A", "B"):
        s = user_text(t, task, "PROMPT {x} TEXT")
        assert "{" not in s.replace("PROMPT {x} TEXT", "") and "PROMPT {x} TEXT" in s
        assert len(set(answers(t, task))) == (2 if task == "A" else 10)


def test_answer_surface_forms():
    assert [a.strip() for a in answers(ALL["t1"], "A")] == TASK_A_LABELS
    assert [a.strip() for a in answers(ALL["h2"], "A")] == ["Safe", "Unsafe"]
    assert [a.strip() for a in answers(ALL["t1"], "B")] == TASK_B_LABELS
    assert answers(ALL["t3"], "A")[0] == " safe" and answers(ALL["t1"], "A")[0] == "safe"
    assert answers(ALL["t6"], "A")[0] == "safe"  # raw text already ends with a newline


def test_banks():
    assert len(BANKS["train"]) == 8 and len(BANKS["fixed"]) == 1
    assert not {t.name for t in BANKS["train"]} & {"h1", "h2", "h3", "h4"}


def test_train_item_masks_prefix():
    tok = FakeTok()
    for name in ("t1", "t3"):
        ids, lab = build_train_item(tok, "other", ALL[name], "B", "hello", TASK_B_LABELS[2], "native", 4000, "<E>")
        k = next(i for i, x in enumerate(lab) if x != -100)
        assert all(x == -100 for x in lab[:k]) and lab[k:] == ids[k:]
        assert "".join(chr(i) for i in ids[k:]).strip() == TASK_B_LABELS[2] + "<E>"


def test_clipping_keeps_answer():
    ids, _lab = build_train_item(FakeTok(), "other", ALL["t3"], "A", "x" * 5000, "unsafe", "native", 700, "<E>")
    assert len(ids) <= 700 and "".join(chr(i) for i in ids[-9:]).endswith("unsafe<E>")


def test_encode_pair_boundary():
    _pre, ans = encode_pair(FakeTok(), render_prefix(FakeTok(), "other", ALL["t3"], "A", "p"), " safe")
    assert "".join(map(chr, ans)) == " safe"
