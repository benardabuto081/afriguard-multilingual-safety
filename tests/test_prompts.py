import pandas as pd

from afriguard.prompts import build_messages, select_shots


def _train():
    return pd.DataFrame({"prompt": [f"p{i}" for i in range(20)], "response": [f"r{i}" for i in range(20)],
                         "label": ["safe", "unsafe"] * 10})


def test_shots_deterministic():
    a, b = select_shots(_train(), "A", 3, 7), select_shots(_train(), "A", 3, 7)
    assert len(a) == 3 and list(a.index) == list(b.index)
    assert len(select_shots(_train(), "A", 0, 7)) == 0


def test_messages_structure():
    tr = _train()
    row = {"prompt": "hello", "response": "world"}
    m = build_messages("A", row, select_shots(tr, "A", 3, 1), True, "label", 1500, 500)
    assert len(m) == 7 and m[-1]["role"] == "user" and m[-2]["role"] == "assistant"
    assert m[-2]["content"] in ("safe", "unsafe")
    ml = build_messages("A", row, select_shots(tr, "A", 1, 1), True, "letter", 1500, 500)
    assert ml[1]["content"] in ("A", "B")
    assert "Assistant response" not in build_messages("A", row, tr.iloc[0:0], False, "label", 1500, 500)[0]["content"]
