from afriguard.parsing import parse_label


def test_task_a():
    assert parse_label("Unsafe.", "A") == "unsafe"
    assert parse_label("safe", "A") == "safe"
    assert parse_label("The intent is unsafe", "A") == "unsafe"
    assert parse_label("<think>\n\n</think>\n\nsafe", "A") == "safe"
    assert parse_label("", "A") is None
    assert parse_label("maybe", "A") is None


def test_task_b():
    assert parse_label("Privacy & Data Misuse", "B") == "Privacy & Data Misuse"
    assert parse_label("privacy and data misuse.", "B") == "Privacy & Data Misuse"
    assert parse_label("7", "B") == "Privacy & Data Misuse"
    assert parse_label("Criminal Activity", "B") == "Criminal Activity & Exploitation"
    assert parse_label("blah", "B") is None
