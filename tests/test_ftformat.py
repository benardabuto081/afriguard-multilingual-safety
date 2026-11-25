from afriguard.ftformat import build_user_text, format_answer, parse_answer
from afriguard.sampling import language_weights


def test_roundtrip():
    assert parse_answer(format_answer("unsafe", "Privacy & Data Misuse")) == ("unsafe", "Privacy & Data Misuse")


def test_partial_and_junk():
    assert parse_answer("safety: safe") == ("safe", None)
    assert parse_answer("junk") == (None, None)


def test_clip_and_no_response():
    assert len(build_user_text("p" * 5000, "r" * 5000, True, 100)) < 800
    assert "Model response" not in build_user_text("p", "r", False)


def test_sampling():
    c = {"a": 100, "b": 400}
    assert abs(sum(language_weights(c, 0.5).values()) - 1) < 1e-9
    assert language_weights(c, 0.0) == {"a": 0.5, "b": 0.5}
    assert abs(language_weights(c, 1.0)["b"] - 0.8) < 1e-9
