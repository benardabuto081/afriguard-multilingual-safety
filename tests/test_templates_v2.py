from afriguard.templates import ALL, BANKS, answers, render_prefix


class Tok:
    bos_token = "<B>"


def test_v2_bank_and_probes():
    assert len(BANKS["v2"]) == 14
    names = {t.name for t in BANKS["v2"]}
    assert not names & {"h1", "h2", "h3", "h4", "s4", "s5", "s6"}
    assert answers(ALL["y2"], "A") == ["Yes", "No"]
    assert answers(ALL["y1"], "A") == [" no", " yes"]
    assert answers(ALL["m1"], "B") == list("ABCDEFGHIJ")
    assert answers(ALL["s5"], "A") == [" Yes", " No"]


def test_v2_bos_flags():
    assert not render_prefix(Tok(), "other", ALL["n1"], "A", "p").startswith("<B>")
    assert render_prefix(Tok(), "other", ALL["t3"], "A", "p").startswith("<B>")
    assert not render_prefix(Tok(), "other", ALL["s6"], "A", "p").startswith("<B>")
