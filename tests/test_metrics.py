from afriguard.evaluate import average_over_languages, per_language_weighted_f1, weighted_f1


def test_perfect():
    assert weighted_f1([0, 1, 1], [0, 1, 1]) == 1.0


def test_per_language():
    s = per_language_weighted_f1([0, 1, 0, 1], [0, 1, 1, 0], ["sw", "sw", "ha", "ha"])
    assert s["sw"] == 1.0 and s["ha"] == 0.0
    assert average_over_languages(s) == 0.5
