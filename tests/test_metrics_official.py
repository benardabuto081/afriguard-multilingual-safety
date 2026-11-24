import importlib.util
import random
from pathlib import Path

import pytest

from afriguard.evaluate import weighted_f1
from afriguard.labels import normalize_key

KIT = Path("data/starting_kit/afriguard_eval.py")
pytestmark = pytest.mark.skipif(not KIT.exists(), reason="starting kit not present")


def _official():
    spec = importlib.util.spec_from_file_location("afriguard_eval", KIT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_weighted_f1_matches_official():
    ev, rng, labs = _official(), random.Random(0), ["a", "b", "c", "d"]
    for _ in range(20):
        y = [rng.choice(labs) for _ in range(200)]
        p = [rng.choice(labs) for _ in range(200)]
        assert abs(weighted_f1(y, p) - ev.weighted_f1(y, p)) < 1e-9


def test_normalize_key_matches_official():
    ev = _official()
    for s in ["Privacy & Data Misuse", "privacy_and_data_misuse", "Unsafe or High-Risk Advice", "7", " SAFE. "]:
        assert normalize_key(s) == ev.normalize_key(s)
