import random

import numpy as np

from afriguard.utils import set_seed


def test_seed_determinism():
    set_seed(7)
    a = (random.random(), np.random.rand())
    set_seed(7)
    b = (random.random(), np.random.rand())
    assert a == b
