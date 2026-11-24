import pytest

from afriguard.utils import load_config


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 42\nlr: 0.0002\n")
    cfg = load_config(p)
    assert cfg["seed"] == 42


def test_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
