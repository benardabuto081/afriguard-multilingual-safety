import pytest

from afriguard.data import list_data_files, resolve_data_dir


def test_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_data_dir(tmp_path / "missing")


def test_list_files(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    assert [p.name for p in list_data_files(tmp_path)] == ["a.csv"]
