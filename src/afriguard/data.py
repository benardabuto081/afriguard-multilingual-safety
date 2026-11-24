"""Data-path handling and split loading."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .labels import TASK_DIRS

RAW_DIR = Path("data/raw")


def resolve_data_dir(data_dir: str | Path = RAW_DIR) -> Path:
    p = Path(data_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Data directory not found: {p}")
    return p


def list_data_files(data_dir: str | Path = RAW_DIR) -> list[Path]:
    root = resolve_data_dir(data_dir)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep")


def load_split(data_dir: str | Path, task: str, language: str, split: str) -> pd.DataFrame:
    """Load one task/language/split CSV as strings (blank cells stay empty strings)."""
    path = Path(data_dir) / TASK_DIRS[task] / language / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)
