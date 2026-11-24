"""Data-path handling. Schema-specific loading is added after the Stage 2 audit."""
from __future__ import annotations

from pathlib import Path

RAW_DIR = Path("data/raw")


def resolve_data_dir(data_dir: str | Path = RAW_DIR) -> Path:
    """Return an existing data directory or raise a clear error."""
    p = Path(data_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Data directory not found: {p}")
    return p


def list_data_files(data_dir: str | Path = RAW_DIR) -> list[Path]:
    """List all files under the data directory (recursively), sorted."""
    root = resolve_data_dir(data_dir)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep")
