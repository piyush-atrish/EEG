"""Path builders for every artifact in the README contracts (no path is ever hard-coded)."""

from __future__ import annotations

from pathlib import Path


def annotations_csv(cfg: dict) -> Path:
    return Path(cfg["paths"]["interim"]) / "annotations.csv"


def file_index_csv(cfg: dict) -> Path:
    return Path(cfg["paths"]["interim"]) / "file_index.csv"


def preprocessed_path(cfg: dict, case: str, file_stem: str) -> Path:
    return Path(cfg["paths"]["preprocessed"]) / case / f"{file_stem}.npy"


def windows_path(cfg: dict, case: str) -> Path:
    return Path(cfg["paths"]["windows"]) / f"{case}.parquet"


def features_path(cfg: dict, case: str) -> Path:
    return Path(cfg["paths"]["features"]) / f"{case}.parquet"


def predictions_path(cfg: dict, arm: str, model: str) -> Path:
    return Path(cfg["paths"]["predictions"]) / f"{arm}__{model}.parquet"


def preprocess_status_csv(cfg: dict) -> Path:
    return Path(cfg["paths"]["logs"]) / "preprocess_status.csv"


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory of ``path`` (if needed) and return ``path`` as a Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
