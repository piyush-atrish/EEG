"""
PLACEHOLDER — owned by Member A (Step 1). Not part of Member B's deliverable.
Minimal path helpers matching the signatures named in README Section 9.
"""
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
