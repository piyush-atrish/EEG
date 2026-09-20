"""Windowing and labelling (owner: Member B, Step 6). Contract C4."""

from __future__ import annotations

import pandas as pd


def seizure_intervals_for_file(annotations: pd.DataFrame, file: str) -> list[tuple[float, float]]:
    """Return ``[(start_s, end_s), ...]`` for ``file`` from annotations (contract C1)."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def make_windows(
    n_samples: int,
    fs: float,
    window_s: float,
    overlap: float,
    seizure_intervals: list[tuple[float, float]],
    drop_boundary: bool = True,
) -> pd.DataFrame:
    """Return columns ``start_sample`` (int), ``t_start_s`` (float), ``label`` (int8)."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def build_window_table(
    file_index: pd.DataFrame,
    annotations: pd.DataFrame,
    cfg: dict,
    patients: list[str] | None = None,
) -> pd.DataFrame:
    """Return the full window table (contract C4) for every included file."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")
