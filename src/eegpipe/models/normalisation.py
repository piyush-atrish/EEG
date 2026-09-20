"""Calibration marking and per-patient standardisation (owner: Member C, Step 12)."""

from __future__ import annotations

import pandas as pd


def mark_calibration(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add a boolean ``is_calibration`` column (first interictal windows before first seizure)."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def standardise_per_patient(
    df: pd.DataFrame, feature_cols: list[str], eps: float = 1e-8
) -> pd.DataFrame:
    """Standardise each patient's features with that patient's own calibration statistics."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def apply_arm(df: pd.DataFrame, arm: str, cfg: dict) -> pd.DataFrame:
    """Arm ``raw`` or ``subject_standardised``; both share the same ``is_calibration`` mask."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")
