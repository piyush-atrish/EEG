"""Leave-one-subject-out loop, sampling and tuning (owner: Member C, Steps 10-11)."""

from __future__ import annotations

import pandas as pd


def run_loso(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    cfg: dict,
    arm: str,
    patients: list[str] | None = None,
    shuffle_train_labels: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(predictions C6, per_patient C7)``."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def subsample_training(df: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Balance TRAINING data only (leakage rule L3)."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def tune_model(pipe, grid: dict, X, y, groups, cfg: dict, seed: int):
    """Nested, patient-grouped tuning (leakage rule L4); returns the fitted estimator."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")
