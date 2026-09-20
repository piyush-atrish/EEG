"""Leakage guards L1-L8 (owner: Member C, Step 10)."""

from __future__ import annotations

import pandas as pd


def assert_patient_disjoint(train_patients, test_patients) -> None:
    """L1: train and test patient sets must not overlap."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def assert_patient_grouping(df: pd.DataFrame, patient_map: dict) -> None:
    """L1: no ``patient`` value equals a key of patient_map; each case maps to one patient."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def assert_finite_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Raise if any feature value is NaN or inf."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def assert_no_calibration_scored(pred_df: pd.DataFrame, calibration_keys) -> None:
    """Raise if any calibration window appears among the scored predictions."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def label_shuffle_check(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    cfg: dict,
    n_test_patients: int = 3,
    seed: int = 0,
) -> float:
    """L8: shuffle TRAINING labels only, run LOSO on a few patients, return mean AUC (~0.5)."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")
