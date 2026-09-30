"""Calibration marking and per-patient standardisation (owner: Member C, Step 12)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)


def mark_calibration(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add a boolean `is_calibration` column: the first calibration-length stretch of
    interictal windows before each patient's first seizure.

    Per patient: sort by `(case, file, start_sample)`; find the first ictal window (a patient
    with no seizures at all is treated as if it occurred after their last window, so every
    window is a candidate); among the interictal windows strictly before it, mark the first
    `n = ceil(minutes * 60 / step_s)` as calibration, where
    `step_s = window_s * (1 - overlap)`. If fewer than `min_windows` candidates are available
    there (a patient whose seizure comes very early), fall back to the first `n` interictal
    windows anywhere in the patient's timeline instead, and log a warning -- this trades a
    small amount of "calibration happens exactly at the start" purity for actually having a
    calibration period at all.
    """
    minutes = cfg["evaluation"]["calibration"]["minutes"]
    min_windows = cfg["evaluation"]["calibration"]["min_windows"]
    step_s = float(cfg["segmentation"]["window_s"]) * (1 - float(cfg["segmentation"]["overlap"]))
    n = math.ceil(minutes * 60 / step_s)

    out = df.copy()
    out["is_calibration"] = False

    for patient, group in df.groupby("patient", sort=False):
        ordered = group.sort_values(["case", "file", "start_sample"])
        labels = ordered["label"].to_numpy()
        ictal_positions = np.flatnonzero(labels == 1)
        first_ictal_pos = int(ictal_positions[0]) if len(ictal_positions) else len(ordered)

        before = ordered.iloc[:first_ictal_pos]
        candidates = before.index[before["label"].to_numpy() == 0]

        if len(candidates) < min_windows:
            logger.warning(
                "Patient %s: only %d interictal window(s) before the first seizure (need >= "
                "%d for calibration); falling back to the first %d interictal windows overall",
                patient, len(candidates), min_windows, n,
            )
            candidates = ordered.index[ordered["label"].to_numpy() == 0]

        out.loc[candidates[:n], "is_calibration"] = True

    return out


def standardise_per_patient(
    df: pd.DataFrame, feature_cols: list[str], eps: float = 1e-8
) -> pd.DataFrame:
    """Standardise each patient's features with that patient's own calibration statistics.

    For each patient: mean and (population) std are computed from that patient's calibration
    windows only (`df["is_calibration"]`, added by `mark_calibration`), then
    `(x - mean) / max(std, eps)` is applied to ALL of that patient's windows, calibration or
    not. No other patient's data is ever read (leakage rule L5) -- this is a per-patient
    transform, not a fit-on-the-whole-cohort one.
    """
    out = df.copy()

    for patient, group in df.groupby("patient", sort=False):
        calib = group.loc[group["is_calibration"], feature_cols]
        if calib.empty:
            logger.warning(
                "Patient %s has no calibration windows; leaving their features unstandardised",
                patient,
            )
            continue
        mean = calib.mean()
        std = calib.std(ddof=0).clip(lower=eps)
        out.loc[group.index, feature_cols] = (group[feature_cols] - mean) / std

    return out


def apply_arm(df: pd.DataFrame, arm: str, cfg: dict) -> pd.DataFrame:
    """Arm `raw` or `subject_standardised`; both share the same `is_calibration` mask.

    `raw`: `mark_calibration` only, features unchanged. `subject_standardised`:
    `mark_calibration` then `standardise_per_patient`. Both start from the same call to
    `mark_calibration(df, cfg)`, so both return the same rows in the same order with an
    identical `is_calibration` mask -- `run_loso` scores exactly the same windows either way,
    the only difference is the feature values.
    """
    if arm not in ("raw", "subject_standardised"):
        raise ValueError(f"Unknown arm {arm!r}; expected 'raw' or 'subject_standardised'")

    marked = mark_calibration(df, cfg)
    if arm == "raw":
        return marked

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    return standardise_per_patient(marked, feature_cols)