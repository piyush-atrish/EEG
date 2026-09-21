"""Calibration-window marking and per-patient standardisation (the two arms).

Arm ``raw``                  features unchanged.
Arm ``subject_standardised`` each patient's features are standardised with the
                             mean and SD of that patient's *own* early
                             interictal calibration windows (rule L5).

In both arms the same calibration windows are flagged (``is_calibration``) and
are excluded from scoring, so both arms are scored on identical windows.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

ARMS = ("raw", "subject_standardised")


def mark_calibration(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add a boolean ``is_calibration`` column.

    Per patient: order windows by ``(case, file, start_sample)``, find the first
    ictal window, and among the interictal windows *before* it flag the first
    ``n = ceil(minutes * 60 / step_s)`` (300 with the default 10 min and 2 s
    step). If fewer than ``min_windows`` such windows exist (or the patient has
    no ictal window at all and fewer than ``min_windows`` interictal windows),
    fall back to the first ``n`` interictal windows overall and log a warning.

    Parameters
    ----------
    df : DataFrame
        Needs ``patient, case, file, start_sample, label``.
    cfg : dict

    Returns
    -------
    DataFrame
        A copy of ``df`` (original row order and index preserved) with the new
        ``is_calibration`` column.
    """
    cal = cfg["evaluation"]["calibration"]
    seg = cfg["segmentation"]
    step_s = float(seg["window_s"]) * (1.0 - float(seg["overlap"]))
    n_target = math.ceil(float(cal["minutes"]) * 60.0 / step_s)
    min_windows = int(cal["min_windows"])

    out = df.copy()
    mask = np.zeros(len(out), dtype=bool)
    order_cols = ["case", "file", "start_sample"]
    patient_arr = out["patient"].to_numpy()
    for patient in sorted(set(patient_arr)):
        rows = np.flatnonzero(patient_arr == patient)  # positions of this patient's windows
        sub = out.iloc[rows][order_cols + ["label"]].reset_index(drop=True)
        sub = sub.sort_values(order_cols, kind="stable")
        order = sub.index.to_numpy()  # chronological order, as positions within `rows`
        label = sub["label"].to_numpy()
        pos = np.arange(len(sub))
        ictal_pos = np.flatnonzero(label == 1)
        inter_pos = pos[label == 0]
        before = inter_pos[inter_pos < ictal_pos[0]] if len(ictal_pos) else inter_pos
        if len(before) >= min_windows:
            chosen = before[:n_target]
        else:
            chosen = inter_pos[:n_target]
            logger.warning(
                "patient %s: only %d interictal windows before the first seizure (< min_windows=%d); "
                "using the first %d interictal windows overall for calibration",
                patient, len(before), min_windows, len(chosen))
        mask[rows[order[chosen]]] = True
    out["is_calibration"] = mask
    return out


def standardise_per_patient(df: pd.DataFrame, feature_cols: list[str], eps: float = 1e-8) -> pd.DataFrame:
    """Standardise each patient's features using that patient's calibration windows only.

    For every patient: mean and SD (ddof=0) are computed from the rows with
    ``is_calibration == True`` of *that patient alone* (no labels are used to
    compute them, no other patient is involved: rule L5), then
    ``(x - mean) / max(std, eps)`` is applied to **all** windows of that
    patient. Features that are (near-)constant during calibration are logged
    because ``eps`` then amplifies later deviations.

    Returns
    -------
    DataFrame
        Copy of ``df`` with standardised feature columns (original dtype kept).
    """
    if "is_calibration" not in df.columns:
        raise ValueError("call mark_calibration first: 'is_calibration' column is missing")
    out = df.copy()
    patient_arr = out["patient"].to_numpy()
    calib_arr = out["is_calibration"].to_numpy(dtype=bool)
    col_idx = [out.columns.get_loc(c) for c in feature_cols]
    dtype = out[feature_cols].dtypes.iloc[0]
    for patient in sorted(set(patient_arr)):
        rows = np.flatnonzero(patient_arr == patient)
        block = out.iloc[rows, col_idx].to_numpy(dtype=np.float64)
        calib = block[calib_arr[rows]]
        if len(calib) == 0:
            raise ValueError(f"patient {patient} has no calibration windows")
        mean = calib.mean(axis=0)
        std = calib.std(axis=0)  # ddof=0
        n_flat = int((std < eps).sum())
        if n_flat:
            logger.warning("patient %s: %d feature(s) are constant in the calibration windows "
                           "(std < eps=%g)", patient, n_flat, eps)
        out.iloc[rows, col_idx] = ((block - mean) / np.maximum(std, eps)).astype(dtype)
    return out


def apply_arm(df: pd.DataFrame, arm: str, cfg: dict) -> pd.DataFrame:
    """Prepare ``df`` for one experimental arm.

    * ``raw``: :func:`mark_calibration` only (features unchanged).
    * ``subject_standardised``: :func:`mark_calibration`, then
      :func:`standardise_per_patient`.

    Both arms return the same rows and the same ``is_calibration`` mask, so both
    are scored on identical windows.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    marked = mark_calibration(df, cfg)
    if arm == "raw":
        return marked
    feature_cols = [c for c in marked.columns if c.startswith("f_")]
    return standardise_per_patient(marked, feature_cols)
