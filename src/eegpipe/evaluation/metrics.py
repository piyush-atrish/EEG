"""Window-level metrics (contract C7) and their aggregation across patients.

All metrics are computed per held-out patient at that patient's natural class
balance. Undefined quantities (for example sensitivity for a patient with no
ictal windows) are returned as ``nan`` and never raise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Columns of the per-patient metrics table (contract C7), in order.
PER_PATIENT_COLUMNS = [
    "patient", "n_windows", "n_ictal", "tp", "fp", "tn", "fn",
    "sensitivity", "specificity", "f1", "auc", "fa_per_hour",
]

#: Metrics that are aggregated across patients in ``summary_all.csv``.
SUMMARY_METRICS = ["sensitivity", "specificity", "f1", "auc", "fa_per_hour"]


def compute_metrics(y_true, y_score, y_pred, step_s: float) -> dict:
    """Compute window-level metrics for one held-out patient.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth window labels.
    y_score : array-like of float
        Continuous score (higher = more ictal). Only used for AUC.
    y_pred : array-like of {0, 1}
        The model's own hard decisions (``predict``).
    step_s : float
        Window step in seconds (``window_s * (1 - overlap)``); converts the
        number of windows to hours for ``fa_per_hour``.

    Returns
    -------
    dict
        Keys ``n_windows, n_ictal, tp, fp, tn, fn, sensitivity, specificity,
        f1, auc, fa_per_hour``. Undefined metrics are ``nan``.

    Notes
    -----
    ``fa_per_hour = fp / (n_windows * step_s / 3600)`` is a window-based
    approximation on the *included* data only (seizure-free data is capped),
    so it is not a clinical false-alarm rate.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError("y_true, y_score and y_pred must have the same length")

    n = int(len(y_true))
    n_ictal = int((y_true == 1).sum())
    n_inter = n - n_ictal
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    nan = float("nan")
    sensitivity = tp / (tp + fn) if n_ictal > 0 else nan
    specificity = tn / (tn + fp) if n_inter > 0 else nan
    if n_ictal > 0:
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else nan
    else:
        f1 = nan  # no positives in the ground truth: F1 is not meaningful
    if n_ictal > 0 and n_inter > 0 and np.all(np.isfinite(y_score)):
        auc = float(roc_auc_score(y_true, y_score))
    else:
        auc = nan
    fa_per_hour = fp / (n * step_s / 3600.0) if n > 0 else nan

    return {
        "n_windows": n, "n_ictal": n_ictal,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sensitivity, "specificity": specificity,
        "f1": f1, "auc": auc, "fa_per_hour": fa_per_hour,
    }


def aggregate_metrics(per_patient: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-patient metrics into mean / SD / median per metric.

    NaN values are ignored per metric. ``n_patients`` is the number of
    patients that *contributed* (non-NaN) to that metric and ``n_ignored`` is
    the number that were dropped because the metric was undefined.

    Parameters
    ----------
    per_patient : pandas.DataFrame
        One row per held-out patient (contract C7 per-patient table).

    Returns
    -------
    pandas.DataFrame
        Columns ``metric, mean, std, median, n_patients, n_ignored``.
        ``std`` is the sample standard deviation across patients (ddof=1);
        it is ``nan`` when fewer than two patients contribute.
    """
    rows = []
    for metric in SUMMARY_METRICS:
        if metric not in per_patient.columns:
            continue
        vals = pd.to_numeric(per_patient[metric], errors="coerce")
        valid = vals.dropna()
        n_ignored = int(len(vals) - len(valid))
        if n_ignored:
            logger.info("aggregate_metrics: %s ignored %d NaN patient(s)", metric, n_ignored)
        rows.append({
            "metric": metric,
            "mean": float(valid.mean()) if len(valid) else float("nan"),
            "std": float(valid.std(ddof=1)) if len(valid) > 1 else float("nan"),
            "median": float(valid.median()) if len(valid) else float("nan"),
            "n_patients": int(len(valid)),
            "n_ignored": n_ignored,
        })
    return pd.DataFrame(rows, columns=["metric", "mean", "std", "median", "n_patients", "n_ignored"])
