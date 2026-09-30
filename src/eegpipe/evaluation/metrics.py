"""Window-level metrics (owner: Member C, Step 10). Contract C7."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true, y_score, y_pred, step_s: float) -> dict:
    """Window-level classification metrics for one held-out patient (a Contract C7 row).

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth window labels for the scored (non-calibration) windows of one patient.
    y_score : array-like of float
        Ranking score used for AUC only (SVM: `decision_function`; RF: `predict_proba[:, 1]`).
    y_pred : array-like of {0, 1}
        The model's own hard decision (`predict`), used for the confusion matrix. Never
        derived from `y_score` here -- thresholding is the model's job, not this function's.
    step_s : float
        Seconds between consecutive window starts (`window_s * (1 - overlap)`), used to turn
        false positives into a rate.

    Returns
    -------
    dict with keys `n_windows, n_ictal, tp, fp, tn, fn, sensitivity, specificity, f1, auc,
    fa_per_hour`. A rate metric is `nan` (never a crash, never a silently wrong number) when
    the class it depends on is absent from `y_true` -- for example `sensitivity` and `auc`
    when the held-out patient has zero ictal windows.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred).astype(int)

    n_windows = int(y_true.shape[0])
    n_ictal = int((y_true == 1).sum())
    n_interictal = n_windows - n_ictal

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = (tp / n_ictal) if n_ictal > 0 else float("nan")
    specificity = (tn / n_interictal) if n_interictal > 0 else float("nan")

    f1_denom = 2 * tp + fp + fn
    f1 = (2 * tp / f1_denom) if (n_ictal > 0 and f1_denom > 0) else float("nan")

    if n_ictal > 0 and n_interictal > 0:
        auc = float(roc_auc_score(y_true, y_score))
    else:
        auc = float("nan")

    fa_per_hour = (fp / (n_windows * step_s / 3600)) if (n_windows > 0 and step_s > 0) else float("nan")

    return {
        "n_windows": n_windows,
        "n_ictal": n_ictal,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "fa_per_hour": fa_per_hour,
    }


def aggregate_metrics(per_patient: pd.DataFrame) -> pd.DataFrame:
    """Summarise a Contract-C7 per-patient table into mean/std/median per metric.

    Every numeric column except `patient` is treated as a metric. NaNs (a metric undefined
    for a given patient, e.g. no ictal windows) are ignored rather than propagated: how many
    were ignored is logged per metric, and `n_patients` in the output is the number of
    patients that actually contributed to that metric's summary (not the table's row count).
    `std` is the sample standard deviation (ddof=1); it is `nan` when fewer than two patients
    contribute.

    Returns
    -------
    DataFrame with columns `metric, mean, std, median, n_patients`, one row per metric.
    """
    metric_cols = [
        c for c in per_patient.columns if c != "patient" and pd.api.types.is_numeric_dtype(per_patient[c])
    ]

    rows = []
    for metric in metric_cols:
        values = per_patient[metric].to_numpy(dtype=float)
        finite = values[~np.isnan(values)]
        n_ignored = len(values) - len(finite)
        if n_ignored:
            logger.warning("%s: ignoring %d/%d NaN patient value(s) in aggregate", metric, n_ignored, len(values))
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
                "median": float(np.median(finite)) if len(finite) else float("nan"),
                "n_patients": int(len(finite)),
            }
        )
    return pd.DataFrame(rows, columns=["metric", "mean", "std", "median", "n_patients"])