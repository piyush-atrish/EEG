"""Within-patient feature separability and cross-patient direction consistency.

For every patient and feature we compute the AUC of "this feature separates ictal from
interictal windows *within that patient*" (0.5 = no information, >0.5 = higher in seizures,
<0.5 = lower in seizures). Two summaries follow:

* **separability** -- mean |AUC - 0.5| over features: how much seizure information a patient's
  features carry at all;
* **agreement** -- among features with a strong, consistent direction in the *other* patients,
  the share whose direction in this patient matches (leave-one-out). A shared classifier can only
  work if seizures move the features in the same direction across patients; a patient with low
  agreement is expected to be a failure case for any cross-subject model.

The same numbers are the "contrast kept" diagnostic (M2) of docs/adaptive_design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CHUNK = 64  # features ranked at a time, to keep memory modest on 50k-window patients


def _auc_columns(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC (ictal vs interictal) of every column of ``values`` via average ranks."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ranks = rankdata(values, axis=0)
    return (ranks[y == 1].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def feature_auc_by_patient(
    df: pd.DataFrame, feature_cols: list[str], min_ictal: int = 30
) -> pd.DataFrame:
    """Table of shape (features, patients) of within-patient AUCs.

    Patients with fewer than ``min_ictal`` ictal windows, or no interictal windows, are omitted.
    NaNs in a feature are replaced by that patient's median for the feature before ranking.
    """
    columns: dict[str, np.ndarray] = {}
    for patient, group in df.groupby("patient", sort=True):
        y = group["label"].to_numpy(dtype=int)
        if y.sum() < min_ictal or y.sum() == len(y):
            continue
        values = np.array(group[feature_cols], dtype=np.float64)  # explicit copy: we edit it
        medians = np.nanmedian(values, axis=0)
        nan_rows, nan_cols = np.where(np.isnan(values))
        values[nan_rows, nan_cols] = medians[nan_cols]
        aucs = np.concatenate(
            [_auc_columns(values[:, i : i + CHUNK], y) for i in range(0, values.shape[1], CHUNK)]
        )
        columns[patient] = aucs
    return pd.DataFrame(columns, index=feature_cols)


def direction_summary(auc: pd.DataFrame, strong: float = 0.10) -> pd.DataFrame:
    """Per patient: separability, number of strong consensus features and direction agreement.

    The consensus for a patient is the median AUC of all *other* patients; a feature is strong when
    ``|consensus - 0.5| >= strong``. Needs at least three patients.
    """
    rows = []
    for patient in auc.columns:
        others = auc.drop(columns=patient)
        row = {"patient": patient, "separability": float((auc[patient] - 0.5).abs().mean())}
        if others.shape[1] >= 2:
            consensus = others.median(axis=1)
            mask = (consensus - 0.5).abs() >= strong
            agree = np.sign(auc.loc[mask, patient] - 0.5) == np.sign(consensus[mask] - 0.5)
            row["n_strong_features"] = int(mask.sum())
            row["agreement"] = float(agree.mean()) if mask.any() else float("nan")
        else:
            row["n_strong_features"], row["agreement"] = 0, float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("patient")
