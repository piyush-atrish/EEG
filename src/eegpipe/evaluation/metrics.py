"""Window-level metrics (owner: Member C, Step 10). Contract C7."""

from __future__ import annotations

import pandas as pd


def compute_metrics(y_true, y_score, y_pred, step_s: float) -> dict:
    """Return n_windows, n_ictal, tp, fp, tn, fn, sensitivity, specificity, f1, auc, fa_per_hour."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def aggregate_metrics(per_patient: pd.DataFrame) -> pd.DataFrame:
    """Return mean, std, median and n_patients per metric."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")
