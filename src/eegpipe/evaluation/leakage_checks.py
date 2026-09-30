"""Automated leakage guards L1-L8 (owner: Member C, Step 10).

Every check here protects the credibility of the whole project's baseline numbers. Each
function raises `AssertionError` with a message naming exactly what leaked and how -- these
are meant to fail loudly during development and CI, not to be caught and ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eegpipe.evaluation.loso import run_loso
from eegpipe.evaluation.metrics import aggregate_metrics

_KEY_COLS = ["patient", "case", "file", "start_sample"]


def assert_patient_disjoint(train_patients, test_patients) -> None:
    """L1: the same patient must never appear in both a training and a test set."""
    overlap = set(train_patients) & set(test_patients)
    assert not overlap, f"Patient(s) {sorted(overlap)} appear in both train and test (L1)"


def assert_patient_grouping(df: pd.DataFrame, patient_map: dict) -> None:
    """L1: `patient` must be the grouping key, never a raw case id like `chb21`.

    Two checks:
    * no value in the `patient` column is a key of `patient_map` (a duplicate-recording case
      id, e.g. `chb21`, must have already been mapped to its patient, e.g. `chb01`);
    * every `case` maps to exactly one `patient` (a case can't be split across patients).
    """
    leaked_case_ids = set(df["patient"].unique()) & set(patient_map.keys())
    assert not leaked_case_ids, (
        f"Case id(s) {sorted(leaked_case_ids)} found in the `patient` column -- "
        "these must be mapped via patient_map, not used as patient ids directly (L1)"
    )

    patients_per_case = df.groupby("case")["patient"].nunique()
    inconsistent_cases = patients_per_case[patients_per_case > 1]
    assert inconsistent_cases.empty, (
        f"Case(s) {list(inconsistent_cases.index)} map to more than one patient (L1)"
    )


def assert_finite_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Contract C5's own rule ('no NaN or inf'), re-checked wherever features are consumed."""
    values = df[feature_cols].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if finite.all():
        return
    bad_cols = [c for c, ok in zip(feature_cols, finite.all(axis=0)) if not ok]
    raise AssertionError(
        f"{int((~finite).sum())} non-finite feature value(s) found in {len(bad_cols)} "
        f"column(s), e.g. {bad_cols[:5]}"
    )


def assert_no_calibration_scored(pred_df: pd.DataFrame, calibration_keys) -> None:
    """A held-out patient's calibration windows must never appear in scored predictions.

    `calibration_keys` is either a DataFrame with an `is_calibration` column (the key set is
    extracted from the rows where it's True) or any iterable of
    `(patient, case, file, start_sample)` tuples identifying calibration windows directly.
    """
    if isinstance(calibration_keys, pd.DataFrame):
        mask = calibration_keys["is_calibration"]
        calibration_keys = set(map(tuple, calibration_keys.loc[mask, _KEY_COLS].to_numpy()))
    else:
        calibration_keys = set(calibration_keys)

    if not calibration_keys:
        return

    pred_keys = set(map(tuple, pred_df[_KEY_COLS].to_numpy()))
    overlap = pred_keys & calibration_keys
    assert not overlap, (
        f"{len(overlap)} calibration window(s) were scored in predictions, "
        f"e.g. {next(iter(overlap))}"
    )


def label_shuffle_check(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    cfg: dict,
    n_test_patients: int = 3,
    seed: int = 0,
) -> float:
    """L8: shuffling TRAINING labels should destroy predictive signal.

    Runs `run_loso` with `shuffle_train_labels=True` (arm `raw`, the most neutral choice),
    restricted to `n_test_patients` held-out patients chosen deterministically from `seed` --
    this is a fast sanity check, not a full run. Returns the mean AUC across those patients.

    Expect a result near 0.5: a model trained on shuffled labels has learned nothing real, so
    it should not separate the genuinely-labelled test patients any better than chance. A
    result far from 0.5 means labels are leaking into training some other way, and every
    other result in the project should be treated as suspect until that's found.
    """
    rng = np.random.default_rng(seed)
    all_patients = np.sort(df["patient"].unique())
    n_test_patients = min(n_test_patients, len(all_patients))
    test_patients = rng.choice(all_patients, size=n_test_patients, replace=False).tolist()

    _, per_patient = run_loso(
        df,
        feature_cols,
        model_name,
        cfg,
        arm="raw",
        patients=test_patients,
        shuffle_train_labels=True,
    )

    auc_summary = aggregate_metrics(per_patient)
    auc_row = auc_summary.loc[auc_summary["metric"] == "auc"]
    if auc_row.empty or int(auc_row["n_patients"].iloc[0]) == 0:
        return float("nan")
    return float(auc_row["mean"].iloc[0])