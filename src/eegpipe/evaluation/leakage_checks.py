"""Automated guards for the leakage rules L1-L8 (see README Section 2).

Every function raises :class:`AssertionError` (explicitly, so it also works
under ``python -O``) when a rule is violated.
"""
from __future__ import annotations

import copy
from typing import Iterable

import numpy as np
import pandas as pd

from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)


def assert_patient_disjoint(train_patients: Iterable[str], test_patients: Iterable[str]) -> None:
    """L1: train and test patient sets must not overlap."""
    overlap = set(train_patients) & set(test_patients)
    if overlap:
        raise AssertionError(f"L1 violated: patients in both train and test: {sorted(overlap)}")


def assert_patient_grouping(df: pd.DataFrame, patient_map: dict) -> None:
    """L1: LOSO groups by ``patient``, never by ``case``.

    * No value of ``df['patient']`` may be a key of ``patient_map`` (for
      example ``chb21`` must never appear as a patient).
    * Every ``case`` maps to exactly one ``patient``, and cases listed in
      ``patient_map`` map to their mapped patient.
    """
    bad = set(df["patient"].unique()) & set(patient_map)
    if bad:
        raise AssertionError(
            f"L1 violated: {sorted(bad)} appear in the 'patient' column but are mapped "
            f"to {[patient_map[b] for b in sorted(bad)]}; LOSO must group by patient, not case"
        )
    per_case = df.groupby("case")["patient"].nunique()
    multi = per_case[per_case > 1]
    if len(multi):
        raise AssertionError(f"L1 violated: cases mapping to more than one patient: {list(multi.index)}")
    for case, patient in patient_map.items():
        present = set(df.loc[df["case"] == case, "patient"].unique())
        if present and present != {patient}:
            raise AssertionError(
                f"L1 violated: case {case} should map to patient {patient}, found {sorted(present)}"
            )


def assert_finite_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Contract C5: no NaN or inf in any feature column."""
    if not feature_cols:
        raise AssertionError("no feature columns (columns starting with 'f_') were found")
    bad_cols: list[str] = []
    for i in range(0, len(feature_cols), 64):  # chunked: the real table is large
        chunk = feature_cols[i:i + 64]
        finite = np.isfinite(df[chunk].to_numpy(dtype=np.float64))
        bad_cols.extend(c for c, ok in zip(chunk, finite.all(axis=0)) if not ok)
    if bad_cols:
        raise AssertionError(
            f"non-finite feature values in {len(bad_cols)} column(s), first few: {bad_cols[:5]}"
        )


def assert_no_calibration_scored(pred_df: pd.DataFrame, calibration_keys) -> None:
    """Calibration windows must never appear in the scored predictions.

    Parameters
    ----------
    pred_df : DataFrame
        Predictions (contract C6) with columns ``case, file, start_sample``.
    calibration_keys : DataFrame or iterable of tuples
        Calibration windows identified by ``(case, file, start_sample)``.
    """
    if isinstance(calibration_keys, pd.DataFrame):
        keys = set(zip(calibration_keys["case"], calibration_keys["file"],
                       calibration_keys["start_sample"].astype(int)))
    else:
        keys = {(c, f, int(s)) for c, f, s in calibration_keys}
    scored = set(zip(pred_df["case"], pred_df["file"], pred_df["start_sample"].astype(int)))
    leaked = keys & scored
    if leaked:
        raise AssertionError(f"{len(leaked)} calibration window(s) were scored, e.g. {sorted(leaked)[:3]}")


def label_shuffle_check(df: pd.DataFrame, feature_cols: list[str], model_name: str, cfg: dict,
                        n_test_patients: int = 3, seed: int = 0) -> float:
    """L8 sanity gate: LOSO with *training* labels shuffled must give AUC ~ 0.5.

    The held-out patient's labels are never shuffled (they are only used for
    scoring). If this returns a value far from 0.5, information is leaking from
    the test patient into training.

    Returns
    -------
    float
        Mean AUC over ``n_test_patients`` randomly chosen held-out patients
        (NaN AUCs ignored).
    """
    from eegpipe.evaluation.loso import run_loso  # local import: loso imports this module

    cfg = copy.deepcopy(cfg)
    cfg["project"]["seed"] = int(seed)
    patients = sorted(df["patient"].unique())
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(patients, size=min(n_test_patients, len(patients)), replace=False).tolist())
    _, per_patient = run_loso(df, feature_cols, model_name, cfg, arm="label_shuffle",
                              patients=chosen, shuffle_train_labels=True)
    auc = float(np.nanmean(per_patient["auc"].to_numpy(dtype=float)))
    logger.info("label_shuffle_check(%s): held-out=%s mean AUC=%.3f", model_name, chosen, auc)
    return auc
