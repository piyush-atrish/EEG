"""Leave-One-Subject-Out evaluation loop, training subsampling and nested tuning.

Leakage rules enforced here (README Section 2):

* L1  folds are formed by ``patient`` (``chb21`` is already mapped to ``chb01``).
* L2  the scaler and the feature selector live inside the sklearn ``Pipeline``.
* L3  only the *training* data is subsampled; the held-out patient is scored
      at its natural class balance.
* L4  hyper-parameters are tuned with ``GroupKFold`` grouped by patient, inside
      the training patients only.
* L7  every random choice is seeded from ``cfg["project"]["seed"]``.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from eegpipe.evaluation.leakage_checks import (
    assert_finite_features,
    assert_no_calibration_scored,
    assert_patient_disjoint,
)
from eegpipe.evaluation.metrics import PER_PATIENT_COLUMNS, compute_metrics
from eegpipe.models.classifiers import make_model
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.seed import set_global_seed

logger = get_logger(__name__)

PRED_COLUMNS = ["patient", "case", "file", "start_sample", "y_true", "y_score", "y_pred", "arm", "model"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns are always inferred as those starting with ``f_``."""
    return [c for c in df.columns if c.startswith("f_")]


def step_seconds(cfg: dict) -> float:
    """Window step in seconds (``window_s * (1 - overlap)``, 2 s by default)."""
    seg = cfg["segmentation"]
    return float(seg["window_s"]) * (1.0 - float(seg["overlap"]))


# --------------------------------------------------------------------------- #
# Training-set subsampling (L3: training data only)
# --------------------------------------------------------------------------- #
def _allocate(counts: np.ndarray, total: int) -> np.ndarray:
    """Integer quotas proportional to ``counts`` with sum ``min(total, counts.sum())``.

    Uses largest-remainder rounding and never gives a group more than it has.
    Deterministic (stable tie-breaking).
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = int(min(max(total, 0), counts.sum()))
    quotas = np.zeros_like(counts)
    remaining = total
    while remaining > 0:
        room = counts - quotas
        weight = counts * (room > 0)
        raw = remaining * weight / weight.sum()
        add = np.minimum(np.floor(raw).astype(np.int64), room)
        if add.sum() > 0:
            quotas += add
            remaining -= int(add.sum())
            continue
        for i in np.argsort(-(raw - np.floor(raw)), kind="stable"):
            if remaining == 0:
                break
            if room[i] > 0:
                quotas[i] += 1
                remaining -= 1
    return quotas


def subsample_training(df: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Subsample TRAINING windows to control class imbalance and training cost.

    Must only ever be called on training patients (L3); the held-out patient is
    never resampled. Needs only the ``patient`` and ``label`` columns, so a
    light metadata frame may be passed (the original index is preserved).

    * All ictal windows are kept, except when the size cap forces scaling down.
    * Interictal windows are sampled to ``interictal_to_ictal_ratio`` x (number
      of ictal windows kept), drawn proportionally to each patient's number of
      interictal windows.
    * If ``n_ictal * (1 + ratio)`` would exceed ``max_train_windows``, both
      classes are scaled down keeping the ratio. Every patient that has ictal
      windows keeps at least one.

    Returns
    -------
    pandas.DataFrame
        The selected rows of ``df`` (original order and index preserved).
    """
    ts = cfg["evaluation"]["train_sampling"]
    ratio = float(ts["interictal_to_ictal_ratio"])
    max_n = int(ts["max_train_windows"])
    rng = np.random.default_rng(seed)

    patients = np.asarray(sorted(df["patient"].unique()))
    pat = df["patient"].to_numpy()
    lab = df["label"].to_numpy()
    ictal_pos = {p: np.flatnonzero((pat == p) & (lab == 1)) for p in patients}
    inter_pos = {p: np.flatnonzero((pat == p) & (lab == 0)) for p in patients}
    n_ictal = np.array([len(ictal_pos[p]) for p in patients])
    n_inter = np.array([len(inter_pos[p]) for p in patients])

    # ---- ictal quotas
    if n_ictal.sum() * (1.0 + ratio) > max_n and n_ictal.sum() > 0:
        target = int(max_n // (1.0 + ratio))
        q_ictal = _allocate(n_ictal, target)
        q_ictal = np.where(n_ictal > 0, np.maximum(q_ictal, 1), 0)  # every patient keeps >= 1
    else:
        q_ictal = n_ictal.copy()
    ictal_kept = int(q_ictal.sum())

    # ---- interictal quotas
    if ictal_kept > 0:
        inter_target = int(min(round(ratio * ictal_kept), max_n - ictal_kept, n_inter.sum()))
    else:  # degenerate: no ictal windows in training; run_loso will refuse to fit
        inter_target = int(min(max_n, n_inter.sum()))
    q_inter = _allocate(n_inter, max(inter_target, 0))

    chosen = []
    for i, p in enumerate(patients):
        if q_ictal[i] > 0:
            chosen.append(rng.choice(ictal_pos[p], size=int(q_ictal[i]), replace=False))
        if q_inter[i] > 0:
            chosen.append(rng.choice(inter_pos[p], size=int(q_inter[i]), replace=False))
    positions = np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=np.int64)
    return df.iloc[positions]


# --------------------------------------------------------------------------- #
# Nested (patient-grouped) tuning (L4)
# --------------------------------------------------------------------------- #
def tune_model(pipe: Pipeline, grid: dict, X, y, groups, cfg: dict, seed: int):
    """Fit ``pipe``, tuning hyper-parameters by patient-grouped CV or fixed values.

    Parameters
    ----------
    pipe, grid
        As returned by :func:`eegpipe.models.classifiers.make_model`.
    X, y
        Training features and labels (training patients only).
    groups : array-like
        Patient id of every training window; used by ``GroupKFold`` (L4), so no
        patient is ever split between the inner train and validation folds.
    cfg : dict
        Reads ``evaluation.tuning`` and ``evaluation.fixed_params``.
    seed : int
        Reserved for symmetry with the other functions; ``GroupKFold`` is
        deterministic and the estimator already carries its own seed.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The fitted pipeline (scaler and selector fitted on training data only).
    """
    tun = cfg["evaluation"]["tuning"]
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    is_svm = isinstance(pipe.named_steps["clf"], SVC)
    name = "svm" if is_svm else "rf"

    def _fit_fixed() -> Pipeline:
        params = {f"clf__{k}": v for k, v in cfg["evaluation"]["fixed_params"][name].items()}
        return pipe.set_params(**params).fit(X, y)

    if tun["mode"] == "fixed":
        return _fit_fixed()
    if tun["mode"] != "nested":
        raise ValueError(f"unknown tuning mode {tun['mode']!r} (expected 'nested' or 'fixed')")
    if n_groups < 2:
        logger.warning("tune_model: only %d training patient(s); falling back to fixed parameters", n_groups)
        return _fit_fixed()

    cv = GroupKFold(n_splits=min(int(tun["inner_splits"]), n_groups))
    search = GridSearchCV(pipe, grid, cv=cv, scoring=tun["scoring"], refit=True,
                          n_jobs=1, error_score=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant-feature and undefined-AUC warnings inside inner folds
        try:
            search.fit(X, y, groups=groups)
        except Exception as exc:  # e.g. every inner fold had a single class -> all scores NaN
            logger.warning("tune_model: nested search failed (%s); falling back to fixed parameters", exc)
            return _fit_fixed()
    logger.info("tune_model(%s): best params %s (inner AUC %.3f)", name, search.best_params_,
                search.best_score_)
    return search.best_estimator_


# --------------------------------------------------------------------------- #
# LOSO loop
# --------------------------------------------------------------------------- #
def _score(fitted: Pipeline, X: np.ndarray, model_name: str) -> np.ndarray:
    """RF: P(class 1). SVM: signed distance from the decision function."""
    if model_name == "rf":
        classes = list(fitted.classes_)
        return fitted.predict_proba(X)[:, classes.index(1)]
    return fitted.decision_function(X)


def run_loso(df: pd.DataFrame, feature_cols: list[str], model_name: str, cfg: dict, arm: str,
             patients: list[str] | None = None,
             shuffle_train_labels: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-One-Subject-Out evaluation for one arm and one model.

    ``df`` must already be arm-prepared by the caller
    (:func:`eegpipe.models.normalisation.apply_arm`): ``arm`` here is only a
    label written to the outputs. A missing ``is_calibration`` column is
    treated as all False.

    For every held-out patient: train on all other patients (subsampled, tuned
    by patient-grouped CV), then score the held-out patient's non-calibration
    windows at their natural class balance.

    Parameters
    ----------
    df : DataFrame
        Contract C5 table (plus optional ``is_calibration``).
    feature_cols : list of str
        Feature columns (``f_*``).
    model_name : {"svm", "rf"}
    cfg : dict
    arm : str
        Label for the output tables.
    patients : list of str, optional
        Held-out patients to evaluate (default: every patient). Training always
        uses *all* other patients present in ``df``.
    shuffle_train_labels : bool
        L8 sanity mode: permute the labels of the training windows (never the
        test patient's).

    Returns
    -------
    (predictions, per_patient) : tuple of DataFrame
        Contract C6 (scored windows only) and the contract C7 per-patient table.
    """
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)
    step_s = step_seconds(cfg)

    if not df.index.equals(pd.RangeIndex(len(df))):
        df = df.reset_index(drop=True)
    if "is_calibration" not in df.columns:
        df = df.assign(is_calibration=False)
    assert_finite_features(df, feature_cols)

    all_patients = sorted(df["patient"].unique())
    if len(all_patients) < 2:
        raise ValueError("LOSO needs at least two patients")
    held_list = list(all_patients) if patients is None else sorted(patients)
    unknown = set(held_list) - set(all_patients)
    if unknown:
        raise ValueError(f"unknown held-out patients: {sorted(unknown)}")

    meta = df[["patient", "label"]]
    patient_arr = df["patient"].to_numpy()
    calib_arr = df["is_calibration"].to_numpy(dtype=bool)
    rng = np.random.default_rng(seed)

    pred_frames, metric_rows = [], []
    for held in held_list:
        t0 = time.time()
        train_mask = patient_arr != held
        test_mask = (patient_arr == held) & ~calib_arr
        train_patients = set(patient_arr[train_mask])
        assert_patient_disjoint(train_patients, {held})  # L1

        if not test_mask.any():
            logger.warning("held-out %s has no scorable windows after calibration exclusion; skipped", held)
            continue

        train_meta = meta[train_mask]
        if shuffle_train_labels:  # L8: permute TRAINING labels only
            train_meta = train_meta.assign(label=rng.permutation(train_meta["label"].to_numpy()))
        sub = subsample_training(train_meta, cfg, seed)  # L3: training data only
        if sub["label"].nunique() < 2:
            raise ValueError(f"fold {held}: training data has a single class; cannot fit")
        X_train = df.loc[sub.index, feature_cols].to_numpy(dtype=np.float32)
        y_train = sub["label"].to_numpy(dtype=int)
        groups = sub["patient"].to_numpy()

        pipe, grid = make_model(model_name, cfg, len(feature_cols), seed)
        fitted = tune_model(pipe, grid, X_train, y_train, groups, cfg, seed)  # L2, L4

        test = df[test_mask]  # natural class balance (L3)
        X_test = test[feature_cols].to_numpy(dtype=np.float32)
        y_true = test["label"].to_numpy(dtype=int)
        y_score = _score(fitted, X_test, model_name)
        y_pred = fitted.predict(X_test)

        pred_frames.append(pd.DataFrame({
            "patient": test["patient"].to_numpy(), "case": test["case"].to_numpy(),
            "file": test["file"].to_numpy(), "start_sample": test["start_sample"].to_numpy(),
            "y_true": y_true.astype(np.int8), "y_score": y_score.astype(float),
            "y_pred": np.asarray(y_pred).astype(np.int8), "arm": arm, "model": model_name,
        }))
        metric_rows.append({"patient": held, **compute_metrics(y_true, y_score, y_pred, step_s)})
        logger.info("LOSO %s/%s held-out %s: train=%d (ictal %d) test=%d (ictal %d) AUC=%.3f  %.1fs",
                    arm, model_name, held, len(y_train), int(y_train.sum()), len(y_true),
                    int(y_true.sum()), metric_rows[-1]["auc"], time.time() - t0)

    predictions = (pd.concat(pred_frames, ignore_index=True)[PRED_COLUMNS] if pred_frames
                   else pd.DataFrame(columns=PRED_COLUMNS))
    per_patient = pd.DataFrame(metric_rows, columns=PER_PATIENT_COLUMNS)

    if calib_arr.any() and len(predictions):  # calibration windows must never be scored
        assert_no_calibration_scored(predictions, df.loc[calib_arr, ["case", "file", "start_sample"]])
    return predictions, per_patient
