"""Leave-one-subject-out loop, sampling and tuning (owner: Member C, Steps 10-11).

`run_loso` is the harness; `subsample_training` and `tune_model` are the two pieces Step 11
adds (Step 10 used a hardcoded fixed model, now replaced by `models.classifiers.make_model`).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from eegpipe.evaluation.metrics import compute_metrics
from eegpipe.models.classifiers import make_model
from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _score(model: Pipeline, X: np.ndarray, model_name: str) -> np.ndarray:
    """RF: probability of class 1. SVM: `decision_function` value. `y_pred` uses `.predict`."""
    if model_name == "rf":
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _sample_per_patient(
    pool: pd.DataFrame, target_total: int, rng: np.random.Generator, min_per_patient: int = 0
) -> pd.DataFrame:
    """Sample up to `target_total` rows from `pool`, allocated across `patient` groups
    proportional to each patient's share of `pool`, with an optional per-patient floor.

    A floor above 0 can make the realised total slightly exceed `target_total` (a patient
    that would round to 0 under strict proportionality still gets `min_per_patient`, if they
    have that many rows available) -- that trade-off is deliberate, see `subsample_training`.
    """
    counts = pool.groupby("patient").size()
    total = int(counts.sum())
    target_total = min(max(target_total, 0), total)
    if target_total >= total:
        return pool

    raw_alloc = counts / total * target_total
    alloc = np.floor(raw_alloc).astype(int)
    if min_per_patient:
        alloc = np.maximum(alloc, np.minimum(min_per_patient, counts))

    shortfall = target_total - alloc.sum()
    if shortfall > 0:
        fractional = (raw_alloc - alloc).sort_values(ascending=False)
        headroom = counts - alloc
        for patient in fractional.index:
            if shortfall <= 0:
                break
            if headroom[patient] > 0:
                alloc[patient] += 1
                shortfall -= 1
    alloc = alloc.clip(upper=counts)

    parts = []
    for patient, n in alloc.items():
        if n <= 0:
            continue
        patient_pool = pool[pool["patient"] == patient]
        idx = rng.choice(patient_pool.index, size=int(n), replace=False)
        parts.append(patient_pool.loc[idx])
    return pd.concat(parts) if parts else pool.iloc[0:0]


def subsample_training(df: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Balance TRAINING data only (leakage rule L3).

    Keeps every ictal window, samples interictal windows at
    `cfg["evaluation"]["train_sampling"]["interictal_to_ictal_ratio"]` (allocated per patient,
    proportional to that patient's own interictal count). If the combined total would exceed
    `max_train_windows`, both classes are scaled down together (preserving the ratio) --
    except that every patient keeps at least one ictal window even if strict proportional
    scaling would round them down to zero.

    It is the caller's responsibility (`run_loso`) to only ever call this on a training split;
    this function has no way to know which patient is held out, and must not be given one.
    """
    rng = np.random.default_rng(seed)
    ratio = cfg["evaluation"]["train_sampling"]["interictal_to_ictal_ratio"]
    max_total = cfg["evaluation"]["train_sampling"]["max_train_windows"]

    ictal = df[df["label"] == 1]
    interictal = df[df["label"] == 0]
    if ictal.empty:
        return df  # nothing to anchor the ratio to; leave the caller's data untouched

    target_interictal = int(round(len(ictal) * ratio))
    sampled_interictal = _sample_per_patient(interictal, target_interictal, rng)

    total = len(ictal) + len(sampled_interictal)
    if total > max_total:
        scale = max_total / total
        ictal = _sample_per_patient(
            ictal, max(1, int(round(len(ictal) * scale))), rng, min_per_patient=1
        )
        sampled_interictal = _sample_per_patient(
            sampled_interictal, max(0, int(round(len(sampled_interictal) * scale))), rng
        )

    return pd.concat([ictal, sampled_interictal]).sort_index()


def tune_model(pipe: Pipeline, grid: dict, X, y, groups, cfg: dict, seed: int):
    """Fit `pipe`, tuned per `cfg["evaluation"]["tuning"]["mode"]`. Returns the fitted estimator.

    `mode == "nested"`: `GridSearchCV` over `grid`, with `GroupKFold` on `groups` (the training
    windows' patient ids) so a hyperparameter is never chosen using a fold that contains a
    window from the same patient it's validated on (leakage rule L4). `inner_splits` is capped
    at the number of distinct training patients, so this can't ask for more folds than groups.

    `mode == "fixed"`: skip the search and set `cfg["evaluation"]["fixed_params"][name]`
    directly (the fallback when nested tuning is too slow; `name` is read off the pipeline's
    own `clf` step, not passed in, since `tune_model`'s signature doesn't take it).

    Either way, the scaler and selector live inside `pipe` already (see `make_model`), so they
    are refit on training data only, per fold (leakage rule L2) -- this function doesn't need
    to do anything extra to guarantee that.
    """
    mode = cfg["evaluation"]["tuning"]["mode"]

    if mode == "fixed":
        name = "svm" if isinstance(pipe.named_steps["clf"], SVC) else "rf"
        pipe.named_steps["clf"].set_params(**cfg["evaluation"]["fixed_params"][name])
        pipe.fit(X, y)
        return pipe

    if mode == "nested":
        n_groups = len(np.unique(groups))
        inner_splits = min(cfg["evaluation"]["tuning"]["inner_splits"], n_groups)
        cv = GroupKFold(n_splits=inner_splits)
        search = GridSearchCV(pipe, grid, cv=cv, scoring=cfg["evaluation"]["tuning"]["scoring"], refit=True)
        search.fit(X, y, groups=groups)
        return search.best_estimator_

    raise ValueError(f"Unknown evaluation.tuning.mode {mode!r}; expected 'nested' or 'fixed'")


def run_loso(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    cfg: dict,
    arm: str,
    patients: list[str] | None = None,
    shuffle_train_labels: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-subject-out evaluation loop.

    For each held-out patient: subsample the rest of the cohort's windows for training (L3
    balancing; see `subsample_training`), tune/fit a model (see `tune_model`), then score this
    patient's own non-calibration windows at their **natural class balance** (L3 -- the
    held-out set is never resampled). `arm` and `model_name` are not applied here; they only
    label the output rows (the caller is expected to have already run
    `models.normalisation.apply_arm` to produce the feature values for the arm being
    evaluated).

    Parameters
    ----------
    df : pd.DataFrame
        Contract C5 (+ optional `is_calibration` bool column from `mark_calibration`; a
        missing column is treated as all False, i.e. no calibration windows to exclude).
    feature_cols : list[str]
        Usually `[c for c in df.columns if c.startswith("f_")]`.
    model_name : {"svm", "rf"}
    arm : str
        Label only, e.g. "raw" or "subject_standardised" -- copied into the output tables.
    patients : list[str], optional
        Restrict which patients are held out (used by `label_shuffle_check` for speed).
        Training always uses every *other* patient in the full `df`, regardless of this list.
    shuffle_train_labels : bool
        L8 sanity check: permute the (post-subsampling) training labels only, never the test
        labels, before fitting -- a real signal should disappear (AUC near 0.5) if nothing
        else leaks.

    Returns
    -------
    (predictions, per_patient) : Contract C6 and C7 DataFrames.
    """
    from eegpipe.evaluation.leakage_checks import assert_patient_disjoint  # local: avoids a
    # circular import (leakage_checks.label_shuffle_check calls run_loso).

    if "is_calibration" not in df.columns:
        df = df.assign(is_calibration=False)

    held_patients = patients if patients is not None else sorted(df["patient"].unique())
    base_seed = int(cfg.get("project", {}).get("seed", 0))

    pred_chunks: list[pd.DataFrame] = []
    per_patient_rows: list[dict] = []

    for i, held in enumerate(held_patients):
        train_df = df[df["patient"] != held]
        test_df = df[(df["patient"] == held) & (~df["is_calibration"])]

        assert_patient_disjoint(train_df["patient"].unique(), [held])

        if test_df.empty:
            logger.warning("Patient %s has no scored (non-calibration) windows; skipping", held)
            continue

        seed_i = base_seed + i
        t0 = time.perf_counter()

        train_df = subsample_training(train_df, cfg, seed_i)
        y_train = train_df["label"].to_numpy()
        if shuffle_train_labels:
            y_train = np.random.default_rng(seed_i).permutation(y_train)
        X_train = train_df[feature_cols].to_numpy(dtype=float)
        groups_train = train_df["patient"].to_numpy()

        pipe, grid = make_model(model_name, cfg, n_features=len(feature_cols), seed=seed_i)
        model = tune_model(pipe, grid, X_train, y_train, groups_train, cfg, seed_i)

        X_test = test_df[feature_cols].to_numpy(dtype=float)
        y_true = test_df["label"].to_numpy()
        y_score = _score(model, X_test, model_name)
        y_pred = model.predict(X_test)

        logger.info(
            "Fold %s (%d/%d, %s/%s): %d train windows, fit+score in %.1fs",
            held, i + 1, len(held_patients), model_name, arm, len(train_df),
            time.perf_counter() - t0,
        )

        pred_chunk = test_df[["patient", "case", "file", "start_sample"]].copy()
        pred_chunk["y_true"] = y_true.astype("int8")
        pred_chunk["y_score"] = y_score.astype(float)
        pred_chunk["y_pred"] = np.asarray(y_pred).astype("int8")
        pred_chunk["arm"] = arm
        pred_chunk["model"] = model_name
        pred_chunks.append(pred_chunk)

        step_s = float(cfg["segmentation"]["window_s"]) * (1 - float(cfg["segmentation"]["overlap"]))
        metrics = compute_metrics(y_true, y_score, y_pred, step_s)
        per_patient_rows.append({"patient": held, **metrics})

    predictions = pd.concat(pred_chunks, ignore_index=True) if pred_chunks else pd.DataFrame(
        columns=["patient", "case", "file", "start_sample", "y_true", "y_score", "y_pred", "arm", "model"]
    )
    per_patient = pd.DataFrame(per_patient_rows)
    return predictions, per_patient