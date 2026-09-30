"""Tests for evaluation/leakage_checks.py and the Step 10 LOSO loop (Member C)."""

import numpy as np
import pandas as pd
import pytest

from eegpipe.evaluation.leakage_checks import (
    assert_finite_features,
    assert_no_calibration_scored,
    assert_patient_disjoint,
    assert_patient_grouping,
    label_shuffle_check,
)
from eegpipe.evaluation.loso import run_loso
from tests.fixtures.synth_features import make_synthetic_features


@pytest.fixture(scope="module")
def cfg(repo_cfg):
    return repo_cfg


@pytest.fixture(scope="module")
def df(cfg):
    return make_synthetic_features(
        cfg, n_patients=5, windows_per_patient=400, effect=1.5, patient_shift=0.0, seed=0
    )


@pytest.fixture(scope="module")
def feature_cols(df):
    return [c for c in df.columns if c.startswith("f_")]


# ---------------------------------------------------------------- L1: patient grouping


def test_disjointness_passes_for_disjoint_sets():
    assert_patient_disjoint(["chb01", "chb02"], ["chb03"])


def test_disjointness_fires_when_patients_overlap():
    with pytest.raises(AssertionError, match="chb02"):
        assert_patient_disjoint(["chb01", "chb02"], ["chb02", "chb03"])


def test_grouping_passes_on_clean_data(df, cfg):
    assert_patient_grouping(df, cfg["dataset"]["patient_map"])


def test_grouping_fires_on_chb21_used_as_patient(df, cfg):
    bad = df.copy()
    bad.loc[bad["case"] == "chb21", "patient"] = "chb21"
    with pytest.raises(AssertionError, match="chb21"):
        assert_patient_grouping(bad, cfg["dataset"]["patient_map"])


def test_grouping_fires_when_a_case_maps_to_two_patients(df, cfg):
    bad = df.copy()
    bad.loc[bad.index[0], "patient"] = "someone_else"
    with pytest.raises(AssertionError, match="more than one patient"):
        assert_patient_grouping(bad, cfg["dataset"]["patient_map"])


def test_synthetic_data_has_one_patient_with_two_cases(df):
    cases_per_patient = df.groupby("patient")["case"].nunique()
    assert cases_per_patient["chb01"] == 2
    assert (cases_per_patient.drop("chb01") == 1).all()


# ---------------------------------------------------------------- finite features


def test_finite_features_passes_and_fires(df, feature_cols):
    assert_finite_features(df, feature_cols)

    bad = df.copy()
    bad.loc[0, feature_cols[0]] = np.nan
    bad.loc[3, feature_cols[2]] = np.inf
    with pytest.raises(AssertionError, match="2 non-finite"):
        assert_finite_features(bad, feature_cols)


# ---------------------------------------------------------------- calibration exclusion


def _with_calibration(df, n=100):
    out = df.copy()
    out["is_calibration"] = False
    first = out.sort_values(["patient", "case", "file", "start_sample"]).groupby("patient").head(n)
    out.loc[first.index, "is_calibration"] = True
    return out


def test_no_calibration_scored_fires_on_overlap_and_passes_otherwise(df):
    marked = _with_calibration(df)
    key_cols = ["patient", "case", "file", "start_sample"]

    leaky = marked[key_cols].iloc[:5]  # first rows are calibration windows
    with pytest.raises(AssertionError, match="calibration window"):
        assert_no_calibration_scored(leaky, marked)

    clean = marked.loc[~marked["is_calibration"], key_cols]
    assert_no_calibration_scored(clean, marked)


def test_no_calibration_scored_accepts_key_tuples(df):
    marked = _with_calibration(df)
    key_cols = ["patient", "case", "file", "start_sample"]
    keys = set(map(tuple, marked.loc[marked["is_calibration"], key_cols].to_numpy()))
    with pytest.raises(AssertionError):
        assert_no_calibration_scored(marked[key_cols], keys)
    assert_no_calibration_scored(marked[key_cols], set())


# ---------------------------------------------------------------- LOSO loop properties


def test_loop_returns_one_row_per_held_out_patient(df, feature_cols, cfg):
    preds, per_patient = run_loso(df, feature_cols, "svm", cfg, arm="raw")

    assert len(per_patient) == df["patient"].nunique()
    assert sorted(per_patient["patient"]) == sorted(df["patient"].unique())
    assert per_patient["patient"].is_unique
    assert list(preds.columns) == [
        "patient", "case", "file", "start_sample", "y_true", "y_score", "y_pred", "arm", "model",
    ]
    assert (preds["arm"] == "raw").all() and (preds["model"] == "svm").all()


def test_held_out_patient_keeps_natural_class_ratio(df, feature_cols, cfg):
    """L3: the test set is never rebalanced; scored windows are exactly the patient's own."""
    preds, per_patient = run_loso(df, feature_cols, "svm", cfg, arm="raw")

    for patient, group in df.groupby("patient"):
        scored = preds[preds["patient"] == patient]
        assert len(scored) == len(group)
        assert scored["y_true"].mean() == pytest.approx(group["label"].mean())
        row = per_patient.set_index("patient").loc[patient]
        assert row["n_windows"] == len(group)
        assert row["n_ictal"] == group["label"].sum()


def test_calibration_windows_are_excluded_but_ratio_stays_natural(df, feature_cols, cfg):
    marked = _with_calibration(df)
    preds, per_patient = run_loso(marked, feature_cols, "svm", cfg, arm="raw")

    assert_no_calibration_scored(preds, marked)
    expected = marked[~marked["is_calibration"]]
    assert len(preds) == len(expected)
    for patient, group in expected.groupby("patient"):
        scored = preds[preds["patient"] == patient]
        assert scored["y_true"].mean() == pytest.approx(group["label"].mean())


def test_loop_trains_on_all_other_patients_even_when_restricted(df, feature_cols, cfg):
    preds, per_patient = run_loso(df, feature_cols, "svm", cfg, arm="raw", patients=["chb02"])
    assert set(preds["patient"]) == {"chb02"}
    assert per_patient["patient"].tolist() == ["chb02"]


def test_loop_is_deterministic(df, feature_cols, cfg):
    a, _ = run_loso(df, feature_cols, "svm", cfg, arm="raw", patients=["chb03"])
    b, _ = run_loso(df, feature_cols, "svm", cfg, arm="raw", patients=["chb03"])
    pd.testing.assert_frame_equal(a, b)


def test_loop_learns_real_signal(df, feature_cols, cfg):
    _, per_patient = run_loso(df, feature_cols, "svm", cfg, arm="raw")
    assert per_patient["auc"].mean() > 0.9


def test_unknown_model_name_raises(df, feature_cols, cfg):
    with pytest.raises(ValueError, match="Unknown model name"):
        run_loso(df, feature_cols, "xgboost", cfg, arm="raw", patients=["chb01"])


# ---------------------------------------------------------------- L8: label shuffle


@pytest.mark.parametrize("model_name", ["svm", "rf"])
def test_label_shuffle_auc_is_near_chance(df, feature_cols, cfg, model_name):
    auc = label_shuffle_check(df, feature_cols, model_name, cfg, n_test_patients=3, seed=0)
    assert 0.35 <= auc <= 0.65