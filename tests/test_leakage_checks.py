"""Tests for eegpipe.evaluation.leakage_checks and the basic LOSO loop guarantees."""
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
from eegpipe.evaluation.loso import feature_columns, run_loso
from eegpipe.models.normalisation import apply_arm
from tests.fixtures.synth_features import make_synthetic_features, make_test_config

PATIENT_MAP = {"chb21": "chb01"}


@pytest.fixture
def cfg(tmp_path):
    return make_test_config(tmp_path, tuning_mode="fixed")


@pytest.fixture
def df(cfg):
    return make_synthetic_features(cfg, n_patients=6, effect=1.5, patient_shift=0.5, seed=1)


# ---------------------------------------------------------------- L1 / C5 guards
def test_patient_disjoint_passes_and_fires():
    assert_patient_disjoint({"chb02", "chb03"}, {"chb04"})
    with pytest.raises(AssertionError):
        assert_patient_disjoint({"chb01", "chb02"}, {"chb02"})


def test_patient_grouping_accepts_mapped_data(df):
    assert_patient_grouping(df, PATIENT_MAP)
    assert set(df.loc[df.case == "chb21", "patient"]) == {"chb01"}  # fixture really has two cases


def test_patient_grouping_fires_on_chb21_as_patient(df):
    bad = df.copy()
    bad.loc[bad.case == "chb21", "patient"] = "chb21"
    with pytest.raises(AssertionError, match="chb21"):
        assert_patient_grouping(bad, PATIENT_MAP)


def test_patient_grouping_fires_when_case_maps_to_two_patients(df):
    bad = df.copy()
    first = bad.index[(bad.case == "chb02")][:5]
    bad.loc[first, "patient"] = "chb99"
    with pytest.raises(AssertionError):
        assert_patient_grouping(bad, PATIENT_MAP)


def test_finite_features_fires_on_nan_and_inf(df):
    cols = feature_columns(df)
    assert_finite_features(df, cols)
    bad = df.copy()
    bad.loc[3, cols[7]] = np.nan
    with pytest.raises(AssertionError, match="non-finite"):
        assert_finite_features(bad, cols)
    bad = df.copy()
    bad.loc[5, cols[100]] = np.inf
    with pytest.raises(AssertionError):
        assert_finite_features(bad, cols)


def test_no_calibration_scored_fires():
    pred = pd.DataFrame({"case": ["chb01", "chb01"], "file": ["chb01_01.edf"] * 2,
                         "start_sample": [0, 512]})
    calib_ok = pd.DataFrame({"case": ["chb01"], "file": ["chb01_01.edf"], "start_sample": [1024]})
    assert_no_calibration_scored(pred, calib_ok)
    calib_bad = pd.DataFrame({"case": ["chb01"], "file": ["chb01_01.edf"], "start_sample": [512]})
    with pytest.raises(AssertionError):
        assert_no_calibration_scored(pred, calib_bad)
    with pytest.raises(AssertionError):
        assert_no_calibration_scored(pred, [("chb01", "chb01_01.edf", 0)])


# ---------------------------------------------------------------- L8 label shuffle
@pytest.mark.parametrize("model_name", ["svm", "rf"])
def test_label_shuffle_auc_near_chance(df, cfg, model_name):
    prepared = apply_arm(df, "raw", cfg)
    auc = label_shuffle_check(prepared, feature_columns(prepared), model_name, cfg, n_test_patients=3, seed=0)
    assert 0.35 <= auc <= 0.65


# ---------------------------------------------------------------- LOSO loop basics
def test_loop_returns_one_row_per_heldout_patient(df, cfg):
    prepared = apply_arm(df, "raw", cfg)
    pred, per_patient = run_loso(prepared, feature_columns(prepared), "svm", cfg, "raw")
    assert sorted(per_patient["patient"]) == sorted(df["patient"].unique())
    assert len(per_patient) == df["patient"].nunique()
    assert set(pred["patient"]) == set(df["patient"])
    assert list(per_patient.columns)[:2] == ["patient", "n_windows"]


def test_heldout_patient_scored_at_natural_class_balance(df, cfg):
    prepared = apply_arm(df, "raw", cfg)
    pred, per_patient = run_loso(prepared, feature_columns(prepared), "svm", cfg, "raw")
    for _, row in per_patient.iterrows():
        natural = prepared[(prepared.patient == row["patient"]) & ~prepared.is_calibration]
        assert row["n_windows"] == len(natural)               # nothing resampled or dropped
        assert row["n_ictal"] == int(natural["label"].sum())  # ictal count untouched
        got = pred[pred.patient == row["patient"]]["y_true"].sum()
        assert got == row["n_ictal"]


def test_predictions_follow_contract_c6_and_exclude_calibration(df, cfg):
    prepared = apply_arm(df, "raw", cfg)
    pred, _ = run_loso(prepared, feature_columns(prepared), "rf", cfg, "raw")
    assert list(pred.columns) == ["patient", "case", "file", "start_sample", "y_true", "y_score",
                                  "y_pred", "arm", "model"]
    assert str(pred["y_true"].dtype) == "int8" and str(pred["y_pred"].dtype) == "int8"
    assert set(pred["y_score"].between(0, 1)) == {True}          # RF score = probability
    assert (pred["arm"] == "raw").all() and (pred["model"] == "rf").all()
    calib = prepared.loc[prepared.is_calibration, ["case", "file", "start_sample"]]
    assert_no_calibration_scored(pred, calib)
    assert len(pred) == int((~prepared.is_calibration).sum())    # every non-calibration window scored


def test_missing_is_calibration_column_means_all_scored(df, cfg):
    small = df[df.patient.isin(["chb02", "chb03", "chb04"])]
    pred, per_patient = run_loso(small, feature_columns(small), "svm", cfg, "raw", patients=["chb02"])
    assert len(pred) == int((small.patient == "chb02").sum())
    assert list(per_patient["patient"]) == ["chb02"]


def test_loso_rejects_unknown_heldout_and_single_patient(df, cfg):
    with pytest.raises(ValueError):
        run_loso(df, feature_columns(df), "svm", cfg, "raw", patients=["chb77"])
    one = df[df.patient == "chb02"]
    with pytest.raises(ValueError):
        run_loso(one, feature_columns(one), "svm", cfg, "raw")
