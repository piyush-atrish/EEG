"""Tests for models/normalisation.py (Member C, Step 12)."""

import numpy as np
import pandas as pd
import pytest

from eegpipe.models.normalisation import apply_arm, mark_calibration, standardise_per_patient
from tests.fixtures.synth_features import make_synthetic_features


@pytest.fixture(scope="module")
def cfg(repo_cfg):
    return repo_cfg


@pytest.fixture(scope="module")
def df(cfg):
    return make_synthetic_features(
        cfg, n_patients=6, windows_per_patient=600, effect=1.0, patient_shift=1.0, seed=0
    )


@pytest.fixture(scope="module")
def feature_cols(df):
    return [c for c in df.columns if c.startswith("f_")]


def test_calibration_marks_exactly_300_windows_before_first_seizure(df):
    marked = mark_calibration(df, {"evaluation": {"calibration": {"minutes": 10, "min_windows": 60}},
                                    "segmentation": {"window_s": 4.0, "overlap": 0.5}})
    for patient, group in marked.groupby("patient"):
        ordered = group.sort_values(["case", "file", "start_sample"])
        labels = ordered["label"].to_numpy()
        first_ictal = np.flatnonzero(labels == 1)[0]
        assert ordered["is_calibration"].sum() == 300
        assert ordered.iloc[:first_ictal]["is_calibration"].sum() == 300
        assert not ordered.iloc[first_ictal:]["is_calibration"].any()


def test_calibration_falls_back_when_seizure_is_too_early(df, cfg, caplog):
    early = df.copy()
    c3 = early[early["patient"] == "chb03"].sort_values(["case", "file", "start_sample"])
    early.loc[c3.index[:5], "label"] = 1  # force the seizure right to the start
    marked = mark_calibration(early, cfg)
    assert marked[marked["patient"] == "chb03"]["is_calibration"].sum() == 300
    assert "falling back" in caplog.text


def test_apply_arm_raw_leaves_features_unchanged(df, feature_cols, cfg):
    raw = apply_arm(df, "raw", cfg)
    pd.testing.assert_frame_equal(
        raw[feature_cols].reset_index(drop=True), df[feature_cols].reset_index(drop=True)
    )


def test_both_arms_share_the_same_calibration_mask_and_row_order(df, cfg):
    raw = apply_arm(df, "raw", cfg)
    std = apply_arm(df, "subject_standardised", cfg)
    assert (raw["is_calibration"].to_numpy() == std["is_calibration"].to_numpy()).all()
    assert list(raw["patient"]) == list(std["patient"])


def test_standardised_calibration_windows_have_mean_zero_std_one(df, feature_cols, cfg):
    std = apply_arm(df, "subject_standardised", cfg)
    for _patient, group in std.groupby("patient"):
        calib = group.loc[group["is_calibration"], feature_cols]
        assert calib.mean().abs().max() < 1e-4
        assert abs(calib.std(ddof=0).mean() - 1.0) < 1e-4


def test_standardisation_uses_only_that_patients_own_data(df, feature_cols, cfg):
    """L5: no cross-patient leakage. A patient standardised alone must match the same
    patient's result when standardised as part of the full cohort."""
    std_full = apply_arm(df, "subject_standardised", cfg)
    one = df[df["patient"] == "chb02"]
    std_alone = apply_arm(one, "subject_standardised", cfg)

    key_cols = ["case", "file", "start_sample"]
    a = std_alone.set_index(key_cols)[feature_cols[0]].sort_index()
    b = std_full[std_full["patient"] == "chb02"].set_index(key_cols)[feature_cols[0]].sort_index()
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-4)


def test_standardise_per_patient_handles_a_patient_with_no_calibration_windows(feature_cols, caplog):
    df = pd.DataFrame(
        {
            "patient": ["p1", "p1"],
            "is_calibration": [False, False],
            feature_cols[0]: [1.0, 2.0],
        }
    )
    for col in feature_cols[1:]:
        df[col] = 0.0
    out = standardise_per_patient(df, feature_cols)
    pd.testing.assert_frame_equal(out[feature_cols], df[feature_cols])
    assert "no calibration windows" in caplog.text


def test_apply_arm_rejects_unknown_arm(df, cfg):
    with pytest.raises(ValueError, match="Unknown arm"):
        apply_arm(df, "not_an_arm", cfg)