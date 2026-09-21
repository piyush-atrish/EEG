"""Tests for calibration marking, per-patient standardisation and the two arms."""
import logging

import numpy as np
import pytest

from eegpipe.evaluation.loso import feature_columns, run_loso
from eegpipe.models.normalisation import apply_arm, mark_calibration, standardise_per_patient
from tests.fixtures.synth_features import make_synthetic_features, make_test_config


@pytest.fixture
def cfg(tmp_path):
    return make_test_config(tmp_path, tuning_mode="fixed")


@pytest.fixture
def df(cfg):
    return make_synthetic_features(cfg, n_patients=6, effect=1.5, patient_shift=1.0, seed=1)


def chrono(g):
    return g.sort_values(["case", "file", "start_sample"], kind="stable")


# ---------------------------------------------------------------- mark_calibration
def test_calibration_marks_first_300_interictal_windows_before_first_ictal(df, cfg):
    marked = mark_calibration(df, cfg)
    for patient, g in marked.groupby("patient"):
        g = chrono(g).reset_index(drop=True)
        first_ictal = int(np.flatnonzero(g.label.to_numpy() == 1)[0])
        expected = np.zeros(len(g), dtype=bool)
        expected[:300] = True  # synthetic data: windows 0..349 are interictal, seizures start later
        assert first_ictal >= 350
        assert g.is_calibration.sum() == 300, patient
        assert (g.is_calibration.to_numpy() == expected).all(), patient
        assert (g.loc[g.is_calibration, "label"] == 0).all()


def test_calibration_respects_case_order_for_multi_case_patient(df, cfg):
    marked = mark_calibration(df, cfg)
    g = marked[marked.patient == "chb01"]
    # chb01 (case chb01) precedes chb21 chronologically: the 300 calibration windows lie in case chb01
    assert set(g.loc[g.is_calibration, "case"]) == {"chb01"}
    assert g[g.case == "chb01"].is_calibration.sum() == 300


def test_mark_calibration_keeps_row_order_index_and_input(df, cfg):
    shuffled = df.sample(frac=1.0, random_state=0)
    marked = mark_calibration(shuffled, cfg)
    assert marked.index.equals(shuffled.index)
    assert "is_calibration" not in shuffled.columns                  # input not mutated
    ref = mark_calibration(df, cfg).set_index(["case", "file", "start_sample"])["is_calibration"]
    got = marked.set_index(["case", "file", "start_sample"])["is_calibration"]
    assert (ref.sort_index() == got.sort_index()).all()              # same windows regardless of row order


def test_fallback_when_too_few_interictal_windows_before_first_seizure(df, cfg, caplog):
    bad = df.copy()
    g = chrono(bad[bad.patient == "chb03"])
    early_idx = g.index[10]                     # a seizure window after only 10 interictal windows
    bad.loc[early_idx, "label"] = 1
    with caplog.at_level(logging.WARNING):
        marked = mark_calibration(bad, cfg)
    assert "chb03" in caplog.text and "min_windows" in caplog.text
    m = marked[marked.patient == "chb03"]
    assert m.is_calibration.sum() == 300
    # fallback = first 300 interictal windows overall (the seizure window itself is excluded)
    assert not m.loc[early_idx, "is_calibration"]
    other = marked[marked.patient == "chb02"]
    assert other.is_calibration.sum() == 300


def test_patient_without_seizure_windows_uses_first_interictal_windows(df, cfg):
    bad = df.copy()
    bad.loc[bad.patient == "chb04", "label"] = 0
    marked = mark_calibration(bad, cfg)
    assert marked[marked.patient == "chb04"].is_calibration.sum() == 300


# ---------------------------------------------------------------- standardise_per_patient
def test_standardised_calibration_windows_have_mean0_std1(df, cfg):
    out = apply_arm(df, "subject_standardised", cfg)
    cols = feature_columns(out)
    for patient, g in out.groupby("patient"):
        cal = g.loc[g.is_calibration, cols].to_numpy(dtype=np.float64)
        assert np.abs(cal.mean(axis=0)).max() < 1e-4, patient
        assert np.abs(cal.std(axis=0) - 1.0).max() < 1e-3, patient


def test_other_patients_never_change_a_patients_statistics(df, cfg):
    cols = feature_columns(df)
    a = apply_arm(df, "subject_standardised", cfg)
    tampered = df.copy()
    mask = tampered.patient == "chb03"
    tampered.loc[mask, cols] = tampered.loc[mask, cols] * 50.0 + 7.0
    b = apply_arm(tampered, "subject_standardised", cfg)
    for p in ("chb01", "chb02", "chb04", "chb05", "chb06"):
        ga, gb = a[a.patient == p][cols].to_numpy(), b[b.patient == p][cols].to_numpy()
        assert np.array_equal(ga, gb), p
    # a rescaled patient is undone by its own standardisation (affine invariance), up to float32 rounding
    assert np.allclose(a[a.patient == "chb03"][cols].to_numpy(), b[b.patient == "chb03"][cols].to_numpy(), atol=1e-3)


def test_calibration_statistics_use_calibration_windows_only(df, cfg):
    """Changing non-calibration windows must not change the statistics (no label/test leakage, L5)."""
    cols = feature_columns(df)
    marked = mark_calibration(df, cfg)
    tampered = marked.copy()
    m = (tampered.patient == "chb02") & ~tampered.is_calibration
    tampered.loc[m, cols] = tampered.loc[m, cols] + 100.0
    a = standardise_per_patient(marked, cols)
    b = standardise_per_patient(tampered, cols)
    cal = (marked.patient == "chb02") & marked.is_calibration
    assert np.array_equal(a.loc[cal, cols].to_numpy(), b.loc[cal, cols].to_numpy())


def test_standardise_requires_calibration_mask_and_flags_constant_features(df, cfg, caplog):
    cols = feature_columns(df)
    with pytest.raises(ValueError):
        standardise_per_patient(df, cols)
    marked = mark_calibration(df, cfg)
    marked.loc[marked.patient == "chb02", cols[0]] = 3.0             # constant feature
    with caplog.at_level(logging.WARNING):
        out = standardise_per_patient(marked, cols)
    assert "constant" in caplog.text and np.isfinite(out[cols].to_numpy()).all()


# ---------------------------------------------------------------- apply_arm
def test_arms_share_rows_and_calibration_mask(df, cfg):
    raw = apply_arm(df, "raw", cfg)
    std = apply_arm(df, "subject_standardised", cfg)
    assert raw.index.equals(std.index) and len(raw) == len(df)
    assert (raw.is_calibration.to_numpy() == std.is_calibration.to_numpy()).all()
    cols = feature_columns(df)
    assert np.array_equal(raw[cols].to_numpy(), df[cols].to_numpy())  # raw arm leaves features unchanged
    assert not np.allclose(std[cols].to_numpy(), df[cols].to_numpy())
    for c in ("patient", "case", "file", "start_sample", "label"):
        assert (raw[c].to_numpy() == std[c].to_numpy()).all()


def test_unknown_arm_rejected(df, cfg):
    with pytest.raises(ValueError):
        apply_arm(df, "adaptive", cfg)


def test_both_arms_are_scored_on_identical_windows(df, cfg):
    keys = {}
    for arm in ("raw", "subject_standardised"):
        prepared = apply_arm(df, arm, cfg)
        pred, _ = run_loso(prepared, feature_columns(prepared), "svm", cfg, arm, patients=["chb02", "chb03"])
        keys[arm] = set(zip(pred.case, pred.file, pred.start_sample))
    assert keys["raw"] == keys["subject_standardised"]


def test_standardisation_beats_raw_under_strong_subject_shift(cfg):
    df = make_synthetic_features(cfg, n_patients=8, effect=1.0, patient_shift=8.0, seed=5)
    auc = {}
    for arm in ("raw", "subject_standardised"):
        prepared = apply_arm(df, arm, cfg)
        _, per_patient = run_loso(prepared, feature_columns(prepared), "svm", cfg, arm)
        auc[arm] = float(per_patient["auc"].mean())
    assert auc["subject_standardised"] > auc["raw"] + 0.1, auc
