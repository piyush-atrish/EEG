"""Tests for models/classifiers.py and the Step 11 additions to evaluation/loso.py."""

import pandas as pd
import pytest

from eegpipe.evaluation.loso import run_loso, subsample_training
from tests.fixtures.synth_features import make_synthetic_features


def _with_tuning_mode(cfg: dict, mode: str) -> dict:
    """A shallow copy of cfg with evaluation.tuning.mode overridden -- never mutate repo_cfg
    in place, since it's a session-scoped fixture shared with every other test module."""
    return {**cfg, "evaluation": {**cfg["evaluation"], "tuning": {**cfg["evaluation"]["tuning"], "mode": mode}}}


def _with_train_sampling(cfg: dict, **overrides) -> dict:
    sampling = {**cfg["evaluation"]["train_sampling"], **overrides}
    return {**cfg, "evaluation": {**cfg["evaluation"], "train_sampling": sampling}}


@pytest.fixture(scope="module")
def cfg(repo_cfg):
    return repo_cfg


@pytest.fixture(scope="module")
def fixed_cfg(cfg):
    """Fixed tuning mode: no GridSearchCV, so the bulk of these tests run in milliseconds."""
    return _with_tuning_mode(cfg, "fixed")


@pytest.fixture(scope="module")
def df(cfg):
    return make_synthetic_features(
        cfg, n_patients=5, windows_per_patient=600, effect=1.5, patient_shift=0.0, seed=0
    )


@pytest.fixture(scope="module")
def feature_cols(df):
    return [c for c in df.columns if c.startswith("f_")]


# ---------------------------------------------------------------- subsample_training


def test_subsample_keeps_all_ictal_and_respects_ratio(df, cfg):
    out = subsample_training(df, cfg, seed=0)
    ratio = cfg["evaluation"]["train_sampling"]["interictal_to_ictal_ratio"]

    assert (out["label"] == 1).sum() == (df["label"] == 1).sum(), "every ictal window is kept"
    n_ictal, n_interictal = (out["label"] == 1).sum(), (out["label"] == 0).sum()
    assert n_interictal == pytest.approx(n_ictal * ratio, abs=n_ictal)  # per-patient rounding
    assert len(out) <= len(df)


def test_subsample_respects_the_cap(df, cfg):
    capped_cfg = _with_train_sampling(cfg, max_train_windows=200)
    out = subsample_training(df, capped_cfg, seed=0)
    # a small overshoot is allowed by design (see next test): the per-patient ictal floor
    # can push the realised total slightly above the cap.
    assert len(out) <= 200 + df["patient"].nunique()


def test_subsample_never_drops_a_patients_last_ictal_window(df, cfg):
    """Even an extreme cap must not erase any patient's seizure signal entirely."""
    tiny_cfg = _with_train_sampling(cfg, max_train_windows=10)
    out = subsample_training(df, tiny_cfg, seed=0)
    ictal_per_patient = out[out["label"] == 1].groupby("patient").size()
    assert (ictal_per_patient >= 1).all()
    assert ictal_per_patient.shape[0] == df["patient"].nunique()


def test_subsample_is_deterministic(df, cfg):
    a = subsample_training(df, cfg, seed=3)
    b = subsample_training(df, cfg, seed=3)
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_subsample_with_no_ictal_returns_input_unchanged(df, cfg):
    no_ictal = df[df["label"] == 0]
    out = subsample_training(no_ictal, cfg, seed=0)
    assert len(out) == len(no_ictal)


def test_subsample_only_touches_training_data_via_run_loso(df, feature_cols, fixed_cfg):
    """L3: the held-out patient's own windows must never be subsampled -- run_loso must call
    `subsample_training` on the training split only. Verified by construction: a held-out
    patient's scored window count and class ratio must exactly equal that patient's true
    (unsampled) data, regardless of what `train_sampling` does to everyone else."""
    preds, per_patient = run_loso(df, feature_cols, "svm", fixed_cfg, arm="raw")
    for patient, group in df.groupby("patient"):
        scored = preds[preds["patient"] == patient]
        assert len(scored) == len(group)
        assert scored["y_true"].sum() == group["label"].sum()


# ---------------------------------------------------------------- signal-strength behaviour


def test_auc_near_chance_with_no_signal(cfg, fixed_cfg):
    df0 = make_synthetic_features(cfg, n_patients=5, windows_per_patient=500, effect=0.0, seed=1)
    feat_cols = [c for c in df0.columns if c.startswith("f_")]
    _, per_patient = run_loso(df0, feat_cols, "svm", fixed_cfg, arm="raw")
    assert 0.3 <= per_patient["auc"].mean() <= 0.7


def test_auc_high_with_strong_signal_and_no_patient_shift(cfg, fixed_cfg):
    df1 = make_synthetic_features(
        cfg, n_patients=5, windows_per_patient=500, effect=3.0, patient_shift=0.0, seed=1
    )
    feat_cols = [c for c in df1.columns if c.startswith("f_")]
    _, per_patient = run_loso(df1, feat_cols, "svm", fixed_cfg, arm="raw")
    assert per_patient["auc"].mean() > 0.9


def test_raw_arm_degrades_under_strong_patient_shift(cfg, fixed_cfg):
    """Reused in Step 12: this is exactly the gap `subject_standardised` is meant to close."""
    feat_cols_key = None
    aucs = {}
    for patient_shift in (0.0, 5.0):
        d = make_synthetic_features(
            cfg, n_patients=5, windows_per_patient=500, effect=1.0, patient_shift=patient_shift, seed=1
        )
        feat_cols_key = [c for c in d.columns if c.startswith("f_")]
        _, per_patient = run_loso(d, feat_cols_key, "svm", fixed_cfg, arm="raw")
        aucs[patient_shift] = per_patient["auc"].mean()

    assert aucs[0.0] - aucs[5.0] > 0.1, aucs


# ---------------------------------------------------------------- both models, both modes


@pytest.mark.parametrize("model_name", ["svm", "rf"])
@pytest.mark.parametrize("mode", ["fixed", "nested"])
def test_both_models_run_end_to_end_in_both_tuning_modes(df, feature_cols, cfg, model_name, mode):
    run_cfg = _with_tuning_mode(cfg, mode)
    preds, per_patient = run_loso(df, feature_cols, model_name, run_cfg, arm="raw", patients=["chb01"])
    assert len(per_patient) == 1
    assert (preds["model"] == model_name).all()
    assert not preds["y_score"].isna().any()


# ---------------------------------------------------------------- L4: nested tuning groups


def test_nested_tuning_never_sees_the_held_out_patient(df, feature_cols, cfg, monkeypatch):
    import eegpipe.evaluation.loso as loso_mod

    seen_groups = []
    original_tune_model = loso_mod.tune_model

    def spy(pipe, grid, X, y, groups, cfg, seed):
        seen_groups.append(set(groups))
        return original_tune_model(pipe, grid, X, y, groups, cfg, seed)

    monkeypatch.setattr(loso_mod, "tune_model", spy)
    loso_mod.run_loso(df, feature_cols, "rf", cfg, arm="raw", patients=["chb01"])

    assert len(seen_groups) == 1
    assert "chb01" not in seen_groups[0]
    assert seen_groups[0] == set(df["patient"].unique()) - {"chb01"}


# ---------------------------------------------------------------- determinism


def test_run_loso_is_deterministic_with_fixed_tuning(df, feature_cols, fixed_cfg):
    a, _ = run_loso(df, feature_cols, "svm", fixed_cfg, arm="raw", patients=["chb02"])
    b, _ = run_loso(df, feature_cols, "svm", fixed_cfg, arm="raw", patients=["chb02"])
    pd.testing.assert_frame_equal(a, b)


def test_run_loso_is_deterministic_with_nested_tuning(df, feature_cols, cfg):
    a, _ = run_loso(df, feature_cols, "svm", cfg, arm="raw", patients=["chb02"])
    b, _ = run_loso(df, feature_cols, "svm", cfg, arm="raw", patients=["chb02"])
    pd.testing.assert_frame_equal(a, b)