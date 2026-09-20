"""Tests for classifiers, training subsampling, nested tuning, the LOSO loop and scripts 05/06."""
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from eegpipe.evaluation import loso
from eegpipe.evaluation.loso import (
    _allocate,
    feature_columns,
    run_loso,
    subsample_training,
    tune_model,
)
from eegpipe.models.classifiers import make_model
from eegpipe.models.normalisation import apply_arm
from tests.fixtures.synth_features import make_synthetic_features, make_test_config

REPO = Path(__file__).resolve().parents[1]


def mean_auc(df, cfg, model="svm", arm="raw", **kw):
    prepared = apply_arm(df, arm, cfg)
    _, per_patient = run_loso(prepared, feature_columns(prepared), model, cfg, arm, **kw)
    return float(per_patient["auc"].mean())


@pytest.fixture
def cfg(tmp_path):
    return make_test_config(tmp_path, tuning_mode="fixed")


@pytest.fixture
def df(cfg):
    return make_synthetic_features(cfg, n_patients=6, effect=1.5, patient_shift=0.5, seed=1)


# ---------------------------------------------------------------- make_model
@pytest.mark.parametrize("name,klass", [("svm", SVC), ("rf", RandomForestClassifier)])
def test_make_model_structure(cfg, name, klass):
    pipe, grid = make_model(name, cfg, n_features=576, seed=3)
    assert isinstance(pipe, Pipeline)
    assert [s for s, _ in pipe.steps] == ["scale", "select", "clf"]   # scaler + selector inside (L2)
    assert isinstance(pipe.named_steps["clf"], klass)
    assert pipe.named_steps["select"].k == 100
    assert pipe.named_steps["clf"].random_state == 3
    assert all(key.startswith("clf__") for key in grid)


def test_make_model_clips_k_and_rejects_unknown(cfg):
    pipe, _ = make_model("svm", cfg, n_features=40, seed=0)
    assert pipe.named_steps["select"].k == 40
    with pytest.raises(ValueError):
        make_model("knn", cfg, 40, 0)
    assert make_model("svm", cfg, 576, 0)[0].named_steps["clf"].class_weight == "balanced"
    assert make_model("rf", cfg, 576, 0)[0].named_steps["clf"].class_weight == "balanced_subsample"


# ---------------------------------------------------------------- subsample_training
def test_allocate_invariants():
    q = _allocate(np.array([100, 50, 0, 10]), 40)
    assert q.sum() == 40 and (q <= np.array([100, 50, 0, 10])).all() and q[2] == 0
    assert _allocate(np.array([3, 2]), 999).tolist() == [3, 2]      # capped by availability
    assert _allocate(np.array([3, 2]), 0).tolist() == [0, 0]


def test_subsample_respects_ratio_and_touches_only_training(df, cfg):
    train = df[df.patient != "chb01"]
    sub = subsample_training(train, cfg, seed=0)
    assert set(sub.patient) == set(train.patient) and "chb01" not in set(sub.patient)
    n_ictal_all = int(train.label.sum())
    assert int(sub.label.sum()) == n_ictal_all                      # all ictal kept
    assert (sub.label == 0).sum() == 5 * n_ictal_all                # ratio 5:1 (default config)
    assert sub.index.is_monotonic_increasing and set(sub.index) <= set(train.index)


def test_subsample_respects_cap_and_keeps_ratio_and_one_ictal_per_patient(df, cfg):
    cfg["evaluation"]["train_sampling"]["max_train_windows"] = 300
    sub = subsample_training(df, cfg, seed=0)
    n_ictal, n_inter = int(sub.label.sum()), int((sub.label == 0).sum())
    assert len(sub) <= 300
    assert n_inter == 5 * n_ictal or abs(n_inter - 5 * n_ictal) <= 5
    assert (sub.groupby("patient")["label"].sum() >= 1).all()


def test_subsample_keeps_at_least_one_ictal_when_cap_is_tiny(df, cfg):
    cfg["evaluation"]["train_sampling"]["max_train_windows"] = 60
    sub = subsample_training(df, cfg, seed=0)
    assert (sub.groupby("patient")["label"].sum() >= 1).all()
    assert (sub.label == 0).sum() > 0


def test_subsample_is_proportional_per_patient_and_deterministic(df, cfg):
    a = subsample_training(df, cfg, seed=7)
    b = subsample_training(df, cfg, seed=7)
    c = subsample_training(df, cfg, seed=8)
    assert a.index.equals(b.index) and not a.index.equals(c.index)
    per = a[a.label == 0].groupby("patient").size()
    assert per.max() - per.min() <= 2       # patients have equal interictal counts -> equal shares


def test_subsample_uses_only_patient_and_label_columns(df, cfg):
    meta = df[["patient", "label"]]
    assert subsample_training(meta, cfg, 0).index.equals(subsample_training(df, cfg, 0).index)


# ---------------------------------------------------------------- tune_model
def test_tune_model_fixed_and_nested_return_fitted_pipeline(cfg):
    df = make_synthetic_features(cfg, n_patients=5, effect=2.0, patient_shift=0.3, seed=2)
    cols = feature_columns(df)
    X, y, g = df[cols].to_numpy(), df["label"].to_numpy(), df["patient"].to_numpy()
    for mode in ("fixed", "nested"):
        cfg["evaluation"]["tuning"]["mode"] = mode
        for name in ("svm", "rf"):
            pipe, grid = make_model(name, cfg, len(cols), 0)
            fitted = tune_model(pipe, grid, X, y, g, cfg, 0)
            assert isinstance(fitted, Pipeline)
            assert fitted.predict(X[:5]).shape == (5,)


def test_tune_model_single_group_falls_back_and_bad_mode_raises(df, cfg, caplog):
    cols = feature_columns(df)
    one = df[df.patient == "chb02"]
    cfg["evaluation"]["tuning"]["mode"] = "nested"
    pipe, grid = make_model("svm", cfg, len(cols), 0)
    with caplog.at_level(logging.WARNING):
        fitted = tune_model(pipe, grid, one[cols].to_numpy(), one["label"].to_numpy(),
                            one["patient"].to_numpy(), cfg, 0)
    assert "fixed parameters" in caplog.text and isinstance(fitted, Pipeline)
    cfg["evaluation"]["tuning"]["mode"] = "bogus"
    with pytest.raises(ValueError):
        tune_model(pipe, grid, one[cols].to_numpy(), one["label"].to_numpy(), one["patient"].to_numpy(), cfg, 0)


def test_nested_tuning_uses_only_training_patients(df, cfg, monkeypatch):
    cfg["evaluation"]["tuning"]["mode"] = "nested"
    seen = {}
    orig_tune, orig_sub = loso.tune_model, loso.subsample_training

    def spy_tune(pipe, grid, X, y, groups, cfg_, seed):
        seen["tune_groups"] = set(groups)
        return orig_tune(pipe, grid, X, y, groups, cfg_, seed)

    def spy_sub(d, cfg_, seed):
        seen["sub_patients"] = set(d["patient"])
        return orig_sub(d, cfg_, seed)

    monkeypatch.setattr(loso, "tune_model", spy_tune)
    monkeypatch.setattr(loso, "subsample_training", spy_sub)
    prepared = apply_arm(df, "raw", cfg)
    run_loso(prepared, feature_columns(prepared), "svm", cfg, "raw", patients=["chb03"])
    others = set(df.patient) - {"chb03"}
    assert "chb03" not in seen["tune_groups"] and seen["tune_groups"] == others   # L4
    assert "chb03" not in seen["sub_patients"] and seen["sub_patients"] == others  # L3


# ---------------------------------------------------------------- LOSO behaviour on synthetic data
def test_no_signal_gives_chance_auc(cfg):
    df = make_synthetic_features(cfg, n_patients=8, effect=0.0, patient_shift=1.0, seed=3)
    for model in ("svm", "rf"):
        assert 0.35 <= mean_auc(df, cfg, model) <= 0.65


def test_strong_signal_without_subject_variability_gives_high_auc(cfg):
    df = make_synthetic_features(cfg, n_patients=8, effect=3.0, patient_shift=0.0, seed=4)
    for model in ("svm", "rf"):
        assert mean_auc(df, cfg, model) > 0.9


def test_large_patient_shift_degrades_raw_arm(cfg):
    kw = dict(n_patients=8, effect=1.0, seed=5)
    no_shift = mean_auc(make_synthetic_features(cfg, patient_shift=0.0, **kw), cfg)
    big_shift = mean_auc(make_synthetic_features(cfg, patient_shift=8.0, **kw), cfg)
    assert no_shift > 0.9
    assert big_shift < no_shift - 0.1


def test_both_models_run_in_both_tuning_modes(df, cfg):
    prepared = apply_arm(df, "raw", cfg)
    cols = feature_columns(prepared)
    for mode in ("fixed", "nested"):
        cfg["evaluation"]["tuning"]["mode"] = mode
        for model in ("svm", "rf"):
            pred, per_patient = run_loso(prepared, cols, model, cfg, "raw", patients=["chb02", "chb05"])
            assert len(per_patient) == 2 and len(pred) > 0
            assert per_patient["auc"].notna().all()


def test_same_seed_gives_identical_results(df, cfg):
    prepared = apply_arm(df, "raw", cfg)
    cols = feature_columns(prepared)
    for model in ("svm", "rf"):
        a, _ = run_loso(prepared, cols, model, cfg, "raw", patients=["chb02", "chb04"])
        b, _ = run_loso(prepared, cols, model, cfg, "raw", patients=["chb02", "chb04"])
        pd.testing.assert_frame_equal(a, b)


def test_svm_scores_are_decision_values_and_train_labels_shuffle_changes_model(df, cfg):
    prepared = apply_arm(df, "raw", cfg)
    cols = feature_columns(prepared)
    real, _ = run_loso(prepared, cols, "svm", cfg, "raw", patients=["chb02"])
    shuf, _ = run_loso(prepared, cols, "svm", cfg, "raw", patients=["chb02"], shuffle_train_labels=True)
    assert real["y_score"].min() < 0 < real["y_score"].max()        # decision_function, not a probability
    assert not np.allclose(real["y_score"], shuf["y_score"])
    assert (real["y_true"].to_numpy() == shuf["y_true"].to_numpy()).all()  # test labels never shuffled


# ---------------------------------------------------------------- scripts 05 / 06
def _load_script(filename):
    spec = importlib.util.spec_from_file_location("script_" + filename.split("_")[0], REPO / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scripts_05_and_06_end_to_end_on_synthetic_data(cfg, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    s05, s06 = _load_script("05_run_baseline.py"), _load_script("06_make_report.py")

    rc = s05.main(["--config", str(cfg_path), "--synthetic", "--tuning", "fixed", "--models", "svm",
                   "--max-patients", "4", "--label-shuffle"])
    assert rc == 0
    tables = Path(cfg["paths"]["tables"])
    for arm in ("raw", "subject_standardised"):
        t = pd.read_csv(tables / f"per_patient_{arm}__svm.csv")
        assert list(t.columns) == ["patient", "n_windows", "n_ictal", "tp", "fp", "tn", "fn",
                                   "sensitivity", "specificity", "f1", "auc", "fa_per_hour"]
        assert len(t) == 4
        assert (Path(cfg["paths"]["predictions"]) / f"{arm}__svm.parquet").exists()
    # both arms are scored on identical windows
    a = pd.read_csv(tables / "per_patient_raw__svm.csv")[["patient", "n_windows", "n_ictal"]]
    b = pd.read_csv(tables / "per_patient_subject_standardised__svm.csv")[["patient", "n_windows", "n_ictal"]]
    pd.testing.assert_frame_equal(a, b)

    assert s06.main(["--config", str(cfg_path)]) == 0
    summary = pd.read_csv(tables / "summary_all.csv")
    assert list(summary.columns) == ["arm", "model", "metric", "mean", "std", "median", "n_patients"]
    assert set(summary["arm"]) == {"raw", "subject_standardised"} and set(summary["model"]) == {"svm"}
    assert {"sensitivity", "specificity", "f1", "auc", "fa_per_hour"} == set(summary["metric"])
    figs = sorted(p.name for p in Path(cfg["paths"]["figures"]).glob("*.png"))
    assert figs == ["per_patient_auc__svm.png", "per_patient_sensitivity__svm.png",
                    "per_patient_specificity__svm.png"]


def test_script_05_maps_chb21_to_chb01_and_needs_two_patients(cfg, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    s05 = _load_script("05_run_baseline.py")
    with pytest.raises(SystemExit):
        s05.main(["--config", str(cfg_path), "--synthetic", "--tuning", "fixed", "--patients", "chb21"])
