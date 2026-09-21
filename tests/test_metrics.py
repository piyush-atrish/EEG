"""Tests for eegpipe.evaluation.metrics (hand-computed expectations)."""
import math

import numpy as np
import pandas as pd
import pytest

from eegpipe.evaluation.metrics import PER_PATIENT_COLUMNS, aggregate_metrics, compute_metrics

STEP_S = 2.0


def test_hand_computed_confusion_matrix():
    y_true = [1, 1, 0, 0, 0, 0]
    y_pred = [1, 0, 1, 0, 0, 0]
    y_score = [0.9, 0.4, 0.6, 0.2, 0.1, 0.3]
    m = compute_metrics(y_true, y_score, y_pred, STEP_S)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 3)
    assert m["n_windows"] == 6 and m["n_ictal"] == 2
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(2 * 1 / (2 * 1 + 1 + 1))
    # 4 of the 8 (ictal, interictal) pairs are ranked correctly with ties absent: AUC = 7/8
    assert m["auc"] == pytest.approx(7 / 8)
    # 1 false alarm in 6 windows * 2 s = 12 s -> 300 per hour
    assert m["fa_per_hour"] == pytest.approx(300.0)


def test_fa_per_hour_formula():
    y_true = np.zeros(1800, dtype=int)          # 1800 windows * 2 s = 1 hour
    y_pred = np.zeros(1800, dtype=int)
    y_pred[:3] = 1                              # 3 false alarms
    m = compute_metrics(y_true, np.zeros(1800), y_pred, STEP_S)
    assert m["fa_per_hour"] == pytest.approx(3.0)


def test_perfect_classifier():
    y = [0, 0, 0, 1, 1]
    m = compute_metrics(y, [0.1, 0.2, 0.3, 0.8, 0.9], y, STEP_S)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0
    assert m["f1"] == 1.0 and m["auc"] == 1.0 and m["fa_per_hour"] == 0.0


def test_constant_classifier():
    y_true = [0, 0, 0, 1, 1]
    m = compute_metrics(y_true, [0.0] * 5, [0] * 5, STEP_S)
    assert m["sensitivity"] == 0.0 and m["specificity"] == 1.0
    assert m["f1"] == 0.0
    assert m["auc"] == pytest.approx(0.5)
    assert m["fa_per_hour"] == 0.0


def test_single_class_patient_gives_nan_not_crash():
    only_inter = compute_metrics([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], [0, 1, 0, 0], STEP_S)
    assert math.isnan(only_inter["sensitivity"])
    assert math.isnan(only_inter["f1"]) and math.isnan(only_inter["auc"])
    assert only_inter["specificity"] == pytest.approx(0.75)
    assert only_inter["fa_per_hour"] > 0

    only_ictal = compute_metrics([1, 1, 1], [0.5, 0.6, 0.7], [1, 1, 0], STEP_S)
    assert math.isnan(only_ictal["specificity"]) and math.isnan(only_ictal["auc"])
    assert only_ictal["sensitivity"] == pytest.approx(2 / 3)


def test_empty_input_does_not_crash():
    m = compute_metrics([], [], [], STEP_S)
    assert m["n_windows"] == 0 and math.isnan(m["sensitivity"]) and math.isnan(m["fa_per_hour"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.1], [0, 1], STEP_S)


def test_metric_keys_cover_contract_c7():
    m = compute_metrics([0, 1], [0.1, 0.9], [0, 1], STEP_S)
    assert [c for c in PER_PATIENT_COLUMNS if c != "patient"] == list(m.keys())


def test_aggregate_ignores_nan_and_reports_counts():
    per_patient = pd.DataFrame({
        "patient": ["a", "b", "c", "d"],
        "sensitivity": [0.5, 1.0, float("nan"), 0.0],
        "specificity": [0.9, 0.8, 0.7, 0.6],
        "f1": [0.1, 0.2, 0.3, float("nan")],
        "auc": [0.6, 0.7, 0.8, 0.9],
        "fa_per_hour": [1.0, 2.0, 3.0, 4.0],
    })
    agg = aggregate_metrics(per_patient).set_index("metric")
    assert agg.loc["sensitivity", "mean"] == pytest.approx(0.5)
    assert agg.loc["sensitivity", "median"] == pytest.approx(0.5)
    assert agg.loc["sensitivity", "n_patients"] == 3 and agg.loc["sensitivity", "n_ignored"] == 1
    assert agg.loc["specificity", "mean"] == pytest.approx(0.75)
    assert agg.loc["specificity", "std"] == pytest.approx(np.std([0.9, 0.8, 0.7, 0.6], ddof=1))
    assert agg.loc["f1", "n_patients"] == 3
    assert list(aggregate_metrics(per_patient).columns) == ["metric", "mean", "std", "median",
                                                            "n_patients", "n_ignored"]


def test_aggregate_single_patient_std_is_nan():
    one = pd.DataFrame({"patient": ["a"], "sensitivity": [0.5], "specificity": [0.5],
                        "f1": [0.5], "auc": [0.5], "fa_per_hour": [1.0]})
    agg = aggregate_metrics(one).set_index("metric")
    assert math.isnan(agg.loc["auc", "std"]) and agg.loc["auc", "mean"] == 0.5
