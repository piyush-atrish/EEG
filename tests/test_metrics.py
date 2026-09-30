"""Tests for evaluation/metrics.py (Member C, Step 10)."""

import math

import numpy as np
import pandas as pd
import pytest

from eegpipe.evaluation.metrics import aggregate_metrics, compute_metrics

STEP_S = 2.0


def test_hand_computed_confusion_matrix():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 0, 1]  # tp=3, fn=1, tn=3, fp=1
    y_score = [0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7]

    m = compute_metrics(y_true, y_score, y_pred, STEP_S)

    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 3, 1)
    assert m["n_windows"] == 8
    assert m["n_ictal"] == 4
    assert m["sensitivity"] == pytest.approx(3 / 4)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 1))
    # 1 false positive over 8 windows * 2 s = 16 s of recording
    assert m["fa_per_hour"] == pytest.approx(1 / (16 / 3600))
    # every ictal score (0.9, 0.8, 0.4, 0.7) beats every interictal one except 0.6 vs 0.4
    assert m["auc"] == pytest.approx(15 / 16)


def test_perfect_classifier():
    y = [0, 0, 0, 1, 1, 1]
    m = compute_metrics(y, [0, 0, 0, 1, 1, 1], y, STEP_S)
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["f1"] == 1.0
    assert m["auc"] == 1.0
    assert m["fp"] == 0
    assert m["fa_per_hour"] == 0.0


def test_constant_classifier_is_zero_not_nan():
    """Predicting all-interictal misses every seizure: sensitivity is 0, not undefined."""
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.1, 0.2], [0, 0, 0, 0], STEP_S)
    assert m["sensitivity"] == 0.0
    assert m["specificity"] == 1.0
    assert m["f1"] == 0.0
    assert (m["tp"], m["fn"]) == (0, 2)
    # constant scores carry no ranking information
    m_const = compute_metrics([0, 0, 1, 1], [0.5] * 4, [0, 0, 0, 0], STEP_S)
    assert m_const["auc"] == pytest.approx(0.5)


def test_single_class_patient_gives_nan_not_crash():
    """A held-out patient with no ictal windows: undefined metrics are NaN, defined ones stay."""
    m = compute_metrics([0, 0, 0, 0], [0.1, 0.2, 0.9, 0.1], [0, 0, 1, 0], STEP_S)
    assert m["n_ictal"] == 0
    assert math.isnan(m["sensitivity"])
    assert math.isnan(m["f1"])
    assert math.isnan(m["auc"])
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["fp"] == 1
    assert m["fa_per_hour"] == pytest.approx(1 / (8 / 3600))


def test_all_ictal_patient_gives_nan_specificity_and_auc():
    m = compute_metrics([1, 1, 1], [0.9, 0.2, 0.8], [1, 0, 1], STEP_S)
    assert math.isnan(m["specificity"])
    assert math.isnan(m["auc"])
    assert m["sensitivity"] == pytest.approx(2 / 3)


def test_empty_input_does_not_crash():
    m = compute_metrics([], [], [], STEP_S)
    assert m["n_windows"] == 0
    assert math.isnan(m["fa_per_hour"])
    assert math.isnan(m["auc"])


def test_compute_metrics_returns_all_contract_keys():
    m = compute_metrics([0, 1], [0.1, 0.9], [0, 1], STEP_S)
    assert list(m) == [
        "n_windows", "n_ictal", "tp", "fp", "tn", "fn",
        "sensitivity", "specificity", "f1", "auc", "fa_per_hour",
    ]


def test_aggregate_ignores_nan_and_reports_contributing_patients():
    per_patient = pd.DataFrame(
        {
            "patient": ["chb01", "chb02", "chb03"],
            "sensitivity": [0.8, np.nan, 0.6],
            "auc": [0.9, 0.95, np.nan],
        }
    )
    agg = aggregate_metrics(per_patient).set_index("metric")

    assert list(agg.columns) == ["mean", "std", "median", "n_patients"]
    assert "patient" not in agg.index
    assert agg.loc["sensitivity", "n_patients"] == 2
    assert agg.loc["sensitivity", "mean"] == pytest.approx(0.7)
    assert agg.loc["sensitivity", "median"] == pytest.approx(0.7)
    assert agg.loc["sensitivity", "std"] == pytest.approx(np.std([0.8, 0.6], ddof=1))
    assert agg.loc["auc", "n_patients"] == 2
    assert agg.loc["auc", "mean"] == pytest.approx(0.925)


def test_aggregate_all_nan_and_single_value_columns():
    per_patient = pd.DataFrame(
        {"patient": ["a", "b"], "f1": [np.nan, np.nan], "auc": [0.7, np.nan]}
    )
    agg = aggregate_metrics(per_patient).set_index("metric")
    assert agg.loc["f1", "n_patients"] == 0
    assert math.isnan(agg.loc["f1", "mean"])
    assert agg.loc["auc", "n_patients"] == 1
    assert agg.loc["auc", "mean"] == pytest.approx(0.7)
    assert math.isnan(agg.loc["auc", "std"])  # sample std undefined for one patient