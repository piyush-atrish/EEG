"""Tests for evaluation/feature_direction.py on synthetic feature tables."""

import numpy as np
import pandas as pd
import pytest

from eegpipe.evaluation.feature_direction import (
    _auc_columns,
    direction_summary,
    feature_auc_by_patient,
)


def test_auc_columns_known_values():
    y = np.array([0, 0, 0, 1, 1, 1])
    values = np.column_stack(
        [
            [1, 2, 3, 4, 5, 6],  # ictal always higher -> 1.0
            [6, 5, 4, 3, 2, 1],  # ictal always lower  -> 0.0
            [1, 1, 1, 1, 1, 1],  # no information      -> 0.5
        ]
    ).astype(float)
    assert _auc_columns(values, y) == pytest.approx([1.0, 0.0, 0.5])


def make_df(flip_patient="p5", n_pos=60, n_neg=400, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(1, 6):
        patient = f"p{i}"
        sign = -1.0 if patient == flip_patient else 1.0
        y = np.r_[np.zeros(n_neg), np.ones(n_pos)].astype(int)
        shift = 3.0 * sign * y
        frames.append(
            pd.DataFrame(
                {
                    "patient": patient,
                    "label": y,
                    "f_signal": rng.normal(0, 1, len(y)) + shift,
                    "f_noise": rng.normal(0, 1, len(y)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_table_has_one_column_per_eligible_patient():
    df = make_df()
    df = pd.concat([df, df[df.patient == "p1"].assign(patient="tiny", label=0)], ignore_index=True)
    auc = feature_auc_by_patient(df, ["f_signal", "f_noise"], min_ictal=30)
    assert list(auc.columns) == ["p1", "p2", "p3", "p4", "p5"]  # 'tiny' has no ictal windows
    assert auc.loc["f_signal", "p1"] > 0.95 and auc.loc["f_signal", "p5"] < 0.05
    assert abs(auc.loc["f_noise", "p2"] - 0.5) < 0.1


def test_flipped_patient_has_low_agreement_and_others_high():
    auc = feature_auc_by_patient(make_df(), ["f_signal", "f_noise"])
    summary = direction_summary(auc)
    assert summary.loc["p5", "agreement"] == pytest.approx(0.0)
    assert summary.loc["p1", "agreement"] == pytest.approx(1.0)
    assert summary.loc["p1", "n_strong_features"] == 1  # only f_signal is a strong feature
    assert summary.loc["p5", "separability"] > 0.2  # flipped, but still informative


def test_nan_features_are_tolerated():
    df = make_df()
    df.loc[df.index[::50], "f_signal"] = np.nan
    auc = feature_auc_by_patient(df, ["f_signal", "f_noise"])
    assert np.isfinite(auc.to_numpy()).all()


def test_two_patients_give_no_agreement_estimate():
    auc = feature_auc_by_patient(make_df().query("patient in ['p1', 'p2']"), ["f_signal", "f_noise"])
    assert direction_summary(auc)["agreement"].isna().all()