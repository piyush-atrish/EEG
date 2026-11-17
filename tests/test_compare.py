"""Tests for evaluation/compare.py on synthetic per-patient tables."""

import numpy as np
import pandas as pd
import pytest

from eegpipe.evaluation.compare import compare_arms, load_per_patient_tables

PATIENTS = [f"chb{i:02d}" for i in range(1, 11)]
DEV = PATIENTS[:5]


def table(auc, sens=0.5, spec=0.9, fa=10.0):
    return pd.DataFrame(
        {"auc": auc, "sensitivity": sens, "specificity": spec, "fa_per_hour": fa},
        index=pd.Index(PATIENTS, name="patient"),
    )


def make_tables(shift=0.1):
    base = np.linspace(0.6, 0.9, 10)
    return {
        ("raw", "rf"): table(base),
        ("subject_standardised", "rf"): table(base + shift),
        ("raw", "svm"): table(base),
        ("subject_standardised", "svm"): table(base - shift),
    }


def test_consistent_improvement_is_detected_per_subset():
    out = compare_arms(make_tables(0.1), "raw", DEV, n_boot=500)
    rf = out[(out.model == "rf") & (out.metric == "auc")].set_index("subset")
    assert rf.loc["all", "mean_diff"] == pytest.approx(0.1)
    assert rf.loc["all", "n_patients"] == 10 and rf.loc["dev", "n_patients"] == 5
    assert rf.loc["all", "p_value"] == pytest.approx(2 / 2**10)
    assert rf.loc["dev", "p_value"] == pytest.approx(0.0625)  # five patients cannot do better
    assert rf.loc["all", "ci_low"] == pytest.approx(0.1) and rf.loc["all", "ci_high"] == pytest.approx(0.1)


def test_worse_arm_has_negative_difference():
    out = compare_arms(make_tables(0.1), "raw", DEV, n_boot=500)
    svm = out[(out.model == "svm") & (out.metric == "auc") & (out.subset == "eval")].iloc[0]
    assert svm["mean_diff"] == pytest.approx(-0.1)
    assert svm["n_reference_greater"] == 5 and svm["n_arm_greater"] == 0


def test_reference_rows_are_not_compared_with_themselves():
    out = compare_arms(make_tables(), "raw", DEV, n_boot=200)
    assert set(out["arm"]) == {"subject_standardised"}
    assert set(out["metric"]) == {"auc", "sensitivity", "specificity", "fa_per_hour"}


def test_missing_reference_is_an_error():
    with pytest.raises(ValueError):
        compare_arms(make_tables(), "adaptive", DEV, n_boot=100)


def test_loader_reads_arm_and_model_from_file_names(tmp_path):
    for (arm, model), df in make_tables().items():
        df.reset_index().to_csv(tmp_path / f"per_patient_{arm}__{model}.csv", index=False)
    (tmp_path / "summary_all.csv").write_text("not,a,per,patient,table\n")
    loaded = load_per_patient_tables(tmp_path)
    assert set(loaded) == set(make_tables())
    assert list(loaded[("subject_standardised", "rf")].index) == PATIENTS