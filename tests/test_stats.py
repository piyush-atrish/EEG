"""Tests for evaluation/stats.py."""

import math

import numpy as np
import pytest

from eegpipe.evaluation.stats import (
    bootstrap_ci,
    min_possible_p,
    paired_bootstrap_ci,
    paired_wilcoxon,
)


def test_min_possible_p_matches_exact_test():
    assert min_possible_p(5) == pytest.approx(0.0625)
    assert min_possible_p(6) == pytest.approx(0.03125)
    assert min_possible_p(5, two_sided=False) == pytest.approx(0.03125)
    assert min_possible_p(0) == 1.0


def test_wilcoxon_five_patients_cannot_reach_significance():
    res = paired_wilcoxon([0.9, 0.8, 0.95, 0.85, 0.9], [0.5, 0.4, 0.6, 0.5, 0.55])
    assert res["n_pairs"] == 5 and res["n_a_greater"] == 5 and res["n_b_greater"] == 0
    assert res["p_value"] == pytest.approx(0.0625)
    assert res["p_value"] == pytest.approx(res["min_possible_p"])


def test_wilcoxon_six_consistent_patients_is_significant():
    a = [0.9, 0.8, 0.95, 0.85, 0.9, 0.7]
    b = [0.5, 0.4, 0.6, 0.5, 0.55, 0.3]
    assert paired_wilcoxon(a, b)["p_value"] == pytest.approx(0.03125)
    assert paired_wilcoxon(a, b, alternative="greater")["p_value"] == pytest.approx(0.015625)


def test_wilcoxon_drops_nan_pairs_and_ignores_ties():
    res = paired_wilcoxon([1, 2, np.nan, 5, 5], [0, 1, 3, 5, 4])
    assert res["n_pairs"] == 4 and res["n_nonzero"] == 3


def test_wilcoxon_identical_inputs_give_p_one():
    res = paired_wilcoxon([1, 2, 3], [1, 2, 3])
    assert res["p_value"] == 1.0 and math.isnan(res["statistic"])


def test_wilcoxon_rejects_bad_shapes_and_alternatives():
    with pytest.raises(ValueError):
        paired_wilcoxon([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        paired_wilcoxon([1, 2], [2, 3], alternative="sideways")


def test_bootstrap_is_reproducible_and_contains_estimate():
    values = [0.2, 0.5, 0.9, 0.4, 0.7, 0.6]
    first = bootstrap_ci(values, seed=1)
    second = bootstrap_ci(values, seed=1)
    assert first == second
    assert first["low"] <= first["estimate"] <= first["high"]
    assert first["estimate"] == pytest.approx(np.mean(values))


def test_bootstrap_edge_cases():
    constant = bootstrap_ci([0.5, 0.5, 0.5, 0.5])
    assert constant["low"] == constant["high"] == 0.5
    single = bootstrap_ci([0.7])
    assert single["estimate"] == pytest.approx(0.7) and math.isnan(single["low"])
    assert bootstrap_ci([np.nan, np.nan])["n"] == 0
    assert bootstrap_ci([1, 2, 3, np.nan])["n"] == 3


def test_bootstrap_supports_median_and_binary_detection_rates():
    assert bootstrap_ci([1, 2, 3, 4, 100], statistic=np.median)["estimate"] == 3.0
    detected = [1] * 28 + [0] * 2  # 28 of 30 seizures
    ci = bootstrap_ci(detected, seed=0)
    assert ci["estimate"] == pytest.approx(28 / 30) and ci["low"] < ci["estimate"] <= ci["high"]


def test_paired_bootstrap_of_constant_improvement():
    ci = paired_bootstrap_ci([0.9, 0.8, 0.7, 0.9], [0.4, 0.3, 0.2, 0.4])
    assert ci["estimate"] == pytest.approx(0.5)
    assert ci["low"] == pytest.approx(0.5) and ci["high"] == pytest.approx(0.5)
