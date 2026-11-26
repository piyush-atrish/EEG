"""Operation counts and latency budget of tools/hardware_table.py (no data, no timing)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from eegpipe.config import load_config

_SPEC = importlib.util.spec_from_file_location(
    "hardware_table", Path(__file__).resolve().parents[1] / "tools" / "hardware_table.py"
)
ht = importlib.util.module_from_spec(_SPEC)
sys.modules["hardware_table"] = ht  # dataclasses look the module up here
_SPEC.loader.exec_module(ht)


def test_nlms_counts_match_kernel_for_l8():
    ops = ht.nlms_ops(8)
    assert (ops.mul, ops.add, ops.div) == (25, 26, 1)


def test_nlms_minimal_is_cheaper_but_grows_linearly():
    assert ht.nlms_ops(16, minimal=True).total < ht.nlms_ops(16).total
    step = ht.nlms_ops(16).total - ht.nlms_ops(8).total
    assert step == pytest.approx(6 * 8)  # 3 mul + 3 add per extra tap


def test_rls_counts_match_kernel_for_l8_and_grow_quadratically():
    ops = ht.rls_ops(8)
    assert (ops.mul, ops.add, ops.div) == (132, 133, 80)
    assert ht.rls_ops(16).total > 3 * ht.rls_ops(8).total
    minimal = ht.rls_ops(8, minimal=True)
    assert minimal.div == 2
    assert minimal.mul + minimal.div == ops.mul + ops.div  # divisions became multiplies


def test_kalman_is_cheapest_and_exp_amortises():
    assert ht.kalman_ops(256).total < ht.nlms_ops(8).total
    assert ht.kalman_ops(256).transcendental > 1.0
    assert ht.kalman_ops(256, minimal=True).transcendental < 0.01


def test_state_sizes():
    assert ht.state_doubles("kalman") == 5
    assert ht.state_doubles("nlms", 8) == 16
    assert ht.state_doubles("rls", 8) == 16 + 64


def test_table_without_timing_has_expected_rows_and_ordering():
    table = ht.build_table(seconds=2.0, timing=False)
    assert len(table) == 7
    assert "measured_us_per_sample" not in table.columns
    cost = dict(zip(table["stage"], table["ops_per_sample_as_implemented"], strict=True))
    assert cost["RLS L=16"] > cost["RLS L=8"] > cost["NLMS L=16"] > cost["NLMS L=8"]
    fir = next(v for k, v in cost.items() if "FIR" in k)
    assert fir > cost["RLS L=16"]  # the preprocessing FIR is the biggest stage of all


def test_symmetric_fir_halves_multiplies_and_rls_division_saving_is_visible():
    plain, sym = ht.fir_ops(931), ht.fir_ops(931, symmetric=True)
    assert plain.mul == 931 and sym.mul == 466 and plain.add == sym.add == 930
    table = ht.build_table(seconds=2.0, timing=False)
    row = table[table["stage"] == "RLS L=8"].iloc[0]
    assert row["div_per_sample"] == 80 and row["div_per_sample_minimal"] == 2


def test_latency_budget_total_matches_components():
    cfg = load_config()
    numtaps, delay = ht.fir_numtaps_and_delay(cfg)
    assert numtaps % 2 == 1
    assert delay == pytest.approx((numtaps - 1) / 2 / 256)
    budget = ht.latency_budget(cfg, min_consecutive=3)
    window = float(cfg["segmentation"]["window_s"])
    step = window * (1 - float(cfg["segmentation"]["overlap"]))
    assert budget["seconds"].iloc[3] == pytest.approx(window + 2 * step)  # as evaluated
    assert budget["seconds"].iloc[-1] == pytest.approx(delay + window + 2 * step)  # deployed
    one = ht.latency_budget(cfg, min_consecutive=1)
    assert one["seconds"].iloc[3] == pytest.approx(window)
