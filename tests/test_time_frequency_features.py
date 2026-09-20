"""Tests for eegpipe.features.time_domain and .frequency_domain (Member B, Step 7).

Tolerances below are set from values verified by direct numeric probing
during development (see PR description), not guessed.
"""
from __future__ import annotations

import numpy as np
import pytest

from eegpipe.features.frequency_domain import frequency_features
from eegpipe.features.time_domain import time_domain_features

FS = 256.0
N = 1024  # one 4 s window at 256 Hz
T = np.arange(N) / FS

BASE_CFG = {
    "features": {
        "welch": {"nperseg_s": 1.0, "noverlap_frac": 0.5},
        "bands": {
            "delta": [0.5, 4.0],
            "theta": [4.0, 8.0],
            "alpha": [8.0, 13.0],
            "beta": [13.0, 30.0],
            "gamma": [30.0, 45.0],
        },
    }
}

TIME_KEYS = [
    "mean", "std", "skew", "kurt", "line_length",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
]
FREQ_KEYS = [
    "logpow_delta", "logpow_theta", "logpow_alpha", "logpow_beta", "logpow_gamma",
    "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
]


# ---------------------------------------------------------------------------
# time_domain_features
# ---------------------------------------------------------------------------


def test_time_domain_key_set_and_shape():
    x = np.zeros((3, N))
    feats = time_domain_features(x)
    assert set(feats.keys()) == set(TIME_KEYS)
    for key in TIME_KEYS:
        assert feats[key].shape == (3,)
        assert feats[key].dtype == np.float32


def test_time_domain_output_order_matches_catalog():
    # dict insertion order must match configs/config.yaml catalog.time
    feats = time_domain_features(np.zeros(N))
    assert list(feats.keys()) == TIME_KEYS


def test_constant_signal_gives_zero_activity_no_nan():
    x = np.full(N, 7.0)
    feats = time_domain_features(x)
    for key in TIME_KEYS:
        val = float(feats[key])
        assert np.isfinite(val), f"{key} is not finite: {val}"
    assert feats["hjorth_activity"] == 0.0
    assert feats["std"] == 0.0
    assert feats["skew"] == 0.0
    assert feats["kurt"] == 0.0
    assert feats["hjorth_mobility"] == 0.0
    assert feats["hjorth_complexity"] == 0.0
    assert feats["mean"] == 7.0


def test_line_length_of_a_known_ramp():
    slope = 2.0
    ramp = np.arange(N, dtype=np.float64) * slope
    feats = time_domain_features(ramp)
    expected = (N - 1) * slope  # sum(abs(diff)) of a pure ramp
    assert feats["line_length"] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("freq_hz", [1.0, 5.0, 10.0, 20.0])
def test_hjorth_mobility_matches_analytic_sine_result(freq_hz):
    sine = 3.0 * np.sin(2 * np.pi * freq_hz * T)
    feats = time_domain_features(sine)
    analytic_mobility = 2 * np.pi * freq_hz / FS
    # Verified empirically: discrete-diff approximation is within ~1.5% of
    # the analytic continuous-derivative value up to 20 Hz at fs=256 Hz.
    assert float(feats["hjorth_mobility"]) == pytest.approx(analytic_mobility, rel=0.02)
    # A sine's "second derivative" is again sinusoidal at the same
    # frequency, so mobility(dx) ~= mobility(x) -> complexity ~= 1.
    assert float(feats["hjorth_complexity"]) == pytest.approx(1.0, rel=0.01)


def test_time_domain_batch_equals_per_window_loop():
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(5, 3, N))
    batched = time_domain_features(batch)
    for i in range(5):
        for c in range(3):
            single = time_domain_features(batch[i, c])
            for key in TIME_KEYS:
                assert batched[key][i, c] == pytest.approx(float(single[key]), abs=1e-4)


def test_time_domain_single_window_shape():
    feats = time_domain_features(np.zeros((18, N)))
    for key in TIME_KEYS:
        assert feats[key].shape == (18,)


# ---------------------------------------------------------------------------
# frequency_features
# ---------------------------------------------------------------------------


def test_frequency_key_set_shape_and_order():
    feats = frequency_features(np.zeros(N), FS, BASE_CFG)
    assert list(feats.keys()) == FREQ_KEYS
    for key in FREQ_KEYS:
        assert feats[key].dtype == np.float32


def test_10hz_sine_gives_alpha_dominance():
    sine = 2.0 * np.sin(2 * np.pi * 10.0 * T)
    feats = frequency_features(sine, FS, BASE_CFG)
    # 10 Hz lands exactly on a Welch bin (1 Hz resolution) -> near-total
    # isolation in the alpha band (verified empirically: rel_alpha == 1.0).
    assert feats["rel_alpha"] > 0.99
    logpows = {k: v for k, v in feats.items() if k.startswith("logpow_")}
    assert max(logpows, key=logpows.get) == "logpow_alpha"


def test_2hz_sine_gives_delta_dominance():
    sine = 2.0 * np.sin(2 * np.pi * 2.0 * T)
    feats = frequency_features(sine, FS, BASE_CFG)
    assert feats["rel_delta"] > 0.99
    logpows = {k: v for k, v in feats.items() if k.startswith("logpow_")}
    assert max(logpows, key=logpows.get) == "logpow_delta"


def test_white_noise_gives_roughly_flat_spectrum_by_band_width():
    rng = np.random.default_rng(1)
    noise = rng.normal(size=N)
    feats = frequency_features(noise, FS, BASE_CFG)

    widths = {"delta": 3.5, "theta": 4.0, "alpha": 5.0, "beta": 17.0, "gamma": 15.0}
    total_width = sum(widths.values())
    expected_rel = {band: w / total_width for band, w in widths.items()}

    # Loose tolerance: single-draw white noise, not averaged over many
    # windows, so we check order-of-magnitude agreement with band width,
    # not exact equality.
    for band, expected in expected_rel.items():
        actual = float(feats[f"rel_{band}"])
        assert actual == pytest.approx(expected, abs=0.06), (band, actual, expected)

    # The two widest bands should dominate the two narrowest.
    assert feats["rel_beta"] > feats["rel_delta"]
    assert feats["rel_gamma"] > feats["rel_delta"]


def test_rel_power_sums_to_one_within_the_total_band():
    rng = np.random.default_rng(2)
    x = rng.normal(size=N)
    feats = frequency_features(x, FS, BASE_CFG)
    total_rel = sum(float(feats[f"rel_{b}"]) for b in ("delta", "theta", "alpha", "beta", "gamma"))
    # Bands are contiguous and span the full [0.5, 45] reference range, so
    # relative powers should sum close to 1 (small trapezoid/edge error).
    assert total_rel == pytest.approx(1.0, abs=0.02)


def test_all_zero_window_does_not_produce_nan():
    feats = frequency_features(np.zeros(N), FS, BASE_CFG)
    for key in FREQ_KEYS:
        assert np.isfinite(float(feats[key])), f"{key} is not finite"
        if key.startswith("rel_"):
            assert feats[key] == 0.0


def test_frequency_batch_equals_per_window_loop():
    rng = np.random.default_rng(3)
    batch = rng.normal(size=(4, 2, N))
    batched = frequency_features(batch, FS, BASE_CFG)
    for i in range(4):
        for c in range(2):
            single = frequency_features(batch[i, c], FS, BASE_CFG)
            for key in FREQ_KEYS:
                assert batched[key][i, c] == pytest.approx(float(single[key]), abs=1e-4)


def test_frequency_single_window_shape():
    feats = frequency_features(np.zeros((18, N)), FS, BASE_CFG)
    for key in FREQ_KEYS:
        assert feats[key].shape == (18,)
