"""Tests for eegpipe.features.nonlinear and .wavelet (Member B, Step 8).

Tolerances and expected sub-band mappings are taken from values verified
by direct numeric probing during development, not guessed.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from eegpipe.features.nonlinear import (
    approximate_entropy,
    nonlinear_features,
    sample_entropy,
)
from eegpipe.features.wavelet import wavelet_features

FS = 256.0
N = 1024  # one 4 s window at 256 Hz
T = np.arange(N) / FS

BASE_CFG = {
    "features": {
        "entropy": {"enabled": True, "m": 2, "r_factor": 0.2, "decimate": 2},
        "wavelet": {"name": "db4", "level": 5},
    }
}

NONLINEAR_KEYS = ["sampen", "apen"]
WAVELET_KEYS = [
    "dwt_logE_d1", "dwt_logE_d2", "dwt_logE_d3", "dwt_logE_d4", "dwt_logE_d5", "dwt_logE_a5",
    "dwt_std_d1", "dwt_std_d2", "dwt_std_d3", "dwt_std_d4", "dwt_std_d5", "dwt_std_a5",
]


# ---------------------------------------------------------------------------
# Slow, independently-written pure-Python/NumPy reference implementations,
# used only to cross-check the numba kernels (different code path: uses
# vectorised broadcasting per row rather than the numba triple nested loop).
# ---------------------------------------------------------------------------


def _slow_sample_entropy(x: np.ndarray, m: int, r: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if r <= 0.0 or n <= m + 1:
        return float("nan")
    n_templates = n - m
    templates_m = np.array([x[i:i + m] for i in range(n_templates)])
    templates_m1 = np.array([x[i:i + m + 1] for i in range(n_templates)])

    b_count = 0
    a_count = 0
    for i in range(n_templates):
        dist_m = np.max(np.abs(templates_m - templates_m[i]), axis=1)
        b_matches = dist_m <= r
        b_matches[i] = False  # exclude self-match
        b_count += int(np.sum(b_matches))

        dist_m1 = np.max(np.abs(templates_m1 - templates_m1[i]), axis=1)
        a_matches = dist_m1 <= r
        a_matches[i] = False
        a_count += int(np.sum(a_matches))

    if a_count == 0 or b_count == 0:
        return float("nan")
    return -np.log(a_count / b_count)


def _slow_phi(x: np.ndarray, m: int, r: float) -> float:
    n = len(x)
    count = n - m + 1
    if count <= 0:
        return float("nan")
    templates = np.array([x[i:i + m] for i in range(count)])
    total = 0.0
    for i in range(count):
        dist = np.max(np.abs(templates - templates[i]), axis=1)
        matches = int(np.sum(dist <= r))  # self-match included, as Pincus defines it
        total += np.log(matches / count)
    return total / count


def _slow_approximate_entropy(x: np.ndarray, m: int, r: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if r <= 0.0 or n <= m + 1:
        return float("nan")
    phi_m = _slow_phi(x, m, r)
    phi_m1 = _slow_phi(x, m + 1, r)
    if np.isnan(phi_m) or np.isnan(phi_m1):
        return float("nan")
    return phi_m - phi_m1


# ---------------------------------------------------------------------------
# sample_entropy / approximate_entropy (numba kernels)
# ---------------------------------------------------------------------------


def test_numba_sample_entropy_matches_slow_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    r = 0.2 * np.std(x)
    fast = sample_entropy(x, 2, r)
    slow = _slow_sample_entropy(x, 2, r)
    assert fast == pytest.approx(slow, abs=1e-6)


def test_numba_approximate_entropy_matches_slow_reference():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    r = 0.2 * np.std(x)
    fast = approximate_entropy(x, 2, r)
    slow = _slow_approximate_entropy(x, 2, r)
    assert fast == pytest.approx(slow, abs=1e-6)


def test_white_noise_has_higher_entropy_than_sine():
    rng = np.random.default_rng(2)
    n = 512
    noise = rng.normal(size=n)
    fs_dec = 128.0
    t_dec = np.arange(n) / fs_dec
    sine = np.sin(2 * np.pi * 5.0 * t_dec)

    r_noise = 0.2 * np.std(noise)
    r_sine = 0.2 * np.std(sine)

    assert sample_entropy(noise, 2, r_noise) > sample_entropy(sine, 2, r_sine)
    assert approximate_entropy(noise, 2, r_noise) > approximate_entropy(sine, 2, r_sine)


def test_constant_signal_returns_nan_without_crashing():
    x = np.full(200, 5.0)
    assert np.isnan(sample_entropy(x, 2, 0.2 * np.std(x)))
    assert np.isnan(approximate_entropy(x, 2, 0.2 * np.std(x)))


def test_too_short_signal_returns_nan():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isnan(sample_entropy(x, 2, 1.0))
    assert np.isnan(approximate_entropy(x, 2, 1.0))


# ---------------------------------------------------------------------------
# nonlinear_features (batched wrapper)
# ---------------------------------------------------------------------------


def test_nonlinear_features_key_set_shape_dtype():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, N))
    feats = nonlinear_features(x, FS, BASE_CFG)
    assert set(feats.keys()) == set(NONLINEAR_KEYS)
    for key in NONLINEAR_KEYS:
        assert feats[key].shape == (3,)
        assert feats[key].dtype == np.float32


def test_nonlinear_features_disabled_returns_empty_dict():
    cfg = {"features": {"entropy": {"enabled": False, "m": 2, "r_factor": 0.2, "decimate": 2}}}
    feats = nonlinear_features(np.zeros((2, N)), FS, cfg)
    assert feats == {}


def test_nonlinear_features_constant_window_is_nan_not_crash():
    x = np.full((2, N), 4.0)
    feats = nonlinear_features(x, FS, BASE_CFG)
    assert np.all(np.isnan(feats["sampen"]))
    assert np.all(np.isnan(feats["apen"]))


def test_nonlinear_features_batch_equals_per_window_loop():
    rng = np.random.default_rng(4)
    batch = rng.normal(size=(4, 2, N))
    batched = nonlinear_features(batch, FS, BASE_CFG)
    for i in range(4):
        for c in range(2):
            single = nonlinear_features(batch[i, c], FS, BASE_CFG)
            for key in NONLINEAR_KEYS:
                b_val, s_val = float(batched[key][i, c]), float(single[key])
                if np.isnan(s_val):
                    assert np.isnan(b_val)
                else:
                    assert b_val == pytest.approx(s_val, abs=1e-4)


def test_nonlinear_features_single_window_shape():
    rng = np.random.default_rng(5)
    feats = nonlinear_features(rng.normal(size=(18, N)), FS, BASE_CFG)
    for key in NONLINEAR_KEYS:
        assert feats[key].shape == (18,)


@pytest.mark.slow
def test_entropy_timing_100_windows_18_channels(caplog):
    """Informational timing (not asserted), per README Step 8's test list."""
    rng = np.random.default_rng(6)
    batch = rng.normal(size=(100, 18, N))
    # Warm up numba JIT compilation before timing.
    nonlinear_features(batch[:1], FS, BASE_CFG)

    t0 = time.perf_counter()
    nonlinear_features(batch, FS, BASE_CFG)
    elapsed = time.perf_counter() - t0

    per_window = elapsed / 100
    logging.getLogger(__name__).info(
        "Entropy cost: %.3fs total for 100 windows x 18 channels (%.4fs/window). "
        "This is the number that decides Step 9's full-cohort runtime.",
        elapsed, per_window,
    )
    print(f"\n[timing] entropy: {elapsed:.3f}s / 100 windows x 18ch = {per_window:.4f}s per window")
    assert elapsed > 0  # sanity only; no performance assertion


# ---------------------------------------------------------------------------
# wavelet_features
# ---------------------------------------------------------------------------


def test_wavelet_key_set_shape_and_order():
    feats = wavelet_features(np.zeros((3, N)), BASE_CFG)
    assert list(feats.keys()) == WAVELET_KEYS
    for key in WAVELET_KEYS:
        assert feats[key].shape == (3,)
        assert feats[key].dtype == np.float32


@pytest.mark.parametrize("freq_hz,expected_subband", [(6.0, "d5"), (12.0, "d4"), (24.0, "d3")])
def test_dwt_log_energy_peaks_in_expected_subband(freq_hz, expected_subband):
    # Verified empirically: at fs=256Hz, level=5, db4, these tones each
    # produce their largest sub-band energy in the listed sub-band.
    tone = np.sin(2 * np.pi * freq_hz * T)
    feats = wavelet_features(tone, BASE_CFG)
    log_energies = {name: float(feats[f"dwt_logE_{name}"]) for name in ["d1", "d2", "d3", "d4", "d5", "a5"]}
    assert max(log_energies, key=log_energies.get) == expected_subband


def test_wavelet_batch_equals_per_window_loop():
    rng = np.random.default_rng(7)
    batch = rng.normal(size=(4, 2, N))
    batched = wavelet_features(batch, BASE_CFG)
    for i in range(4):
        for c in range(2):
            single = wavelet_features(batch[i, c], BASE_CFG)
            for key in WAVELET_KEYS:
                assert batched[key][i, c] == pytest.approx(float(single[key]), abs=1e-3)


def test_wavelet_single_window_shape():
    feats = wavelet_features(np.zeros((18, N)), BASE_CFG)
    for key in WAVELET_KEYS:
        assert feats[key].shape == (18,)


def test_wavelet_features_no_nan_on_zero_input():
    feats = wavelet_features(np.zeros((2, N)), BASE_CFG)
    for key in WAVELET_KEYS:
        assert np.all(np.isfinite(feats[key]))
