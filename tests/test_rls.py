"""Tests for adaptive/rls.py on synthetic AR signals (no real data needed)."""

import numpy as np
import pytest

from eegpipe.adaptive.nlms import NLMSFilter
from eegpipe.adaptive.rls import RLSFilter, rls_filter


def ar2(n, a1=1.5, a2=-0.75, sigma=1.0, seed=0):
    """AR(2) process with known predictor [a1, a2] and innovation variance sigma**2."""
    noise = np.random.default_rng(seed).normal(0.0, sigma, n)
    x = np.zeros(n)
    for i in range(2, n):
        x[i] = a1 * x[i - 1] + a2 * x[i - 2] + noise[i]
    return x


@pytest.fixture(scope="module")
def signal():
    return ar2(30_000)


def test_converges_to_the_true_predictor_and_innovation_power(signal):
    filt = RLSFilter(1, length=4, forgetting=0.999)
    err = filt.process(signal[None])[0]
    assert filt.weights[0, :2] == pytest.approx([1.5, -0.75], abs=0.15)
    assert np.mean(err[-5_000:] ** 2) == pytest.approx(1.0, abs=0.1)


def test_converges_faster_than_nlms(signal):
    rls = RLSFilter(1, length=4, forgetting=0.999).process(signal[:600][None])[0]
    nlms = NLMSFilter(1, length=4, mu=0.02).process(signal[:600][None])[0]
    assert np.mean(rls[200:] ** 2) < 0.85 * np.mean(nlms[200:] ** 2)


def test_matches_batch_ridge_regression_when_nothing_is_forgotten():
    x = ar2(500, seed=3)
    length, delta = 4, 1e-3
    filt = RLSFilter(1, length=length, forgetting=1.0, delta=delta)
    filt.process(x[None])
    u = np.zeros((len(x), length))
    for i in range(len(x)):
        for k in range(length):
            if i - 1 - k >= 0:
                u[i, k] = x[i - 1 - k]
    expected = np.linalg.solve(u.T @ u + delta * np.eye(length), u.T @ x)
    assert filt.weights[0] == pytest.approx(expected, abs=1e-8)


def test_tracks_a_change_in_the_background():
    sig = np.concatenate([ar2(20_000, 1.5, -0.75, 1.0, 1), ar2(20_000, -1.0, -0.6, 1.0, 2)])
    err = RLSFilter(1, length=4, forgetting=0.995).process(sig[None])[0]
    assert np.mean(err[20_000:20_500] ** 2) > 2 * np.mean(err[-5_000:] ** 2)  # the change is felt
    assert np.mean(err[-5_000:] ** 2) < 1.1  # and absorbed


def test_warm_start_removes_the_initial_transient(signal):
    cold = RLSFilter(1, length=4, forgetting=0.999).process(signal[:500][None])[0]
    warmed = RLSFilter(1, length=4, forgetting=0.999)
    warmed.warm_start(signal[2_000:12_000][None])
    warm = warmed.process(signal[:500][None])[0]
    assert np.mean(warm[:200] ** 2) < np.mean(cold[:200] ** 2)


def test_is_causal(signal):
    cut = 3_000
    altered = signal.copy()
    altered[cut + 1 :] = np.random.default_rng(5).normal(0.0, 10.0, len(signal) - cut - 1)
    a = rls_filter(signal[None], length=8, forgetting=0.999)[0]
    b = rls_filter(altered[None], length=8, forgetting=0.999)[0]
    assert np.array_equal(a[: cut + 1], b[: cut + 1])


def test_chunked_processing_equals_single_pass(signal):
    x = signal[:6_000][None]
    whole = rls_filter(x, length=8, forgetting=0.999)
    filt = RLSFilter(1, length=8, forgetting=0.999)
    pieces = [filt.process(x[:, :2_345]), filt.process(x[:, 2_345:])]
    assert np.array_equal(whole, np.concatenate(pieces, axis=1))


def test_state_can_be_copied_to_another_filter(signal):
    x = signal[:4_000][None]
    first = RLSFilter(1, length=8, forgetting=0.999)
    first.process(x[:, :2_000])
    second = RLSFilter(1, length=8, forgetting=0.999)
    second.set_state(first.get_state())
    assert np.array_equal(first.process(x[:, 2_000:]), second.process(x[:, 2_000:]))


def test_channels_are_independent(signal):
    base = np.stack([signal[:4_000], ar2(4_000, seed=3)])
    changed = base.copy()
    changed[1] = ar2(4_000, -1.0, -0.6, 5.0, seed=9)
    a = rls_filter(base, length=8, forgetting=0.999)
    b = rls_filter(changed, length=8, forgetting=0.999)
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[1], b[1])


def test_silent_input_changes_nothing():
    filt = RLSFilter(2, length=8)
    out = filt.process(np.zeros((2, 2_000)))
    assert np.all(out == 0.0) and np.all(filt.weights == 0.0)
    assert np.allclose(filt.p[0], np.eye(8))  # P must not inflate while nothing is learned


def test_long_silence_does_not_wind_up_the_covariance():
    seq = np.concatenate([ar2(5_000, seed=4), np.zeros(50_000), ar2(5_000, seed=5)])
    filt = RLSFilter(1, length=4, forgetting=0.999)
    out = filt.process(seq[None])[0]
    assert np.isfinite(out).all()
    assert np.mean(out[-2_000:] ** 2) < 1.3  # back to normal right after the silence
    assert np.trace(filt.p[0]) <= filt.trace_cap


def test_pure_tone_keeps_the_covariance_bounded_symmetric_and_positive():
    tone = (50.0 * np.sin(2 * np.pi * 10 * np.arange(60_000) / 256))[None]
    filt = RLSFilter(1, length=4, forgetting=0.999)
    out = filt.process(tone)
    assert np.isfinite(out).all() and np.isfinite(filt.weights).all()
    p = filt.p[0]
    assert np.trace(p) <= filt.trace_cap * (1 + 1e-9)
    assert np.allclose(p, p.T)
    assert np.linalg.eigvalsh(p).min() > -1e-9


def test_output_type_and_shape(signal):
    x = np.stack([signal[:800], signal[800:1_600], signal[1_600:2_400]]).astype(np.float32)
    out = rls_filter(x, length=8)
    assert out.shape == x.shape and out.dtype == np.float32


def test_invalid_arguments_are_rejected():
    for kwargs in ({"length": 0}, {"forgetting": 0.0}, {"forgetting": 1.01}, {"delta": 0.0},
                   {"eps": 0.0}, {"cap_factor": 0.5}):
        with pytest.raises(ValueError):
            RLSFilter(1, **kwargs)
    filt = RLSFilter(2)
    with pytest.raises(ValueError):
        filt.process(np.zeros((3, 100)))
    with pytest.raises(ValueError):
        filt.process(np.array([[0.0, np.nan], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        filt.set_state({"weights": np.zeros((1, 16)), "history": np.zeros((1, 16)), "p": np.zeros((1, 16, 16))})
