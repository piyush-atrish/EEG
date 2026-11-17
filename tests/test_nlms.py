"""Tests for adaptive/nlms.py on synthetic AR signals (no real data needed)."""

import numpy as np
import pytest
from scipy.signal import welch

from eegpipe.adaptive.nlms import NLMSFilter, nlms_filter


def ar2(n, a1=1.5, a2=-0.75, sigma=1.0, seed=0):
    """AR(2) process with known predictor [a1, a2] and innovation variance sigma**2."""
    noise = np.random.default_rng(seed).normal(0.0, sigma, n)
    x = np.zeros(n)
    for i in range(2, n):
        x[i] = a1 * x[i - 1] + a2 * x[i - 2] + noise[i]
    return x


def flatness(sig, fs=256):
    _, psd = welch(sig, fs=fs, nperseg=512)
    return float(np.exp(np.mean(np.log(psd))) / np.mean(psd))


@pytest.fixture(scope="module")
def signal():
    return ar2(60_000)


def test_converges_to_the_true_predictor_and_innovation_power(signal):
    filt = NLMSFilter(1, length=4, mu=0.02)
    err = filt.process(signal[None])[0]
    assert filt.weights[0, :2] == pytest.approx([1.5, -0.75], abs=0.08)
    assert np.mean(err[-10_000:] ** 2) == pytest.approx(1.0, abs=0.12)
    assert np.var(signal) > 5.0  # the input is far from white, the error is nearly white


def test_output_is_whiter_than_input(signal):
    err = nlms_filter(signal[None], length=4, mu=0.02)[0]
    assert flatness(err) > 0.9
    assert flatness(err) > 3 * flatness(signal)


def test_tracks_a_change_in_the_background():
    sig = np.concatenate([ar2(30_000, 1.5, -0.75, 1.0, 1), ar2(30_000, -1.0, -0.6, 1.0, 2)])
    err = nlms_filter(sig[None], length=4, mu=0.05)[0]
    just_after = np.mean(err[30_000:31_000] ** 2)
    late = np.mean(err[-5_000:] ** 2)
    assert just_after > late  # the change is felt...
    assert late < 1.15  # ...and the filter re-converges


def test_warm_start_removes_the_initial_transient(signal):
    cold = NLMSFilter(1, length=4, mu=0.02).process(signal[:2_000][None])[0]
    warmed = NLMSFilter(1, length=4, mu=0.02)
    warmed.warm_start(signal[2_000:20_000][None])
    warm = warmed.process(signal[:2_000][None])[0]
    assert np.mean(warm[:1_000] ** 2) < 0.75 * np.mean(cold[:1_000] ** 2)


def test_is_causal(signal):
    cut = 5_000
    altered = signal.copy()
    altered[cut + 1 :] = np.random.default_rng(5).normal(0.0, 10.0, len(signal) - cut - 1)
    a = nlms_filter(signal[None], length=8, mu=0.01)[0]
    b = nlms_filter(altered[None], length=8, mu=0.01)[0]
    assert np.array_equal(a[: cut + 1], b[: cut + 1])


def test_chunked_processing_equals_single_pass(signal):
    x = signal[:20_000][None]
    whole = nlms_filter(x, length=8, mu=0.01)
    filt = NLMSFilter(1, length=8, mu=0.01)
    pieces = [filt.process(x[:, :7_777]), filt.process(x[:, 7_777:])]
    assert np.array_equal(whole, np.concatenate(pieces, axis=1))


def test_state_can_be_copied_to_another_filter(signal):
    x = signal[:10_000][None]
    first = NLMSFilter(1, length=8, mu=0.01)
    first.process(x[:, :5_000])
    second = NLMSFilter(1, length=8, mu=0.01)
    second.set_state(first.get_state())
    assert np.array_equal(first.process(x[:, 5_000:]), second.process(x[:, 5_000:]))


def test_channels_are_independent(signal):
    base = np.stack([signal[:10_000], ar2(10_000, seed=3)])
    changed = base.copy()
    changed[1] = ar2(10_000, -1.0, -0.6, 5.0, seed=9)
    a = nlms_filter(base, length=8, mu=0.01)
    b = nlms_filter(changed, length=8, mu=0.01)
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[1], b[1])


def test_output_scales_with_input_amplitude(signal):
    x = signal[:10_000][None]
    plain = nlms_filter(x, length=4, mu=0.02, eps=1e-12)
    scaled = nlms_filter(7.5 * x, length=4, mu=0.02, eps=1e-12)
    # NLMS whitens the spectral shape but does not normalise the amplitude.
    assert np.max(np.abs(scaled - 7.5 * plain)) < 1e-5 * np.max(np.abs(7.5 * plain))


def test_silent_input_never_updates_or_produces_nan():
    filt = NLMSFilter(2, length=8, mu=0.05)
    out = filt.process(np.zeros((2, 3_000)))
    assert np.all(out == 0.0) and np.all(filt.weights == 0.0)
    out = filt.process(np.full((2, 3_000), 5.0))  # constant (DC) input stays finite
    assert np.isfinite(out).all() and np.isfinite(filt.weights).all()


def test_output_type_and_shape(signal):
    x = np.stack([signal[:1_000], signal[1_000:2_000], signal[2_000:3_000]]).astype(np.float32)
    out = nlms_filter(x, length=8, mu=0.01)
    assert out.shape == x.shape and out.dtype == np.float32


def test_invalid_arguments_are_rejected():
    for kwargs in ({"length": 0}, {"mu": 0.0}, {"mu": 2.0}, {"eps": 0.0}):
        with pytest.raises(ValueError):
            NLMSFilter(1, **kwargs)
    filt = NLMSFilter(2)
    with pytest.raises(ValueError):
        filt.process(np.zeros((3, 100)))  # wrong channel count
    with pytest.raises(ValueError):
        filt.process(np.array([[0.0, np.nan], [0.0, 1.0]]))  # non-finite input
    with pytest.raises(ValueError):
        filt.set_state({"weights": np.zeros((1, 16)), "history": np.zeros((1, 16))})