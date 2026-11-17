"""Tests for adaptive/kalman_gain.py on synthetic noise with known amplitude."""

import numpy as np
import pytest

from eegpipe.adaptive.kalman_gain import KalmanGainTracker, kalman_gain_filter

FS = 256


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def noise(seconds, amplitude, seed):
    return np.random.default_rng(seed).normal(0.0, amplitude, int(seconds * FS))


def test_output_is_normalised_to_unit_rms_whatever_the_input_scale():
    for amplitude in (5.0, 50.0, 500.0):
        x = noise(120, amplitude, seed=1)[None]
        y = kalman_gain_filter(x, q_over_r=1e-2)[0]
        assert rms(y[-30 * FS :]) == pytest.approx(1.0, abs=0.04)


def test_output_does_not_depend_on_input_scale_after_the_first_block():
    x = np.stack([noise(60, 10.0, 2), noise(60, 10.0, 3)])
    plain = kalman_gain_filter(x)
    scaled = kalman_gain_filter(7.5 * x)
    assert np.max(np.abs(plain[:, FS:] - scaled[:, FS:])) < 1e-4


def test_follows_an_amplitude_step_with_a_lag_and_then_settles():
    x = np.concatenate([noise(120, 10.0, 4), noise(120, 40.0, 5)])[None]
    y = kalman_gain_filter(x, q_over_r=1e-2)[0]
    just_after = rms(y[120 * FS : 123 * FS])
    assert 2.5 < just_after < 4.5  # the jump is mostly still visible (one-block lag)
    assert rms(y[-20 * FS :]) == pytest.approx(1.0, abs=0.06)  # and absorbed later


def test_slower_tracking_keeps_more_of_a_seizure_like_burst():
    def contrast(q_over_r):
        x = np.concatenate([noise(300, 10.0, 6), noise(30, 50.0, 7), noise(60, 10.0, 8)])[None]
        y = kalman_gain_filter(x, q_over_r=q_over_r)[0]
        return rms(y[301 * FS : 330 * FS]) / rms(y[200 * FS : 300 * FS])  # input contrast is 5

    slow, medium, fast = contrast(1e-4), contrast(1e-3), contrast(1e-2)
    assert slow > medium > fast
    assert slow > 3.5 and fast < slow - 1.0


def test_gain_of_a_block_never_depends_on_that_block():
    x = noise(60, 10.0, 9)[None]
    altered = x.copy()
    block = slice(10 * FS, 11 * FS)
    altered[:, block] *= 100.0
    a = kalman_gain_filter(x)
    b = kalman_gain_filter(altered)
    ratio_a = a[:, block] / x[:, block]
    ratio_b = b[:, block] / altered[:, block]
    assert np.allclose(ratio_a, ratio_b, rtol=1e-5)  # same gain: the block cannot normalise itself
    assert not np.allclose(a[:, 12 * FS :], b[:, 12 * FS :])  # ...but later blocks do notice it


def test_is_causal():
    x = noise(60, 10.0, 10)[None]
    cut = 20 * FS + 37  # not on a block boundary
    altered = x.copy()
    altered[:, cut + 1 :] = noise(60, 80.0, 11)[cut + 1 :]
    a = kalman_gain_filter(x)[0]
    b = kalman_gain_filter(altered)[0]
    assert np.array_equal(a[: cut + 1], b[: cut + 1])


def test_chunked_processing_equals_single_pass_even_inside_a_block():
    x = np.stack([noise(40, 10.0, 12), noise(40, 3.0, 13)])
    whole = kalman_gain_filter(x)
    filt = KalmanGainTracker(2)
    pieces = [filt.process(x[:, :1_000]), filt.process(x[:, 1_000:])]  # 1000 is mid-block
    assert np.array_equal(whole, np.concatenate(pieces, axis=1))


def test_state_can_be_copied_to_another_filter():
    x = noise(40, 10.0, 14)[None]
    first = KalmanGainTracker(1)
    first.process(x[:, : 15 * FS + 100])
    second = KalmanGainTracker(1)
    second.set_state(first.get_state())
    assert np.array_equal(first.process(x[:, 15 * FS + 100 :]), second.process(x[:, 15 * FS + 100 :]))


def test_channels_are_independent():
    base = np.stack([noise(30, 10.0, 15), noise(30, 10.0, 16)])
    changed = base.copy()
    changed[1] = noise(30, 300.0, 17)
    a = kalman_gain_filter(base)
    b = kalman_gain_filter(changed)
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[1], b[1])


def test_warm_start_removes_the_first_block_transient():
    x = noise(90, 40.0, 18)[None]
    cold = KalmanGainTracker(1).process(x[:, :FS])
    warmed = KalmanGainTracker(1)
    warmed.warm_start(x[:, 30 * FS :])
    warm = warmed.process(x[:, :FS])
    assert rms(cold) > 30.0  # no estimate yet: the first block passes through unchanged
    assert rms(warm) < 1.3


def test_flat_channel_stays_flat_and_the_estimate_is_clipped():
    filt = KalmanGainTracker(1, min_rms=0.1)
    out = filt.process(np.zeros((1, 10 * FS)))
    assert np.all(out == 0.0) and np.isfinite(out).all()
    assert filt.log_amplitude()[0] == pytest.approx(np.log(0.1))


def test_huge_amplitudes_are_clipped_instead_of_overflowing():
    filt = KalmanGainTracker(1, max_rms=1e4)
    out = filt.process(noise(10, 1e6, 19)[None])
    assert np.isfinite(out).all()
    assert filt.log_amplitude()[0] == pytest.approx(np.log(1e4))


def test_no_estimate_before_the_first_block_completes():
    filt = KalmanGainTracker(1)
    filt.process(noise(0.5, 10.0, 20)[None])
    assert np.isnan(filt.log_amplitude()).all()


def test_output_type_and_shape():
    x = np.stack([noise(5, 10.0, 21)] * 3).astype(np.float32)
    out = kalman_gain_filter(x)
    assert out.shape == x.shape and out.dtype == np.float32


def test_invalid_arguments_are_rejected():
    for kwargs in ({"fs": 0.0}, {"block_s": 0.0}, {"block_s": 0.001}, {"q_over_r": 0.0},
                   {"r": 0.0}, {"min_rms": 5.0, "max_rms": 1.0}):
        with pytest.raises(ValueError):
            KalmanGainTracker(1, **kwargs)
    with pytest.raises(ValueError):
        KalmanGainTracker(0)
    filt = KalmanGainTracker(2)
    with pytest.raises(ValueError):
        filt.process(np.zeros((3, 100)))
    with pytest.raises(ValueError):
        filt.process(np.array([[0.0, np.nan], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        filt.set_state({"state": np.zeros((1, 5))})
