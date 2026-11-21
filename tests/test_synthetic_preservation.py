"""Tests for adaptive/synthetic.py and adaptive/preservation.py (the chunk 2.6 screen)."""

import numpy as np
import pytest

from eegpipe.adaptive.preservation import (
    contrast_metrics,
    run_one,
    screen,
    window_separability,
)
from eegpipe.adaptive.synthetic import KINDS, background, burst_waveform, make_recording

FS = 256
# small recordings keep the tests quick: 2 channels, 120 s of background, 30 s warm start
SMALL = dict(fs=FS, pre_s=120.0, post_s=40.0, n_channels=2, warm_start_s=30.0)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def test_background_has_requested_rms_and_is_reproducible():
    a = background(3, 20 * FS, FS, 20.0, np.random.default_rng(1))
    b = background(3, 20 * FS, FS, 20.0, np.random.default_rng(1))
    c = background(3, 20 * FS, FS, 20.0, np.random.default_rng(2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    for channel in a:
        assert rms(channel) == pytest.approx(20.0, rel=1e-6)


def test_background_is_pink_not_white():
    x = background(4, 120 * FS, FS, 20.0, np.random.default_rng(3))
    freqs = np.fft.rfftfreq(x.shape[1], 1 / FS)
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    low = power[:, (freqs >= 0.5) & (freqs < 4)].mean()
    high = power[:, (freqs >= 30) & (freqs < 45)].mean()
    assert low > 5 * high  # a white signal would give a ratio near 1


def test_burst_waveforms_are_unit_rms_and_start_and_end_smoothly():
    for kind in KINDS:
        wave = burst_waveform(kind, 20.0, FS)
        assert len(wave) == 20 * FS and rms(wave) == pytest.approx(1.0, rel=1e-6)
        assert abs(wave[0]) < 0.05 and abs(wave[-1]) < 0.05
    with pytest.raises(ValueError):
        burst_waveform("sine", 5.0, FS)


def test_recording_burst_has_the_requested_amplitude_ratio():
    for kind in KINDS:
        rec = make_recording(20.0, kind=kind, ratio=3.0, n_channels=3, pre_s=60.0, post_s=10.0, seed=4)
        x, onset, end = rec["x"], rec["onset"], rec["end"]
        assert end - onset == 20 * FS and x.shape == (3, end + 10 * FS)
        for channel in x:
            # exact by construction: burst interval RMS = ratio x the requested background RMS (20 uV)
            assert rms(channel[onset:end]) == pytest.approx(3.0 * 20.0, rel=1e-6)
    with pytest.raises(ValueError):
        make_recording(10.0, ratio=1.0)


def test_a_filter_that_changes_nothing_keeps_everything():
    rec = make_recording(30.0, "spikewave", ratio=2.0, n_channels=2, pre_s=120.0, post_s=40.0, seed=5)
    metrics = contrast_metrics(rec["x"], rec["x"], rec["onset"], rec["end"], FS)
    assert metrics["kept_mean"] == pytest.approx(1.0) and metrics["kept_tail"] == pytest.approx(1.0)
    assert metrics["post_ratio"] == pytest.approx(1.0)


def test_separability_is_high_for_a_burst_and_low_for_background_alone():
    rec = make_recording(30.0, "spikewave", ratio=2.0, n_channels=2, pre_s=120.0, post_s=40.0, seed=6)
    assert window_separability(rec["x"], rec["onset"], rec["end"], FS) > 0.6
    only_background = background(2, 200 * FS, FS, 20.0, np.random.default_rng(7))
    assert window_separability(only_background, 100 * FS, 130 * FS, FS) < 0.3
    assert np.isnan(window_separability(rec["x"], 2 * FS, rec["end"], FS))  # no background windows


def test_slow_kalman_tracking_keeps_more_than_fast_tracking():
    slow = run_one("kalman", {"q_over_r": 1e-5}, 30, "spikewave", seed=0, **SMALL)
    fast = run_one("kalman", {"q_over_r": 1e-2}, 30, "spikewave", seed=0, **SMALL)
    assert slow["kept_mean"] > fast["kept_mean"] + 0.2
    assert slow["kept_sep"] > 0.95


def test_screen_removes_fast_settings_and_keeps_slow_ones():
    settings = [
        ("rls", {"length": 8, "forgetting": 0.9999}),
        ("rls", {"length": 8, "forgetting": 0.995}),
        ("kalman", {"q_over_r": 1e-5}),
        ("kalman", {"q_over_r": 1e-2}),
    ]
    detail, summary = screen(
        settings, burst_lengths=(15, 30), seeds=(0,), rule_lengths=(15, 30),
        threshold=0.88, post_s=40.0, pre_s=120.0, n_channels=2, warm_start_s=30.0,
    )
    verdict = summary.set_index("tag")["passes"]
    assert verdict["rls_L8_lam0.9999"] and verdict["kalman_qr1e-05_blk1"]
    assert not verdict["rls_L8_lam0.995"] and not verdict["kalman_qr0.01_blk1"]
    assert {"kept_sep", "kept_mean", "kept_tail", "post_ratio"} <= set(detail.columns)
    assert len(detail) == 4 * 2 * 2  # settings x burst lengths x kinds (one seed)
    assert list(summary.columns) == ["method", "tag", "worst_kept", "passes"]
