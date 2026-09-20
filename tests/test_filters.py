import numpy as np
import pytest
from scipy import signal

from eegpipe.preprocessing.filters import (
    apply_filters,
    design_bandpass_fir,
    filter_summary,
    freq_response,
    save_response_plot,
)
from tests.fixtures.synth_signals import make_synthetic_raw_array

FS = 256.0


def _psd(x):
    return signal.welch(x, FS, nperseg=1024, detrend=False)


def test_design_matches_documented_numbers(repo_cfg):
    s = filter_summary(FS, repo_cfg)
    assert 900 <= s["numtaps"] <= 960 and s["numtaps"] % 2 == 1
    assert s["group_delay_s"] == pytest.approx(1.82, abs=0.05)
    g = s["gain_db"]
    assert g[0.5] == pytest.approx(-6.0, abs=0.5) and g[45.0] == pytest.approx(-6.0, abs=0.5)
    assert g[0.0] < -50 and g[60.0] < -80
    assert s["passband_ripple_db"] < 0.01
    assert s["worst_stopband_above_46hz_db"] < -58


def test_removes_60hz_keeps_10hz_and_dc_is_removed(repo_cfg):
    x = make_synthetic_raw_array(fs=int(FS), seconds=10.0, n_ch=2, seed=42)
    x += 50.0                                                         # DC offset
    y = apply_filters(x, FS, repo_cfg)
    f, p_in = _psd(x[0] - x[0].mean())
    _, p_out = _psd(y[0])
    i10, i60 = np.argmin(np.abs(f - 10)), np.argmin(np.abs(f - 60))
    assert abs(10 * np.log10(p_in[i10] / p_out[i10])) < 0.5
    assert 10 * np.log10(p_in[i60] / p_out[i60]) > 40
    assert abs(y.mean()) / abs(x.mean()) < 0.01     # relative: > 40 dB DC rejection
    assert np.abs(x).mean() > 40                    # a no-op filter would fail the line above


def test_shape_dtype_and_short_and_1d_inputs(repo_cfg):
    x = np.random.default_rng(0).normal(size=(3, 5000)).astype(np.float32)
    y = apply_filters(x, FS, repo_cfg)
    assert y.shape == x.shape and y.dtype == np.float32
    assert apply_filters(x[0], FS, repo_cfg).shape == (5000,)
    assert apply_filters(x[:, :100], FS, repo_cfg).shape == (3, 100)  # shorter than the filter
    with pytest.raises(ValueError):
        apply_filters(np.zeros((2, 2, 2)), FS, repo_cfg)


def test_no_time_shift(repo_cfg):
    t = np.arange(int(20 * FS)) / FS
    x = np.sin(2 * np.pi * 10 * t + 0.3)[np.newaxis, :].astype(np.float32)
    y = apply_filters(x, FS, repo_cfg)[0]
    mid = slice(int(5 * FS), int(15 * FS))
    corr = signal.correlate(y[mid], x[0][mid], mode="full")
    assert np.argmax(corr) - (len(y[mid]) - 1) == 0


def test_channelwise_equals_all_at_once_reference(repo_cfg):
    """The memory-saving per-channel loop must give the same numbers as filtering the array."""
    x = np.random.default_rng(3).normal(size=(4, 4000)).astype(np.float32)
    prep = repo_cfg["preprocessing"]
    h = design_bandpass_fir(FS, *prep["bandpass_hz"], prep["fir"]["ripple_db"],
                            prep["fir"]["transition_hz"])
    b, a = signal.iirnotch(prep["notch_hz"], prep["notch_q"], FS)
    pad = len(h) // 2
    ref = signal.filtfilt(b, a, x.astype(np.float64), axis=-1)
    ref = np.pad(ref, [(0, 0), (pad, pad)], mode="reflect")
    ref = signal.oaconvolve(ref, h[np.newaxis, :], mode="same", axes=-1)[..., pad:-pad]
    np.testing.assert_allclose(apply_filters(x, FS, repo_cfg), ref, atol=1e-4)


def test_response_plot_is_written_to_the_given_path_only(tmp_path, repo_cfg):
    h = design_bandpass_fir(FS, 0.5, 45.0, 60.0, 1.0)
    out = save_response_plot(h, FS, tmp_path / "sub" / "r.png")
    assert out.exists() and out.stat().st_size > 1000
    w, mag = freq_response(h, FS)
    assert len(w) == len(mag)
