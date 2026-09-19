import pytest
import numpy as np
import scipy.signal as signal
from eegpipe.preprocessing.filters import apply_filters
from tests.fixtures.synth_signals import make_synthetic_raw_array

def test_filter_characteristics_and_apply():
    cfg = {
        "preprocessing": {
            "notch_hz": 60.0,
            "notch_q": 30.0,
            "bandpass_hz": [0.5, 45.0],
            "fir": {"window": "kaiser", "ripple_db": 60.0, "transition_hz": 1.0}
        }
    }
    fs = 256.0
    
    # Create signal with 10Hz and 60Hz + heavy DC offset (+50.0)
    x = make_synthetic_raw_array(fs=fs, seconds=10.0, tones_hz=(10.0, 60.0), n_ch=2, seed=42)
    x += 50.0 
    
    x_filtered = apply_filters(x, fs, cfg)
    
    # Output shape and dtype must be preserved
    assert x_filtered.shape == x.shape
    assert x_filtered.dtype == np.float32
    
    # DC COMPLETELY REMOVED: 
    # Use raw mean, NOT signal.welch (which detrends by default and creates a false positive)
    dc_mean_in = np.mean(x)
    dc_mean_out = np.mean(x_filtered)
    assert np.abs(dc_mean_in) > 40.0
    assert np.abs(dc_mean_out) < 0.1
    
    # Measure attenuation using Welch PSD (with detrending off to maintain strict bin power)
    f, pxx_in = signal.welch(x[0], fs, nperseg=1024, detrend=False)
    f, pxx_out = signal.welch(x_filtered[0], fs, nperseg=1024, detrend=False)
    
    idx_10 = np.argmin(np.abs(f - 10.0))
    idx_60 = np.argmin(np.abs(f - 60.0))
    
    pow_10_in = 10 * np.log10(pxx_in[idx_10])
    pow_10_out = 10 * np.log10(pxx_out[idx_10])
    pow_60_in = 10 * np.log10(pxx_in[idx_60])
    pow_60_out = 10 * np.log10(pxx_out[idx_60])
    
    # 10 Hz amplitude preserved within 0.5 dB
    assert np.abs(pow_10_in - pow_10_out) < 0.5
    # 60 Hz attenuated by at least 40 dB
    assert (pow_60_in - pow_60_out) > 40.0
    
    # Zero delay test (cross-correlation peak at lag 0)
    t = np.arange(int(fs * 2.0)) / fs
    s_10 = np.sin(2 * np.pi * 10 * t).astype(np.float32)[np.newaxis, :]
    s_10_filt = apply_filters(s_10, fs, cfg)
    corr = signal.correlate(s_10[0], s_10_filt[0], mode='full')
    lag = np.argmax(corr) - (len(s_10[0]) - 1)
    assert lag == 0