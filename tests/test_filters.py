import pytest
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from pathlib import Path
from eegpipe.preprocessing.filters import design_bandpass_fir, apply_filters, freq_response
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
    
    # Create signal with 10Hz and 60Hz + heavy DC offset
    x = make_synthetic_raw_array(fs=fs, seconds=10.0, tones_hz=(10.0, 60.0), n_ch=2, seed=42)
    x += 50.0 
    
    x_filtered = apply_filters(x, fs, cfg)
    
    # Output shape and dtype must be preserved
    assert x_filtered.shape == x.shape
    assert x_filtered.dtype == np.float32
    
    # Measure attenuation using Welch PSD
    f, pxx_in = signal.welch(x[0], fs, nperseg=1024)
    f, pxx_out = signal.welch(x_filtered[0], fs, nperseg=1024)
    
    idx_10 = np.argmin(np.abs(f - 10.0))
    idx_60 = np.argmin(np.abs(f - 60.0))
    idx_dc = np.argmin(np.abs(f - 0.0))
    
    pow_10_in = 10 * np.log10(pxx_in[idx_10])
    pow_10_out = 10 * np.log10(pxx_out[idx_10])
    pow_60_in = 10 * np.log10(pxx_in[idx_60])
    pow_60_out = 10 * np.log10(pxx_out[idx_60])
    pow_dc_out = 10 * np.log10(pxx_out[idx_dc] + 1e-12)
    
    # 10 Hz amplitude preserved within 0.5 dB
    assert np.abs(pow_10_in - pow_10_out) < 0.5
    # 60 Hz attenuated by at least 40 dB
    assert (pow_60_in - pow_60_out) > 40.0
    # DC completely removed
    assert pow_dc_out < -20.0
    
    # Zero delay test (cross-correlation peak at lag 0)
    t = np.arange(int(fs * 2.0)) / fs
    s_10 = np.sin(2 * np.pi * 10 * t).astype(np.float32)[np.newaxis, :]
    s_10_filt = apply_filters(s_10, fs, cfg)
    corr = signal.correlate(s_10[0], s_10_filt[0], mode='full')
    lag = np.argmax(corr) - (len(s_10[0]) - 1)
    assert lag == 0

def test_filter_frequency_response_and_plot():
    fs = 256.0
    h = design_bandpass_fir(fs, 0.5, 45.0, 60.0, 1.0)
    
    w, mag_db = freq_response(h, fs)
    
    # Stop-band above 46 Hz at or below ~-60 dB
    idx_46 = np.argmin(np.abs(w - 46.0))
    assert np.max(mag_db[idx_46:]) <= -55.0
    
    # Save the required validation plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(w, mag_db, 'b')
    ax1.set_ylabel('Magnitude (dB)', color='b')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylim([-90, 5])
    ax1.grid(True)
    
    w, h_resp = signal.freqz(h, worN=8192, fs=fs)
    phase = np.unwrap(np.angle(h_resp))
    ax2 = ax1.twinx()
    ax2.plot(w, phase, 'g')
    ax2.set_ylabel('Phase (radians)', color='g')
    
    plt.title('Kaiser FIR Band-pass (0.5-45 Hz)')
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "A_fir_response.png")
    plt.close()