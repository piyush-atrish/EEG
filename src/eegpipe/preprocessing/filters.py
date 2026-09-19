import numpy as np
import scipy.signal as signal
from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

def design_bandpass_fir(fs: float, low: float, high: float, ripple_db: float, transition_hz: float) -> np.ndarray:
    """Design a linear-phase Kaiser-window FIR band-pass filter."""
    nyq = fs / 2.0
    numtaps, beta = signal.kaiserord(ripple_db, transition_hz / nyq)
    
    # Force numtaps to be odd for a Type I FIR filter (zero phase shift at Nyquist)
    if numtaps % 2 == 0:
        numtaps += 1
        
    h = signal.firwin(numtaps, [low, high], window=('kaiser', beta), pass_zero=False, fs=fs)
    return h

def design_notch(fs: float, f0: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Design an IIR notch filter."""
    b, a = signal.iirnotch(f0, q, fs)
    return b, a

def apply_filters(x: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Apply 60 Hz notch and Kaiser FIR band-pass with zero delay."""
    notch_hz = cfg["preprocessing"]["notch_hz"]
    notch_q = cfg["preprocessing"]["notch_q"]
    low, high = cfg["preprocessing"]["bandpass_hz"]
    ripple = cfg["preprocessing"]["fir"]["ripple_db"]
    trans = cfg["preprocessing"]["fir"]["transition_hz"]
    
    # 1. Notch filter (Zero-phase via filtfilt)
    b, a = design_notch(fs, notch_hz, notch_q)
    x_notch = signal.filtfilt(b, a, x, axis=-1)
    
    # 2. FIR Band-pass filter (Delay-compensated via oaconvolve)
    h = design_bandpass_fir(fs, low, high, ripple, trans)
    pad_len = len(h) // 2
    
    # Reflect-pad edges to prevent transient ringing
    pad_width = [(0, 0)] * (x.ndim - 1) + [(pad_len, pad_len)]
    x_padded = np.pad(x_notch, pad_width=pad_width, mode='reflect')
    
    # Reshape h for broadcasting over channels if input is 2D
    h_conv = h[np.newaxis, :] if x.ndim == 2 else h
    x_filtered = signal.oaconvolve(x_padded, h_conv, mode='same', axes=-1)
    
    # Trim the padded edges to preserve exact input length
    if pad_len > 0:
        x_filtered = x_filtered[..., pad_len:-pad_len]
        
    return x_filtered.astype(np.float32)

def freq_response(h: np.ndarray, fs: float, n: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Return frequencies in Hz and magnitude response in dB."""
    w, h_resp = signal.freqz(h, worN=n, fs=fs)
    mag_db = 20 * np.log10(np.abs(h_resp) + 1e-12)
    return w, mag_db