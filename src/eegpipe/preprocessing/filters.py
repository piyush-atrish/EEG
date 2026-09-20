"""General pre-processing filters: 60 Hz notch + 0.5-45 Hz linear-phase Kaiser FIR band-pass.

Design notes
------------
* The FIR is designed with ``scipy.signal.kaiserord`` (stop-band attenuation ``ripple_db``,
  transition width ``transition_hz``). ``firwin`` places the cutoffs at the *centre* of each
  transition band, so with a 1 Hz transition both 0.5 Hz and 45 Hz sit at -6 dB and the flat
  passband is roughly 1.0-44.5 Hz. Use :func:`filter_summary` for the exact numbers.
* The filter is applied one channel at a time (float64 per channel, float32 output), which
  gives identical results to filtering the whole array but a much lower memory peak.
* Zero net delay: the signal is reflect-padded by ``numtaps // 2`` and the convolution is
  trimmed back, so the output is aligned with the input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal


def design_bandpass_fir(
    fs: float, low: float, high: float, ripple_db: float, transition_hz: float
) -> np.ndarray:
    """Kaiser-window linear-phase FIR band-pass (odd length)."""
    numtaps, beta = signal.kaiserord(ripple_db, transition_hz / (fs / 2.0))
    if numtaps % 2 == 0:
        numtaps += 1
    return signal.firwin(
        numtaps, [low, high], window=("kaiser", beta), pass_zero=False, fs=fs
    )


def design_notch(fs: float, f0: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """IIR notch (applied zero-phase with ``filtfilt``)."""
    return signal.iirnotch(f0, q, fs)


def _designs(fs: float, cfg: dict):
    prep = cfg["preprocessing"]
    low, high = prep["bandpass_hz"]
    fir = prep["fir"]
    h = design_bandpass_fir(fs, low, high, fir["ripple_db"], fir["transition_hz"])
    b, a = design_notch(fs, prep["notch_hz"], prep["notch_q"])
    return h, b, a


def apply_filters(x: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Notch then band-pass. ``x`` is ``(n_ch, n_samples)`` (or 1-D); returns float32.

    Output length equals input length and there is no time shift.
    """
    x = np.asarray(x)
    squeeze = x.ndim == 1
    x2 = x[np.newaxis, :] if squeeze else x
    if x2.ndim != 2:
        raise ValueError("x must be 1-D or 2-D (n_channels, n_samples)")
    h, b, a = _designs(fs, cfg)
    pad = len(h) // 2
    out = np.empty(x2.shape, dtype=np.float32)
    for c in range(x2.shape[0]):
        channel = signal.filtfilt(b, a, x2[c].astype(np.float64))
        padded = np.pad(channel, (pad, pad), mode="reflect")
        out[c] = signal.oaconvolve(padded, h, mode="same")[pad:-pad]
    return out[0] if squeeze else out


def freq_response(h: np.ndarray, fs: float, n: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies in Hz and magnitude in dB of an FIR filter."""
    w, resp = signal.freqz(h, worN=n, fs=fs)
    return w, 20.0 * np.log10(np.abs(resp) + 1e-12)


def filter_summary(fs: float, cfg: dict) -> dict:
    """Measured properties of the configured filter (used for docs and tests)."""
    h, _, _ = _designs(fs, cfg)
    w, mag = freq_response(h, fs, n=65536)

    def gain(f: float) -> float:
        return float(mag[np.argmin(np.abs(w - f))])

    return {
        "numtaps": len(h),
        "group_delay_s": (len(h) - 1) / 2.0 / fs,
        "gain_db": {f: gain(f) for f in (0.0, 0.25, 0.5, 0.75, 1.0, 10.0, 44.5, 45.0, 45.5, 46.0, 60.0)},
        "passband_ripple_db": float(np.max(np.abs(mag[(w >= 2.0) & (w <= 43.0)]))),
        "worst_stopband_above_46hz_db": float(np.max(mag[w >= 46.0])),
    }


def save_response_plot(h: np.ndarray, fs: float, out_path: str | Path) -> Path:
    """Save a magnitude/phase response figure (matplotlib is imported lazily)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, mag = freq_response(h, fs)
    _, resp = signal.freqz(h, worN=len(w), fs=fs)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(w, mag, "b")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)", color="b")
    ax1.set_ylim(-120, 5)
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(w, np.unwrap(np.angle(resp)), "g")
    ax2.set_ylabel("Phase (rad)", color="g")
    plt.title("Kaiser FIR band-pass (-6 dB at 0.5 and 45 Hz)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """``python -m eegpipe.preprocessing.filters``: print the summary and save the figure."""
    from eegpipe.config import find_repo_root, load_config

    cfg = load_config()
    fs = float(cfg["dataset"]["fs"])
    h, _, _ = _designs(fs, cfg)
    for key, value in filter_summary(fs, cfg).items():
        print(f"{key}: {value}")
    out = save_response_plot(h, fs, find_repo_root() / "docs" / "figures" / "A_fir_response.png")
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
