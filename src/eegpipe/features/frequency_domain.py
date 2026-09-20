"""Frequency-domain features (Contract C5, `catalog.frequency`).

Member B, Step 7. Welch PSD estimate (Hann window, `features.welch` config),
band power via trapezoidal integration of the PSD over each band, then
log and relative power per band. Vectorised over `axis=-1`: works for a
single window `(n_ch, n_samples)` or a batch `(n_win, n_ch, n_samples)`.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

# Total-power reference band for `rel_{band}`, per Contract C5.
_TOTAL_BAND = (0.5, 45.0)

_TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz  # numpy <2.0 fallback


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """`numerator / denominator`, returning 0.0 (not NaN/inf) wherever denominator <= 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_denominator = np.where(denominator > 0, denominator, 1.0)
        ratio = np.where(denominator > 0, numerator / safe_denominator, 0.0)
    return ratio


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    """Trapezoidal integral of `psd` over frequencies in `[low, high]`.

    `psd` has shape `(..., n_freqs)` and shares its last axis with `freqs`.
    Returns an array shaped `psd.shape[:-1]`. If fewer than 2 frequency
    bins fall in the band, returns zeros (a trapezoidal integral needs at
    least two points).
    """
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() < 2:
        return np.zeros(psd.shape[:-1], dtype=np.float64)
    return _TRAPEZOID(psd[..., mask], x=freqs[mask], axis=-1)


def frequency_features(x: np.ndarray, fs: float, cfg: dict) -> dict[str, np.ndarray]:
    """Compute the 10 frequency-domain features defined in Contract C5.

    Parameters
    ----------
    x : np.ndarray
        Shape `(..., n_samples)`. Any number of leading dimensions.
    fs : float
        Sampling rate in Hz.
    cfg : dict
        Uses `cfg["features"]["welch"]` (`nperseg_s`, `noverlap_frac`) and
        `cfg["features"]["bands"]` (name -> `[low, high]`, Hz).

    Returns
    -------
    dict[str, np.ndarray]
        Keys `logpow_{band}` for every band (catalog order), then
        `rel_{band}` for every band (catalog order), each shaped
        `x.shape[:-1]`, dtype float32. `logpow_{band} = log10(power +
        1e-12)`. `rel_{band} = power / total_power_[0.5,45]Hz`, guarded to
        0.0 when total power is 0 (e.g. an all-zero window).
    """
    x = np.asarray(x, dtype=np.float64)
    welch_cfg = cfg["features"]["welch"]
    bands_cfg = cfg["features"]["bands"]

    nperseg = round(welch_cfg["nperseg_s"] * fs)
    noverlap = round(nperseg * welch_cfg["noverlap_frac"])

    freqs, psd = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, axis=-1)

    total_power = _band_power(freqs, psd, *_TOTAL_BAND)

    band_powers = {name: _band_power(freqs, psd, low, high) for name, (low, high) in bands_cfg.items()}

    result: dict[str, np.ndarray] = {}
    for name, power in band_powers.items():
        result[f"logpow_{name}"] = np.log10(power + 1e-12).astype(np.float32)
    for name, power in band_powers.items():
        result[f"rel_{name}"] = _safe_ratio(power, total_power).astype(np.float32)
    return result
