"""Wavelet (DWT) features (Contract C5, `catalog.wavelet`).

Member B, Step 8. 5-level `db4` DWT, log-energy and standard deviation per
sub-band. Vectorised via `pywt.wavedec(..., axis=-1)` over the flattened
leading dimensions (PyWavelets supports batched decomposition along one
axis of an n-D array), so this handles a single window `(n_ch, n_samples)`
or a batch `(n_win, n_ch, n_samples)` with a single call, not a loop.
"""
from __future__ import annotations

import numpy as np
import pywt

# Catalog order (configs/config.yaml -> features.catalog.wavelet):
# log-energy for every sub-band first (d1..d5, a5), then std for every
# sub-band (d1..d5, a5).
_SUBBAND_ORDER = ["d1", "d2", "d3", "d4", "d5", "a5"]


def wavelet_features(x: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Compute 12 DWT sub-band features (Contract C5, `catalog.wavelet`).

    Parameters
    ----------
    x : np.ndarray
        Shape `(..., n_samples)`. Any number of leading dimensions.
    cfg : dict
        Uses `cfg["features"]["wavelet"]` (`name`, `level`).

    Returns
    -------
    dict[str, np.ndarray]
        Keys `dwt_logE_{d1..d5,a5}` (catalog order), then `dwt_std_{d1..d5,a5}`
        (catalog order), each shaped `x.shape[:-1]`, dtype float32.
        `dwt_logE_{b} = log10(sum(coeff**2) + 1e-12)`;
        `dwt_std_{b} = std(coeff)` over that sub-band's coefficients.

    Notes
    -----
    `pywt.wavedec(x, wavelet, level=5, axis=-1)` returns coefficients as
    `[a5, d5, d4, d3, d2, d1]` (coarsest approximation first, then details
    from coarsest to finest). At fs = 256 Hz this splits the already
    band-limited (0.5-45 Hz) signal into approximately: d1 64-128 Hz
    (mostly empty above the band-pass, kept for a fixed catalog), d2
    32-64 Hz, d3 16-32 Hz, d4 8-16 Hz, d5 4-8 Hz, a5 0-4 Hz.
    """
    wavelet_cfg = cfg["features"]["wavelet"]
    wavelet_name = wavelet_cfg["name"]
    level = wavelet_cfg["level"]

    x = np.asarray(x, dtype=np.float64)
    leading_shape = x.shape[:-1]
    n_samples = x.shape[-1]
    flat = x.reshape(-1, n_samples)

    coeffs = pywt.wavedec(flat, wavelet_name, level=level, axis=-1)
    subband_coeffs = dict(zip(["a5", "d5", "d4", "d3", "d2", "d1"], coeffs))

    log_energy: dict[str, np.ndarray] = {}
    std_dev: dict[str, np.ndarray] = {}
    for name in _SUBBAND_ORDER:
        c = subband_coeffs[name]
        energy = np.sum(c**2, axis=-1)
        log_energy[name] = np.log10(energy + 1e-12).reshape(leading_shape).astype(np.float32)
        std_dev[name] = np.std(c, axis=-1).reshape(leading_shape).astype(np.float32)

    result: dict[str, np.ndarray] = {}
    for name in _SUBBAND_ORDER:
        result[f"dwt_logE_{name}"] = log_energy[name]
    for name in _SUBBAND_ORDER:
        result[f"dwt_std_{name}"] = std_dev[name]
    return result
