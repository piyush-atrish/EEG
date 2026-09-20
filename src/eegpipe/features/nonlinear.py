"""Nonlinear (entropy) features (Contract C5, `catalog.nonlinear`).

Member B, Step 8. `sample_entropy`/`approximate_entropy` are 1-D numba
kernels (no numba parallelism: parallelism lives at case level in Step 9,
to avoid nesting parallel regions). `nonlinear_features` is the batched,
config-driven wrapper that flattens arbitrary leading dimensions, decimates,
and loops over rows calling the numba kernels (the loop itself is Python,
but each call is compiled; this matches the spec's "call the numba
functions in a loop over flattened rows").
"""
from __future__ import annotations

import logging

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)


@njit(cache=True)
def sample_entropy(x: np.ndarray, m: int, r: float) -> float:
    """Sample entropy of a 1-D signal, `SampEn(m, r) = -ln(A / B)`.

    `B` counts pairs of length-`m` template vectors (excluding
    self-matches) within Chebyshev distance `r`; `A` counts the same pairs
    extended to length `m + 1`. Undefined (returns `nan`) when `r <= 0`
    (degenerate tolerance, as for a constant window), the signal is too
    short (`n <= m + 1`), or no template pair matches (`A == 0` or `B ==
    0`) -- Step 9 replaces `nan` with `0.0` and counts the replacement.
    """
    n = x.shape[0]
    if r <= 0.0 or n <= m + 1:
        return np.nan

    b_count = 0
    a_count = 0
    for i in range(n - m):
        for j in range(n - m):
            if i == j:
                continue
            matches_m = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > r:
                    matches_m = False
                    break
            if matches_m:
                b_count += 1
                if abs(x[i + m] - x[j + m]) <= r:
                    a_count += 1

    if b_count == 0 or a_count == 0:
        return np.nan
    return -np.log(a_count / b_count)


@njit(cache=True)
def _phi(x: np.ndarray, m: int, r: float) -> float:
    """Pincus's `Phi^m(r)`: mean log fraction of length-m vectors that match each other
    (self-match included)."""
    n = x.shape[0]
    count = n - m + 1
    if count <= 0:
        return np.nan
    total = 0.0
    for i in range(count):
        matches = 0
        for j in range(count):
            is_match = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > r:
                    is_match = False
                    break
            if is_match:
                matches += 1
        total += np.log(matches / count)
    return total / count


@njit(cache=True)
def approximate_entropy(x: np.ndarray, m: int, r: float) -> float:
    """Approximate entropy of a 1-D signal, `ApEn(m, r) = Phi^m(r) - Phi^(m+1)(r)`.

    Same `nan` conditions as `sample_entropy` (degenerate `r`, too-short
    signal).
    """
    n = x.shape[0]
    if r <= 0.0 or n <= m + 1:
        return np.nan
    phi_m = _phi(x, m, r)
    phi_m1 = _phi(x, m + 1, r)
    return phi_m - phi_m1


def nonlinear_features(x: np.ndarray, fs: float, cfg: dict) -> dict[str, np.ndarray]:
    """Compute `sampen`/`apen` for every window (Contract C5, `catalog.nonlinear`).

    Parameters
    ----------
    x : np.ndarray
        Shape `(..., n_samples)`. Any number of leading dimensions.
    fs : float
        Sampling rate in Hz of `x` (informational here; entropy operates
        on the decimated signal and does not otherwise use `fs`).
    cfg : dict
        Uses `cfg["features"]["entropy"]` (`enabled`, `m`, `r_factor`,
        `decimate`).

    Returns
    -------
    dict[str, np.ndarray]
        `{"sampen": ..., "apen": ...}`, each shaped `x.shape[:-1]`, dtype
        float32. Empty dict if `entropy.enabled` is False (Section 12
        fallback ladder, step 1).

    Notes
    -----
    Decimation is done by **plain stride-slicing**, not
    `scipy.signal.decimate`. `scipy.signal.decimate` applies its own
    anti-aliasing low-pass filter before downsampling, which is redundant
    here: the window has already been band-limited to 0.5-45 Hz during
    preprocessing (Contract C3), well inside the decimated Nyquist
    frequency (`fs / decimate / 2`). Stride-slicing is cheaper and avoids
    a second filter's edge transients on an already-short 4 s window.
    """
    entropy_cfg = cfg["features"]["entropy"]
    if not entropy_cfg.get("enabled", True):
        return {}

    x = np.asarray(x, dtype=np.float64)
    leading_shape = x.shape[:-1]
    n_samples = x.shape[-1]
    flat = x.reshape(-1, n_samples)

    decimate_factor = int(entropy_cfg["decimate"])
    decimated = flat[:, ::decimate_factor] if decimate_factor > 1 else flat

    m = int(entropy_cfg["m"])
    r_factor = float(entropy_cfg["r_factor"])

    n_rows = decimated.shape[0]
    sampen_flat = np.empty(n_rows, dtype=np.float64)
    apen_flat = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        row = np.ascontiguousarray(decimated[i])
        r = r_factor * np.std(row)
        sampen_flat[i] = sample_entropy(row, m, r)
        apen_flat[i] = approximate_entropy(row, m, r)

    return {
        "sampen": sampen_flat.reshape(leading_shape).astype(np.float32),
        "apen": apen_flat.reshape(leading_shape).astype(np.float32),
    }
