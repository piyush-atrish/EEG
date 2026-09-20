"""Time-domain features (Contract C5, `catalog.time`).

Member B, Step 7. All functions are fully vectorised over `axis=-1`: the
same code path handles a single window `(n_ch, n_samples)` and a batch
`(n_win, n_ch, n_samples)` (or any other leading-dimension shape) with no
Python-level loop over windows.
"""
from __future__ import annotations

import numpy as np

# Keys, in the exact order of `configs/config.yaml -> features.catalog.time`.
_FEATURE_ORDER = [
    "mean",
    "std",
    "skew",
    "kurt",
    "line_length",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
]


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """`numerator / denominator`, returning 0.0 (not NaN/inf) wherever denominator <= 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_denominator = np.where(denominator > 0, denominator, 1.0)
        ratio = np.where(denominator > 0, numerator / safe_denominator, 0.0)
    return ratio


def time_domain_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the 8 time-domain features defined in Contract C5.

    Parameters
    ----------
    x : np.ndarray
        Shape `(..., n_samples)`. Any number of leading dimensions.

    Returns
    -------
    dict[str, np.ndarray]
        Keys `mean, std, skew, kurt, line_length, hjorth_activity,
        hjorth_mobility, hjorth_complexity`, each shaped `x.shape[:-1]`,
        dtype float32. Divisions by a zero variance/mobility are guarded to
        return 0.0 rather than NaN (e.g. a constant window).
    """
    x = np.asarray(x, dtype=np.float64)

    mean = np.mean(x, axis=-1)
    centered = x - mean[..., np.newaxis]
    m2 = np.mean(centered**2, axis=-1)  # population variance, ddof=0
    m3 = np.mean(centered**3, axis=-1)
    m4 = np.mean(centered**4, axis=-1)
    std = np.sqrt(m2)

    skew = _safe_ratio(m3, std**3)
    kurt = np.where(m2 > 0, _safe_ratio(m4, m2**2) - 3.0, 0.0)

    line_length = np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)

    hjorth_activity = m2  # var(x)

    dx = np.diff(x, axis=-1)
    var_dx = np.var(dx, axis=-1)
    hjorth_mobility = np.sqrt(_safe_ratio(var_dx, m2))

    d2x = np.diff(dx, axis=-1)
    var_d2x = np.var(d2x, axis=-1)
    mobility_dx = np.sqrt(_safe_ratio(var_d2x, var_dx))
    hjorth_complexity = _safe_ratio(mobility_dx, hjorth_mobility)

    values = {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurt": kurt,
        "line_length": line_length,
        "hjorth_activity": hjorth_activity,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
    }
    return {name: values[name].astype(np.float32) for name in _FEATURE_ORDER}
