"""Patient-level statistics: paired Wilcoxon test and bootstrap confidence intervals.

The unit of analysis is the *patient* (or, for detection rates, the seizure), never the
window: windows from one patient are strongly correlated, so treating them as independent
would give overconfident intervals and p-values. Every function here therefore takes one
number per patient (or per seizure).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def min_possible_p(n: int, two_sided: bool = True) -> float:
    """Smallest p-value an exact Wilcoxon signed-rank test can reach with ``n`` non-zero pairs."""
    if n < 1:
        return 1.0
    return min(1.0, (2 if two_sided else 1) / 2**n)


def _paired_arrays(a, b) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of the same length (one value per patient)")
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def paired_wilcoxon(a, b, alternative: str = "two-sided") -> dict:
    """Paired Wilcoxon signed-rank test of ``a`` against ``b`` (one value per patient).

    Pairs with a NaN on either side are dropped, then pairs with zero difference (the
    classical Wilcoxon convention). ``alternative="greater"`` tests whether ``a`` tends to be
    larger than ``b``. SciPy uses the exact distribution for small samples without ties.
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    a, b = _paired_arrays(a, b)
    diff = a - b
    nonzero = diff[diff != 0]
    n_nonzero = len(nonzero)
    if n_nonzero == 0:
        statistic, p_value = float("nan"), 1.0
    else:
        result = wilcoxon(nonzero, alternative=alternative, method="auto")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "n_pairs": len(diff),
        "n_nonzero": n_nonzero,
        "n_a_greater": int((diff > 0).sum()),
        "n_b_greater": int((diff < 0).sum()),
        "mean_diff": float(diff.mean()) if len(diff) else float("nan"),
        "median_diff": float(np.median(diff)) if len(diff) else float("nan"),
        "statistic": statistic,
        "p_value": p_value,
        "min_possible_p": min_possible_p(n_nonzero, alternative == "two-sided"),
    }


def bootstrap_ci(
    values,
    statistic=np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI of ``statistic`` over units (patients or seizures).

    ``statistic`` must accept an ``axis`` argument (``np.mean`` and ``np.median`` do). NaNs are
    dropped. With fewer than two values the interval is undefined and returned as NaN.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    estimate = float(statistic(x))
    if n < 2:
        return {"estimate": estimate, "low": float("nan"), "high": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(0, n, size=(n_boot, n))]
    stats = statistic(samples, axis=1)
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": estimate, "low": float(low), "high": float(high), "n": n}


def paired_bootstrap_ci(a, b, statistic=np.mean, n_boot: int = 10_000,
                        alpha: float = 0.05, seed: int = 0) -> dict:
    """Bootstrap CI of the paired difference ``a - b`` (patients are resampled as pairs)."""
    a, b = _paired_arrays(a, b)
    return bootstrap_ci(a - b, statistic=statistic, n_boot=n_boot, alpha=alpha, seed=seed)
