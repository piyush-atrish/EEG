"""RLS adaptive linear predictor: the fast-converging sibling of the NLMS stage.

Per channel, with u(n) = [x(n-1), ..., x(n-L)] and forgetting factor lambda in (0, 1]:

    k(n) = P u / (lambda + u' P u)
    e(n) = x(n) - w' u                       <- the output of the stage (prediction error)
    w   <- w + k e(n)
    P   <- (P - P u u' P / (lambda + u' P u)) / lambda

The effective memory is about 1 / (1 - lambda) samples (lambda = 0.9995 -> 2000 samples = 8 s at
256 Hz). A long memory learns the patient's background and keeps seizures visible; a short memory
learns the seizure rhythm and cancels it (docs/adaptive_design.md).

Numerical safeguards (each has a unit test):

* P is updated through its upper triangle and mirrored, so it stays exactly symmetric;
* a stretch with (almost) no input power skips the update, so P cannot grow without bound by
  the 1/lambda factor while nothing is being learned;
* the trace of P is capped, so a nearly pure tone (rank-deficient input) cannot wind P up;
* a non-positive denominator (lost positive-definiteness) skips that update instead of
  producing NaN.

Cost is O(L^2) per sample, the most expensive of the three adaptive methods.
"""

from __future__ import annotations

import numpy as np
from numba import njit

DEFAULT_LENGTH = 16
DEFAULT_FORGETTING = 0.9995
DEFAULT_DELTA = 1.0  # P(0) = I / delta; the preprocessed signal is in microvolts
DEFAULT_EPS = 1e-6
DEFAULT_CAP_FACTOR = 100.0  # trace(P) is capped at this many times its initial value


@njit(cache=True)
def _rls_1d(x, w, p, hist, lam, eps, trace_cap):
    """Process one channel. ``w`` (L,), ``p`` (L, L) and ``hist`` (L,) are updated in place."""
    n = x.shape[0]
    length = w.shape[0]
    out = np.empty(n, dtype=np.float64)
    pu = np.empty(length, dtype=np.float64)
    for i in range(n):
        y = 0.0
        power = 0.0
        for a in range(length):
            y += w[a] * hist[a]
            power += hist[a] * hist[a]
        err = x[i] - y
        out[i] = err
        if power > eps:
            denom = lam
            for a in range(length):
                acc = 0.0
                for b in range(length):
                    acc += p[a, b] * hist[b]
                pu[a] = acc
                denom += hist[a] * acc
            if denom > 1e-12:
                for a in range(length):
                    w[a] += pu[a] * err / denom
                trace = 0.0
                for a in range(length):
                    for b in range(a, length):
                        value = (p[a, b] - pu[a] * pu[b] / denom) / lam
                        p[a, b] = value
                        p[b, a] = value
                    trace += p[a, a]
                if trace > trace_cap:
                    scale = trace_cap / trace
                    for a in range(length):
                        for b in range(length):
                            p[a, b] *= scale
        for a in range(length - 1, 0, -1):
            hist[a] = hist[a - 1]
        hist[0] = x[i]
    return out


class RLSFilter:
    """Bank of independent per-channel RLS predictors with carry-over state."""

    def __init__(
        self,
        n_channels: int,
        length: int = DEFAULT_LENGTH,
        forgetting: float = DEFAULT_FORGETTING,
        delta: float = DEFAULT_DELTA,
        eps: float = DEFAULT_EPS,
        cap_factor: float = DEFAULT_CAP_FACTOR,
    ) -> None:
        if n_channels < 1 or length < 1:
            raise ValueError("n_channels and length must be >= 1")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        if delta <= 0.0 or eps <= 0.0 or cap_factor < 1.0:
            raise ValueError("delta and eps must be positive and cap_factor >= 1")
        self.n_channels = int(n_channels)
        self.length = int(length)
        self.forgetting = float(forgetting)
        self.delta = float(delta)
        self.eps = float(eps)
        self.trace_cap = float(cap_factor) * self.length / self.delta
        self.weights = np.zeros((self.n_channels, self.length), dtype=np.float64)
        self.history = np.zeros((self.n_channels, self.length), dtype=np.float64)
        self.p = np.stack([np.eye(self.length) / self.delta for _ in range(self.n_channels)])

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter ``x`` of shape (n_channels, n_samples); returns float32 of the same shape."""
        data = np.asarray(x)
        if data.ndim != 2 or data.shape[0] != self.n_channels:
            raise ValueError(f"expected shape ({self.n_channels}, n_samples), got {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError("input contains NaN or Inf; the filter state would be corrupted")
        out = np.empty(data.shape, dtype=np.float32)
        for c in range(self.n_channels):
            channel = np.ascontiguousarray(data[c], dtype=np.float64)
            out[c] = _rls_1d(
                channel, self.weights[c], self.p[c], self.history[c],
                self.forgetting, self.eps, self.trace_cap,
            )
        return out

    def warm_start(self, x: np.ndarray) -> None:
        """Converge weights and P on a segment, then forget its sample history."""
        self.process(x)
        self.history[:] = 0.0

    def get_state(self) -> dict:
        return {"weights": self.weights.copy(), "history": self.history.copy(), "p": self.p.copy()}

    def set_state(self, state: dict) -> None:
        for key, target in (("weights", self.weights), ("history", self.history), ("p", self.p)):
            if state[key].shape != target.shape:
                raise ValueError("state shape does not match this filter")
        self.weights[:] = state["weights"]
        self.history[:] = state["history"]
        self.p[:] = state["p"]


def rls_filter(
    x: np.ndarray,
    length: int = DEFAULT_LENGTH,
    forgetting: float = DEFAULT_FORGETTING,
    delta: float = DEFAULT_DELTA,
    eps: float = DEFAULT_EPS,
    warm_start_samples: int = 0,
) -> np.ndarray:
    """One-shot convenience: filter a whole (n_channels, n_samples) array from a fresh state."""
    data = np.asarray(x)
    filt = RLSFilter(data.shape[0], length=length, forgetting=forgetting, delta=delta, eps=eps)
    if warm_start_samples > 0:
        filt.warm_start(data[:, :warm_start_samples])
    return filt.process(data)
