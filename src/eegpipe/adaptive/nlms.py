"""NLMS adaptive linear predictor used as a causal, label-free whitening stage.

For each channel the filter predicts x(n) from its own past L samples,

    y(n) = w' u(n),            u(n) = [x(n-1), ..., x(n-L)]
    e(n) = x(n) - y(n)                                   <- the output of the stage
    w   <- w + mu * e(n) * u(n) / (eps + ||u(n)||^2)     (normalised LMS)

With slow adaptation (small ``mu``) the weights follow the patient's *background* spectrum, so
``e`` is what that background cannot predict; a seizure remains visible. Large ``mu`` lets the
filter learn the seizure rhythm and cancel it. See docs/adaptive_design.md.

Properties relied on elsewhere (each is covered by a unit test):

* causal: output at sample n depends only on samples <= n;
* channels are independent;
* state (weights + history) can be carried across calls, so processing a recording in several
  chunks gives bit-identical output to one pass;
* scale-equivariant: multiplying the input by a constant multiplies the output by it. NLMS
  whitens the spectral *shape* but does not normalise amplitude (that is the gain tracker's job);
* flat/silent input never updates the weights and never produces NaN or Inf.
"""

from __future__ import annotations

import numpy as np
from numba import njit

DEFAULT_LENGTH = 16
DEFAULT_MU = 0.005
DEFAULT_EPS = 1e-6


@njit(cache=True)
def _nlms_1d(x, w, hist, mu, eps):
    """Process one channel. ``w`` (L,) and ``hist`` (L,) are updated in place.

    ``hist[0]`` is the most recent past sample. Returns the prediction error e (same length as x).
    """
    n = x.shape[0]
    length = w.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        y = 0.0
        power = 0.0
        for k in range(length):
            y += w[k] * hist[k]
            power += hist[k] * hist[k]
        err = x[i] - y
        out[i] = err
        if power > eps:  # skip silent stretches: nothing to learn, and no division blow-up
            gain = mu * err / (eps + power)
            for k in range(length):
                w[k] += gain * hist[k]
        for k in range(length - 1, 0, -1):
            hist[k] = hist[k - 1]
        hist[0] = x[i]
    return out


class NLMSFilter:
    """Bank of independent per-channel NLMS predictors with carry-over state."""

    def __init__(
        self,
        n_channels: int,
        length: int = DEFAULT_LENGTH,
        mu: float = DEFAULT_MU,
        eps: float = DEFAULT_EPS,
    ) -> None:
        if n_channels < 1 or length < 1:
            raise ValueError("n_channels and length must be >= 1")
        if not 0.0 < mu < 2.0:
            raise ValueError("mu must be in (0, 2) for NLMS to be stable")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.n_channels = int(n_channels)
        self.length = int(length)
        self.mu = float(mu)
        self.eps = float(eps)
        self.weights = np.zeros((self.n_channels, self.length), dtype=np.float64)
        self.history = np.zeros((self.n_channels, self.length), dtype=np.float64)

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
            out[c] = _nlms_1d(channel, self.weights[c], self.history[c], self.mu, self.eps)
        return out

    def warm_start(self, x: np.ndarray) -> None:
        """Converge the weights on a segment (e.g. the first 60 s), then forget its history.

        Afterwards ``process`` can start again from sample 0 with converged weights. Uses no
        labels and only the supplied segment.
        """
        self.process(x)
        self.history[:] = 0.0

    def get_state(self) -> dict:
        return {"weights": self.weights.copy(), "history": self.history.copy()}

    def set_state(self, state: dict) -> None:
        weights, history = state["weights"], state["history"]
        if weights.shape != self.weights.shape or history.shape != self.history.shape:
            raise ValueError("state shape does not match this filter")
        self.weights[:] = weights
        self.history[:] = history


def nlms_filter(
    x: np.ndarray,
    length: int = DEFAULT_LENGTH,
    mu: float = DEFAULT_MU,
    eps: float = DEFAULT_EPS,
    warm_start_samples: int = 0,
) -> np.ndarray:
    """One-shot convenience: filter a whole (n_channels, n_samples) array from a fresh state."""
    data = np.asarray(x)
    filt = NLMSFilter(data.shape[0], length=length, mu=mu, eps=eps)
    if warm_start_samples > 0:
        filt.warm_start(data[:, :warm_start_samples])
    return filt.process(data)