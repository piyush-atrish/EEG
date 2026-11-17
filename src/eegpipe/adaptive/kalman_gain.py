"""Kalman gain tracker: causal, adaptive amplitude normalisation (the third adaptive method).

Each channel is cut into blocks of ``block_s`` seconds. For block k the observation is the
log-RMS of the raw input,

    z_k = 0.5 * ln(mean(x^2 over the block) + floor)       (clipped to [min_rms, max_rms])

and a scalar Kalman filter tracks the slowly varying log-amplitude s as a random walk:

    predict:  P <- P + q                 update:  K = P / (P + r);  s <- s + K (z - s);  P <- (1 - K) P

The output of the stage is  y(n) = x(n) * exp(-s_hat)  where s_hat is the estimate from the
blocks *before* the current one (one-block lag), so a seizure block never normalises itself.
Only the ratio q/r matters for the steady-state behaviour: the gain K settles near sqrt(q/r), i.e.
a time constant of about 1/K blocks (q/r = 1e-4, 1e-3, 1e-2 -> roughly 100, 30, 10 blocks).

This is the time-varying cousin of per-patient standardisation: it removes slow changes of a
patient's amplitude scale, but it does *not* whiten the spectrum (that is what NLMS/RLS do).

Properties covered by unit tests: causal; chunked processing (even splitting a block) is
bit-identical to one pass; channels independent; scale-equivariant (``y(a x) = y(x)`` after the
first block); a flat channel stays flat and finite.
"""

from __future__ import annotations

import numpy as np
from numba import njit

DEFAULT_BLOCK_S = 1.0
DEFAULT_Q_OVER_R = 1e-3
DEFAULT_R = 1.0
FLOOR = 1e-6  # added to the block mean-square before the log (microvolt^2)
DEFAULT_MIN_RMS = 0.1  # microvolts: observations are clipped to [min_rms, max_rms]
DEFAULT_MAX_RMS = 1e4

# layout of the per-channel state vector
_INIT, _S, _P, _SUMSQ, _COUNT = 0, 1, 2, 3, 4


@njit(cache=True)
def _kalman_gain_1d(x, v, block, q, r, z_min, z_max):
    """Process one channel; ``v`` (5,) holds [initialised, s, P, block sum of squares, count]."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        gain = np.exp(-v[1]) if v[0] > 0.5 else 1.0
        out[i] = x[i] * gain
        v[3] += x[i] * x[i]
        v[4] += 1.0
        if v[4] >= block:
            z = 0.5 * np.log(v[3] / block + 1e-6)
            if z < z_min:
                z = z_min
            elif z > z_max:
                z = z_max
            if v[0] < 0.5:
                v[1] = z
                v[2] = r
                v[0] = 1.0
            else:
                p_pred = v[2] + q
                k = p_pred / (p_pred + r)
                v[1] = v[1] + k * (z - v[1])
                v[2] = (1.0 - k) * p_pred
            v[3] = 0.0
            v[4] = 0.0
    return out


class KalmanGainTracker:
    """Bank of independent per-channel log-amplitude trackers with carry-over state."""

    def __init__(
        self,
        n_channels: int,
        fs: float = 256.0,
        block_s: float = DEFAULT_BLOCK_S,
        q_over_r: float = DEFAULT_Q_OVER_R,
        r: float = DEFAULT_R,
        min_rms: float = DEFAULT_MIN_RMS,
        max_rms: float = DEFAULT_MAX_RMS,
    ) -> None:
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if fs <= 0 or block_s <= 0 or int(round(block_s * fs)) < 2:
            raise ValueError("fs and block_s must give a block of at least 2 samples")
        if q_over_r <= 0 or r <= 0:
            raise ValueError("q_over_r and r must be positive")
        if not 0 < min_rms < max_rms:
            raise ValueError("need 0 < min_rms < max_rms")
        self.n_channels = int(n_channels)
        self.block = int(round(block_s * fs))
        self.q = float(q_over_r) * float(r)
        self.r = float(r)
        self.z_min = float(np.log(min_rms))
        self.z_max = float(np.log(max_rms))
        self.state = np.zeros((self.n_channels, 5), dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Normalise ``x`` of shape (n_channels, n_samples); returns float32 of the same shape."""
        data = np.asarray(x)
        if data.ndim != 2 or data.shape[0] != self.n_channels:
            raise ValueError(f"expected shape ({self.n_channels}, n_samples), got {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError("input contains NaN or Inf; the filter state would be corrupted")
        out = np.empty(data.shape, dtype=np.float32)
        for c in range(self.n_channels):
            channel = np.ascontiguousarray(data[c], dtype=np.float64)
            out[c] = _kalman_gain_1d(
                channel, self.state[c], self.block, self.q, self.r, self.z_min, self.z_max
            )
        return out

    def warm_start(self, x: np.ndarray) -> None:
        """Converge the amplitude estimate on a segment, then drop the partial block."""
        self.process(x)
        self.state[:, _SUMSQ] = 0.0
        self.state[:, _COUNT] = 0.0

    def log_amplitude(self) -> np.ndarray:
        """Current log-RMS estimate per channel (NaN until the first block has completed)."""
        est = self.state[:, _S].copy()
        est[self.state[:, _INIT] < 0.5] = np.nan
        return est

    def get_state(self) -> dict:
        return {"state": self.state.copy()}

    def set_state(self, state: dict) -> None:
        if state["state"].shape != self.state.shape:
            raise ValueError("state shape does not match this filter")
        self.state[:] = state["state"]


def kalman_gain_filter(
    x: np.ndarray,
    fs: float = 256.0,
    block_s: float = DEFAULT_BLOCK_S,
    q_over_r: float = DEFAULT_Q_OVER_R,
    warm_start_samples: int = 0,
) -> np.ndarray:
    """One-shot convenience: normalise a whole (n_channels, n_samples) array from a fresh state."""
    data = np.asarray(x)
    filt = KalmanGainTracker(data.shape[0], fs=fs, block_s=block_s, q_over_r=q_over_r)
    if warm_start_samples > 0:
        filt.warm_start(data[:, :warm_start_samples])
    return filt.process(data)
