"""Welch band-power features (owner: Member B, Step 7)."""

from __future__ import annotations

import numpy as np


def frequency_features(x: np.ndarray, fs: float, cfg: dict) -> dict[str, np.ndarray]:
    """``x`` has shape (..., n_samples); returns logpow_* and rel_* features."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")
