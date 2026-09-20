"""Entropy features (owner: Member B, Step 8)."""

from __future__ import annotations

import numpy as np


def nonlinear_features(x: np.ndarray, fs: float, cfg: dict) -> dict[str, np.ndarray]:
    """``x`` has shape (..., n_samples); returns sampen and apen (empty dict if disabled)."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")
