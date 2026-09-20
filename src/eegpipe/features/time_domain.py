"""Time-domain features (owner: Member B, Step 7)."""

from __future__ import annotations

import numpy as np


def time_domain_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """``x`` has shape (..., n_samples); returns {feature: array of shape x.shape[:-1]}."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")
