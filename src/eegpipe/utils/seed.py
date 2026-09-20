"""Global seeding (leakage rule L7: determinism)."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's ``random`` and NumPy's legacy global generator."""
    random.seed(seed)
    np.random.seed(seed)
