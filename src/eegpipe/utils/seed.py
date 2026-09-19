import random

import numpy as np


def set_global_seed(seed: int):
    """Set the random seed for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
