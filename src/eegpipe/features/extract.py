"""Feature assembly and runner (owner: Member B, Step 9). Contract C5."""

from __future__ import annotations

import numpy as np
import pandas as pd


def feature_names(cfg: dict) -> list[str]:
    """Full feature column names, catalog order (outer loop) then channel order (inner loop)."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def extract_window_features(win: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """``win`` is (n_ch, n_samples); returns a 1-D float32 vector ordered as feature_names."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def extract_batch_features(batch: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """``batch`` is (n_win, n_ch, n_samples); returns (n_win, n_features) float32."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def extract_case_features(
    case: str,
    file_index: pd.DataFrame,
    window_table: pd.DataFrame,
    cfg: dict,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Return the C4 key columns plus one float32 column per feature and channel."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")
