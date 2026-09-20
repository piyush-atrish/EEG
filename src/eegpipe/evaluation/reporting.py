"""Tables and figures (owner: Member C, Step 12). Contracts C6 and C7."""

from __future__ import annotations

import pandas as pd


def save_predictions(pred_df: pd.DataFrame, arm: str, model: str, cfg: dict) -> None:
    """Write contract C6."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def save_tables(per_patient_df: pd.DataFrame, arm: str, model: str, cfg: dict) -> None:
    """Write ``per_patient_{arm}__{model}.csv`` (contract C7)."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def make_summary_table(cfg: dict) -> pd.DataFrame:
    """Build and write ``summary_all.csv``."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")


def plot_per_patient_bars(per_patient_by_arm: dict, metric: str, out_path) -> None:
    """Bar chart of ``metric`` per patient for each arm."""
    raise NotImplementedError("Stub created by Member A at kickoff. Owner implements this.")
