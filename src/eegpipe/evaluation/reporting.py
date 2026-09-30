"""Tables and figures (owner: Member C, Step 12). Contracts C6 and C7."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eegpipe.evaluation.metrics import aggregate_metrics
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import (
    ensure_parent,
    per_patient_table_path,
    predictions_path,
    summary_table_path,
)

logger = get_logger(__name__)


def save_predictions(pred_df: pd.DataFrame, arm: str, model: str, cfg: dict) -> None:
    """Write Contract C6: `results/predictions/{arm}__{model}.parquet` (scored windows only,
    calibration windows already excluded by `run_loso`)."""
    path = ensure_parent(predictions_path(cfg, arm, model))
    pred_df.to_parquet(path, index=False)
    logger.info("Wrote %d prediction row(s) to %s", len(pred_df), path)


def save_tables(per_patient_df: pd.DataFrame, arm: str, model: str, cfg: dict) -> None:
    """Write Contract C7's per-patient table:
    `results/tables/per_patient_{arm}__{model}.csv`."""
    path = ensure_parent(per_patient_table_path(cfg, arm, model))
    per_patient_df.to_csv(path, index=False)
    logger.info("Wrote %d patient row(s) to %s", len(per_patient_df), path)


def make_summary_table(cfg: dict) -> pd.DataFrame:
    """Build and write Contract C7's `results/tables/summary_all.csv`.

    Reads back every `per_patient_{arm}__{model}.csv` already written by `save_tables` for the
    arms and models listed in `cfg["evaluation"]`, runs each through `aggregate_metrics`, and
    concatenates the results into one long table: `arm, model, metric, mean, std, median,
    n_patients`. A combination whose per-patient file doesn't exist yet is skipped with a
    warning (so a partial run -- e.g. one model still in progress -- doesn't crash this step).
    """
    rows = []
    for arm in cfg["evaluation"]["arms"]:
        for model in cfg["evaluation"]["models"]:
            path = per_patient_table_path(cfg, arm, model)
            if not path.exists():
                logger.warning("%s not found; skipping %s/%s in the summary", path, arm, model)
                continue
            per_patient = pd.read_csv(path)
            agg = aggregate_metrics(per_patient)
            agg.insert(0, "model", model)
            agg.insert(0, "arm", arm)
            rows.append(agg)

    summary = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["arm", "model", "metric", "mean", "std", "median", "n_patients"])
    )
    path = ensure_parent(summary_table_path(cfg))
    summary.to_csv(path, index=False)
    logger.info("Wrote summary table (%d row(s)) to %s", len(summary), path)
    return summary


def plot_per_patient_bars(per_patient_by_arm: dict, metric: str, out_path) -> None:
    """Grouped bar chart of `metric`, one group of bars per patient, one bar per arm.

    Parameters
    ----------
    per_patient_by_arm : dict[str, pd.DataFrame]
        `{arm_name: per_patient_df}`, e.g. `{"raw": df_raw, "subject_standardised": df_std}`.
        Every DataFrame must have a `patient` column and a `metric` column; NaN values (a
        metric undefined for a patient, e.g. no ictal windows) are plotted as a gap, not a
        zero-height bar -- a missing value must never look like "the model scored zero".
    metric : str
        Column to plot, e.g. "sensitivity", "specificity" or "auc".
    out_path : path-like
        Written as PNG; parent directories are created if needed.
    """
    arms = list(per_patient_by_arm)
    patients = sorted(set().union(*(df["patient"] for df in per_patient_by_arm.values())))

    x = np.arange(len(patients))
    width = 0.8 / max(len(arms), 1)

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(patients) + 2), 4))
    for i, arm in enumerate(arms):
        values = per_patient_by_arm[arm].set_index("patient").reindex(patients)[metric]
        ax.bar(x + i * width, values.to_numpy(), width=width, label=arm)

    ax.set_xticks(x + width * (len(arms) - 1) / 2)
    ax.set_xticklabels(patients, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per patient")
    ax.legend()
    fig.tight_layout()

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote figure to %s", out_path)