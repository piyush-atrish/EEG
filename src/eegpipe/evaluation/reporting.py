"""Saving predictions (C6), per-patient tables and the summary table (C7), and figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eegpipe.evaluation.metrics import PER_PATIENT_COLUMNS, aggregate_metrics  # noqa: E402
from eegpipe.utils.logging_utils import get_logger  # noqa: E402
from eegpipe.utils.paths import predictions_path  # noqa: E402

logger = get_logger(__name__)

SUMMARY_COLUMNS = ["arm", "model", "metric", "mean", "std", "median", "n_patients"]


def per_patient_table_path(cfg: dict, arm: str, model: str) -> Path:
    """``results/tables/per_patient_{arm}__{model}.csv``."""
    return Path(cfg["paths"]["tables"]) / f"per_patient_{arm}__{model}.csv"


def save_predictions(pred_df: pd.DataFrame, arm: str, model: str, cfg: dict) -> Path:
    """Write contract C6 to ``results/predictions/{arm}__{model}.parquet`` (git-ignored)."""
    path = predictions_path(cfg, arm, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(path, index=False)
    logger.info("saved %d predictions -> %s", len(pred_df), path)
    return path


def save_tables(per_patient_df: pd.DataFrame, arm: str, model: str, cfg: dict) -> Path:
    """Write the per-patient metrics table (contract C7) for one arm and model."""
    path = per_patient_table_path(cfg, arm, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_patient_df[PER_PATIENT_COLUMNS].to_csv(path, index=False)
    logger.info("saved per-patient table (%d patients) -> %s", len(per_patient_df), path)
    return path


def make_summary_table(cfg: dict) -> pd.DataFrame:
    """Aggregate every saved per-patient table into ``summary_all.csv``.

    Columns: ``arm, model, metric, mean, std, median, n_patients`` (contract
    C7). ``n_patients`` counts the patients with a defined value for that
    metric; ``std`` is the sample SD across patients (ddof=1).
    """
    ev = cfg["evaluation"]
    frames = []
    for arm in ev["arms"]:
        for model in ev["models"]:
            path = per_patient_table_path(cfg, arm, model)
            if not path.exists():
                logger.warning("missing %s; skipped in summary", path)
                continue
            agg = aggregate_metrics(pd.read_csv(path))
            ignored = agg.loc[agg["n_ignored"] > 0, ["metric", "n_ignored"]]
            for _, r in ignored.iterrows():
                logger.info("%s/%s: %s undefined for %d patient(s)", arm, model, r["metric"], r["n_ignored"])
            agg.insert(0, "model", model)
            agg.insert(0, "arm", arm)
            frames.append(agg[SUMMARY_COLUMNS])
    if not frames:
        raise FileNotFoundError(f"no per-patient tables found in {cfg['paths']['tables']}")
    summary = pd.concat(frames, ignore_index=True)
    out = Path(cfg["paths"]["tables"]) / "summary_all.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    logger.info("saved summary (%d rows) -> %s", len(summary), out)
    return summary


def plot_per_patient_bars(per_patient_by_arm: dict[str, pd.DataFrame], metric: str,
                          out_path: str | Path) -> Path:
    """Grouped bar chart of one metric per patient, one bar per arm.

    Parameters
    ----------
    per_patient_by_arm : dict
        ``{arm_name: per_patient_df}`` for a single model.
    metric : str
        Column to plot (for example ``sensitivity``, ``specificity``, ``auc``).
    out_path : path
        PNG destination (parent directories are created).
    """
    arms = list(per_patient_by_arm)
    patients = sorted(set().union(*[set(d["patient"]) for d in per_patient_by_arm.values()]))
    x = np.arange(len(patients))
    width = 0.8 / max(len(arms), 1)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(patients) + 2), 4))
    for i, arm in enumerate(arms):
        d = per_patient_by_arm[arm].set_index("patient")[metric].reindex(patients)
        vals = d.to_numpy(dtype=float)
        mean = np.nanmean(vals) if np.isfinite(vals).any() else float("nan")
        ax.bar(x + (i - (len(arms) - 1) / 2) * width, np.nan_to_num(vals), width,
               label=f"{arm} (mean {mean:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=60, ha="right")
    ax.set_xlabel("held-out patient")
    ax.set_ylabel(metric)
    if metric != "fa_per_hour":
        ax.set_ylim(0, 1.05)
    ax.set_title(f"{metric} per held-out patient (LOSO)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
