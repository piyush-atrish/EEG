"""Regenerate every report figure from the saved result tables (chunk 8.1).

Reads results/tables and results/logs, writes PNGs to results/figures/report/. A figure whose input
tables do not exist yet is skipped with a message, so the script can be run at any stage.

Usage (from the repo root):
    python scripts/07_make_figures.py
    python scripts/07_make_figures.py --out-dir results/figures/report --dev-arms raw_dev5 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from eegpipe.config import load_config
from eegpipe.evaluation import figures as fg
from eegpipe.evaluation.compare import load_per_patient_tables
from eegpipe.utils.logging_utils import get_logger

DEV_ARMS = ["raw_dev5", "subject_standardised_dev5", "adaptive_kalman_qr3e-05_blk1_dev5",
            "adaptive_rls_L8_lam0.9999_dev5"]
EVAL_ARMS = ["raw", "subject_standardised", "adaptive_kalman_qr3e-05_blk1",
             "adaptive_rls_L8_lam0.9999", "adaptive_nlms_L8_mu0.005"]  # arms without results are skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--dev-arms", nargs="+", default=DEV_ARMS)
    parser.add_argument("--eval-arms", nargs="+", default=EVAL_ARMS)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger = get_logger("07_make_figures")
    tables = Path(cfg["paths"]["tables"])
    logs = Path(cfg["paths"]["logs"])
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["figures"]) / "report"

    def read(path: Path, **kw) -> pd.DataFrame | None:
        if not path.exists():
            logger.warning("Missing %s", path)
            return None
        return pd.read_csv(path, **kw)

    per_patient = load_per_patient_tables(tables)
    jobs = []
    screen = read(tables / "adaptive_screen_summary.csv")
    if screen is not None:
        jobs.append(("fig_screen", lambda: fg.fig_screen(screen, out_dir / "screen.png")))
    status = read(logs / "adaptive_status.csv")
    if status is not None:
        jobs.append(("fig_convergence", lambda: fg.fig_convergence(status, out_dir / "convergence.png")))
    stage_a = read(tables / "pilot_stage_a.csv")
    if stage_a is not None:
        by_feature = read(tables / "pilot_stage_a_by_feature.csv", index_col=0)
        jobs.append(("fig_stage_a", lambda: fg.fig_stage_a(stage_a, by_feature, out_dir / "stage_a.png")))
    for name, arms, label in (("dev5_auc", args.dev_arms, "Development patients (chb01-05)"),
                              ("eval10_auc", args.eval_arms, "All 10 patients (nested LOSO)")):
        present = [a for a in arms if any(a == key[0] for key in per_patient)]
        if present:
            jobs.append((f"fig_per_patient_{name}", lambda p=present, n=name, t=label:
                         fg.fig_per_patient_auc(per_patient, p, out_dir / f"{n}.png", t)))
    comparison = read(tables / "arm_comparison.csv")
    if comparison is not None:
        jobs.append(("fig_forest", lambda: fg.fig_forest(comparison, out_dir / "forest_auc.png")))
    event = read(tables / "event_per_patient.csv")
    if event is not None:
        jobs.append(("fig_event_tradeoff", lambda: fg.fig_event_tradeoff(event, out_dir / "event_tradeoff.png")))
    direction = read(tables / "feature_direction_summary.csv", index_col=0)
    if direction is not None:
        jobs.append(("fig_direction", lambda: fg.fig_direction(direction, out_dir / "direction.png")))

    failed = 0
    for name, job in jobs:
        try:
            logger.info("%s -> %s", name, job())
        except Exception as exc:  # one bad table must not stop the other figures
            failed += 1
            logger.error("%s failed: %s: %s", name, type(exc).__name__, exc)
    logger.info("Wrote %d figure(s), %d failed, to %s", len(jobs) - failed, failed, out_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
