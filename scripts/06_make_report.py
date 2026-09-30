"""CLI: build summary_all.csv and per-patient figures from saved baseline results
(Member C, Step 12).

Usage
-----
    python scripts/06_make_report.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse

import pandas as pd

from eegpipe.config import load_config
from eegpipe.evaluation.reporting import make_summary_table, plot_per_patient_bars
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import figure_path, per_patient_table_path

logger = get_logger(__name__)

# Sensitivity, specificity and AUC per patient, raw vs subject_standardised -- the three
# figures README Section 11 (Step 12) and docs/baseline_report.md ask for.
_REPORT_METRICS = ("sensitivity", "specificity", "auc")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build summary_all.csv and per-patient figures from saved baseline results."
    )
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    arms = cfg["evaluation"]["arms"]
    models = cfg["evaluation"]["models"]

    summary = make_summary_table(cfg)
    logger.info("Summary table: %d row(s)", len(summary))

    n_figures = 0
    for model in models:
        per_patient_by_arm = {}
        for arm in arms:
            path = per_patient_table_path(cfg, arm, model)
            if not path.exists():
                logger.warning("%s not found; skipping %s/%s in the figures", path, arm, model)
                continue
            per_patient_by_arm[arm] = pd.read_csv(path)

        if len(per_patient_by_arm) < 2:
            logger.warning("Fewer than 2 arms available for model=%s; skipping its figures", model)
            continue

        for metric in _REPORT_METRICS:
            out_path = figure_path(cfg, f"{metric}_per_patient_{model}.png")
            plot_per_patient_bars(per_patient_by_arm, metric, out_path)
            n_figures += 1

    logger.info("Wrote %d figure(s)", n_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())