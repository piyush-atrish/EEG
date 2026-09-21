"""Stage 06: build summary_all.csv and the per-patient figures from saved results."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from eegpipe.config import load_config
from eegpipe.evaluation.reporting import make_summary_table, per_patient_table_path, plot_per_patient_bars
from eegpipe.utils.logging_utils import get_logger

logger = get_logger("report")

FIGURE_METRICS = ["sensitivity", "specificity", "auc"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml", help="path to the YAML config")
    p.add_argument("--patients", nargs="*", default=None,
                   help="accepted for interface consistency; the report always uses the saved tables")
    args = p.parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    summary = make_summary_table(cfg)
    print(summary.to_string(index=False))

    fig_dir = Path(cfg["paths"]["figures"])
    n_figs = 0
    for model in cfg["evaluation"]["models"]:
        by_arm = {}
        for arm in cfg["evaluation"]["arms"]:
            path = per_patient_table_path(cfg, arm, model)
            if path.exists():
                by_arm[arm] = pd.read_csv(path)
        if not by_arm:
            continue
        for metric in FIGURE_METRICS:
            plot_per_patient_bars(by_arm, metric, fig_dir / f"per_patient_{metric}__{model}.png")
            n_figs += 1
    logger.info("wrote %d figure(s) to %s", n_figs, fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
