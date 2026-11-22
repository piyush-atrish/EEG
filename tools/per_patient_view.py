"""Patients x arms table of one metric from the saved per-patient results.

Usage (from the repo root):
    python tools/per_patient_view.py --metric auc --arms raw_dev5 subject_standardised_dev5
    python tools/per_patient_view.py --metric sensitivity          # every arm found, both models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eegpipe.config import load_config
from eegpipe.evaluation.compare import load_per_patient_tables, per_patient_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--metric", type=str, default="auc",
                        choices=["auc", "sensitivity", "specificity", "f1", "fa_per_hour"])
    parser.add_argument("--arms", nargs="+", default=None, help="arms to show, in this order")
    args = parser.parse_args(argv)

    tables = load_per_patient_tables(Path(load_config(args.config)["paths"]["tables"]))
    if not tables:
        raise SystemExit("No per_patient_<arm>__<model>.csv files found.")
    for model in sorted({m for _, m in tables}):
        try:
            matrix = per_patient_matrix(tables, model, args.metric, args.arms)
        except ValueError:
            continue
        with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
            print(f"\n== {args.metric} ({model}) ==")
            print(matrix.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
