"""Paired arm comparison from saved per-patient tables (no retraining).

Usage (from the repo root):
    python tools/compare_arms.py                       # reference arm: raw
    python tools/compare_arms.py --reference subject_standardised

For every model, each other arm is compared with the reference arm over the patients in
results/tables/per_patient_<arm>__<model>.csv. Reported for all patients, for the development
patients (default chb01-chb05) and for the evaluation patients. The difference is arm - reference.
Writes results/tables/arm_comparison.csv and prints the AUC rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eegpipe.config import load_config
from eegpipe.evaluation.compare import compare_arms, load_per_patient_tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--reference", type=str, default="raw")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="compare only these arms with the reference (default: every arm found); "
                             "use it to keep runs on different patient sets apart")
    parser.add_argument(
        "--dev", nargs="+", default=["chb01", "chb02", "chb03", "chb04", "chb05"],
        help="development patients (everything else is the evaluation set)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    table_dir = Path(cfg["paths"]["tables"])
    tables = load_per_patient_tables(table_dir)
    if not tables:
        raise SystemExit(f"No per_patient_<arm>__<model>.csv files in {table_dir}")

    if args.arms:
        keep = set(args.arms) | {args.reference}
        tables = {key: table for key, table in tables.items() if key[0] in keep}
    result = compare_arms(tables, args.reference, args.dev)
    out = table_dir / "arm_comparison.csv"
    result.to_csv(out, index=False)

    view = result[result["metric"] == "auc"][
        ["model", "arm", "subset", "n_patients", "mean_reference", "mean_arm", "mean_diff",
         "ci_low", "ci_high", "n_arm_greater", "n_reference_greater", "p_value"]
    ]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(view.to_string(index=False))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
