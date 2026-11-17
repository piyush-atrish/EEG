"""Event-level report from saved predictions (results/predictions/*.parquet).

No retraining needed: it reads the window predictions written by scripts/05_run_baseline.py
and writes seizure-level tables.

Usage (from the repo root):
    python tools/event_report.py
    python tools/event_report.py --min-consecutive 2 --refractory-s 60 --tag mc2_ref60

Outputs (in results/tables/): event_per_patient[_TAG].csv, event_seizures[_TAG].csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eegpipe.config import load_config
from eegpipe.evaluation.events import event_metrics_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--min-consecutive", type=int, default=1)
    parser.add_argument("--merge-gap-s", type=float, default=None)
    parser.add_argument("--refractory-s", type=float, default=0.0)
    parser.add_argument("--tag", type=str, default="", help="suffix for the output file names")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    files = sorted(Path(cfg["paths"]["predictions"]).glob("*.parquet"))
    if not files:
        raise SystemExit("No prediction files found. Run scripts/05_run_baseline.py first.")
    predictions = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    window_s = float(cfg["segmentation"]["window_s"])
    step_s = window_s * (1.0 - float(cfg["segmentation"]["overlap"]))
    summary, seizures = event_metrics_table(
        predictions,
        fs=float(cfg["dataset"]["fs"]),
        window_s=window_s,
        step_s=step_s,
        min_consecutive=args.min_consecutive,
        merge_gap_s=args.merge_gap_s,
        refractory_s=args.refractory_s,
    )

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(cfg["paths"]["tables"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"event_per_patient{suffix}.csv", index=False)
    seizures.to_csv(out_dir / f"event_seizures{suffix}.csv", index=False)

    overall = (
        summary.groupby(["arm", "model"])
        .agg(
            seizures=("n_seizures", "sum"),
            detected=("n_detected", "sum"),
            mean_patient_sensitivity=("event_sensitivity", "mean"),
            mean_fa_per_hour=("fa_per_hour_event", "mean"),
            median_latency_s=("median_latency_s", "median"),
        )
        .round(3)
    )
    print(overall.to_string())
    print(f"\nWrote tables to {out_dir} (suffix '{suffix}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
