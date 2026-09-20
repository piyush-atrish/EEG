"""Standalone diagnostic: summarize real Contract C5 features by label
(ictal vs interictal), averaged across channels, to check
docs/feature_catalog.md's "expected during seizure" column against real
data instead of literature guesses.

This does NOT modify anything -- it only reads the already-extracted
data/processed/features/{case}.parquet files and prints a summary table.

Usage
-----
    python check_feature_sanity.py --config configs/config.yaml
    python check_feature_sanity.py --config configs/config.yaml --cases chb01 chb02
"""
from __future__ import annotations

import argparse

import pandas as pd

from eegpipe.config import load_config
from eegpipe.utils.paths import features_path


def summarize(cfg_path: str, cases: list[str] | None) -> int:
    cfg = load_config(cfg_path)
    catalog = cfg["features"]["catalog"]
    entropy_enabled = cfg["features"]["entropy"].get("enabled", True)

    flat_feature_names: list[str] = []
    for group, feats in catalog.items():
        if group == "nonlinear" and not entropy_enabled:
            continue
        flat_feature_names.extend(feats)

    features_dir = features_path(cfg, "_probe_").parent
    if cases is None:
        if not features_dir.exists():
            print(f"No features directory found at {features_dir}")
            return 1
        cases = sorted(p.stem for p in features_dir.glob("*.parquet"))

    print(f"Features directory: {features_dir}")
    print(f"Cases: {cases}\n")

    frames = []
    for case in cases:
        path = features_path(cfg, case)
        if not path.exists():
            print(f"  (skipping {case}: {path} not found)")
            continue
        frames.append(pd.read_parquet(path))

    if not frames:
        print("No feature parquet files found.")
        return 1

    all_df = pd.concat(frames, ignore_index=True)
    n_ictal = int((all_df["label"] == 1).sum())
    n_interictal = int((all_df["label"] == 0).sum())
    print(f"Total windows: {len(all_df)}  (ictal={n_ictal}, interictal={n_interictal})\n")

    if n_ictal == 0:
        print("WARNING: zero ictal windows across the requested cases -- nothing to compare.")
        return 1

    rows = []
    for feat in flat_feature_names:
        cols = [c for c in all_df.columns if c.startswith(f"f_{feat}_")]
        if not cols:
            continue
        # Mean across all 18 channels, per window, then grouped by label.
        # This is a first-pass, channel-averaged sanity check, not a
        # replacement for per-channel or per-patient analysis.
        per_window_mean = all_df[cols].mean(axis=1)
        interictal_mean = per_window_mean[all_df["label"] == 0].mean()
        ictal_mean = per_window_mean[all_df["label"] == 1].mean()
        pct_change = (ictal_mean - interictal_mean) / abs(interictal_mean) * 100 if interictal_mean != 0 else float("nan")
        rows.append(
            {
                "feature": feat,
                "interictal_mean": round(interictal_mean, 4),
                "ictal_mean": round(ictal_mean, 4),
                "pct_change": round(pct_change, 1),
                "direction": "UP" if ictal_mean > interictal_mean else "DOWN",
            }
        )

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_rows", None)
    print(summary.to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize real Contract C5 features by ictal/interictal label.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()
    return summarize(args.config, args.cases)


if __name__ == "__main__":
    raise SystemExit(main())
