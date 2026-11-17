"""Do seizures move the features in the same direction in every patient?

Computes, for each patient, the within-patient AUC of every feature (ictal vs interictal) and
summarises separability and cross-patient direction agreement. A patient with low agreement is a
likely failure case for any cross-subject classifier.

Usage (from the repo root):
    python tools/feature_direction.py
    python tools/feature_direction.py --focus chb06 --top 15

Only cases whose feature file exists are used. Writes results/tables/feature_direction.csv
(features x patients) and results/tables/feature_direction_summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eegpipe.config import load_config
from eegpipe.evaluation.feature_direction import direction_summary, feature_auc_by_patient
from eegpipe.utils.paths import features_path, file_index_csv


def load_available_features(cfg: dict) -> pd.DataFrame:
    index = pd.read_csv(file_index_csv(cfg))
    cases = sorted(index.loc[index["include"].astype(bool), "case"].unique())
    found = [c for c in cases if features_path(cfg, c).exists()]
    if not found:
        raise SystemExit("No feature files found. Run scripts/04_extract_features.py first.")
    return pd.concat([pd.read_parquet(features_path(cfg, c)) for c in found], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--focus", type=str, default=None, help="patient to examine in detail")
    parser.add_argument("--top", type=int, default=12, help="features listed for --focus")
    parser.add_argument("--min-ictal", type=int, default=30)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df = load_available_features(cfg)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    auc = feature_auc_by_patient(df, feature_cols, min_ictal=args.min_ictal)
    summary = direction_summary(auc)

    out_dir = Path(cfg["paths"]["tables"])
    out_dir.mkdir(parents=True, exist_ok=True)
    auc.to_csv(out_dir / "feature_direction.csv")
    summary.to_csv(out_dir / "feature_direction_summary.csv")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary.to_string())

    if args.focus:
        if args.focus not in auc.columns:
            raise SystemExit(f"{args.focus} has too few ictal windows or is not in the data")
        consensus = auc.drop(columns=args.focus).median(axis=1)
        detail = pd.DataFrame({"auc_focus": auc[args.focus], "auc_others_median": consensus})
        detail["flipped"] = (detail["auc_focus"] - 0.5) * (detail["auc_others_median"] - 0.5) < 0
        strongest = detail.reindex((detail["auc_others_median"] - 0.5).abs().sort_values(ascending=False).index)
        print(f"\nStrongest consensus features, and what {args.focus} does with them:")
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(strongest.head(args.top).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
