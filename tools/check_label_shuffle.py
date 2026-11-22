"""L8 label-shuffle sanity check on REAL features.

Trains on shuffled labels and scores on genuinely labelled held-out patients. A result near
AUC 0.5 means no label leakage; a result far from 0.5 means every other result is suspect.

Usage (from the repo root):
    python tools/check_label_shuffle.py                      # svm and rf, 3 held-out patients
    python tools/check_label_shuffle.py --setting kalman_qr3e-05_blk1   # features of an adaptive setting
    python tools/check_label_shuffle.py --models svm --n-test 2
    python tools/check_label_shuffle.py --seeds 0 1 2        # repeat with different patient draws

Only cases whose feature file exists on disk are used, so it works while the full cohort is
still being processed. The result is appended to results/tables/label_shuffle.csv.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from eegpipe.adaptive import adaptive_features_dir
from eegpipe.config import load_config
from eegpipe.evaluation.leakage_checks import label_shuffle_check
from eegpipe.utils.paths import features_path, file_index_csv

LOW, HIGH = 0.35, 0.65  # the same tolerance band as the unit test


def load_available_features(cfg: dict, setting: str | None = None) -> pd.DataFrame:
    """Concatenate the feature tables of included cases that exist on disk (of an adaptive setting if given)."""
    def path_of(case: str) -> Path:
        return features_path(cfg, case) if setting is None else adaptive_features_dir(cfg, setting) / f"{case}.parquet"

    index = pd.read_csv(file_index_csv(cfg))
    cases = sorted(index.loc[index["include"].astype(bool), "case"].unique())
    found = [c for c in cases if path_of(c).exists()]
    missing = [c for c in cases if c not in found]
    if missing:
        print(f"Skipping cases without a feature file: {', '.join(missing)}")
    if not found:
        raise SystemExit("No feature files found. Run scripts/04_extract_features.py first.")
    return pd.concat([pd.read_parquet(path_of(c)) for c in found], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--models", nargs="+", choices=["svm", "rf"], default=["svm", "rf"])
    parser.add_argument("--n-test", type=int, default=3, help="held-out patients per check")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--setting", type=str, default=None,
                        help="use the features of this adaptive setting (data/processed/features_adaptive/<setting>)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df = load_available_features(cfg, args.setting)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    n_patients = df["patient"].nunique()
    n_test = min(args.n_test, n_patients)
    tuning = cfg["evaluation"]["tuning"]["mode"]
    print(f"Patients available: {n_patients}; features: {len(feature_cols)}; tuning: {tuning}"
          + (f"; setting: {args.setting}" if args.setting else ""))

    rows, worst_ok = [], True
    for model in args.models:
        for seed in args.seeds:
            auc = label_shuffle_check(df, feature_cols, model, cfg, n_test_patients=n_test, seed=seed)
            ok = LOW <= auc <= HIGH
            worst_ok &= ok
            print(f"  {model:>3}  seed {seed}: AUC = {auc:.3f}  {'OK' if ok else 'WARNING'}")
            rows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model": model,
                    "seed": seed,
                    "n_test_patients": n_test,
                    "n_patients_available": n_patients,
                    "tuning_mode": tuning,
                    "setting": args.setting or "",
                    "auc": round(auc, 4),
                    "within_band": ok,
                }
            )

    out = Path(cfg["paths"]["tables"]) / "label_shuffle.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    # merge with the existing log instead of appending lines: older logs have fewer columns
    combined = pd.concat([pd.read_csv(out), new], ignore_index=True) if out.exists() else new
    combined.to_csv(out, index=False)
    print(f"Appended results to {out}")
    if worst_ok:
        print("OK: near chance, as expected when training labels are shuffled.")
        return 0
    print("WARNING: far from 0.5 -- investigate for a leak before trusting any real result.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
