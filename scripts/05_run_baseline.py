"""Stage 05: run the LOSO baseline (arms x models) and save predictions and tables.

Examples
--------
    python scripts/05_run_baseline.py                                  # everything in configs/config.yaml
    python scripts/05_run_baseline.py --patients chb01 chb02 chb03      # subset of patients
    python scripts/05_run_baseline.py --synthetic --tuning fixed        # smoke test, no real data
    python scripts/05_run_baseline.py --label-shuffle                   # add the L8 sanity gate
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from eegpipe.config import load_config
from eegpipe.evaluation.leakage_checks import (
    assert_finite_features,
    assert_patient_grouping,
    label_shuffle_check,
)
from eegpipe.evaluation.loso import feature_columns, run_loso
from eegpipe.evaluation.reporting import make_summary_table, save_predictions, save_tables
from eegpipe.models.normalisation import apply_arm
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.seed import set_global_seed

logger = get_logger("baseline")


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/config.yaml", help="path to the YAML config")
    p.add_argument("--arms", nargs="+", choices=["raw", "subject_standardised"], default=None)
    p.add_argument("--models", nargs="+", choices=["svm", "rf"], default=None)
    p.add_argument("--patients", nargs="+", default=None,
                   help="restrict the whole dataset (training and held-out) to these patients")
    p.add_argument("--max-patients", type=int, default=None,
                   help="keep only the first N patients (sorted) of the dataset")
    p.add_argument("--tuning", choices=["nested", "fixed"], default=None,
                   help="override evaluation.tuning.mode")
    p.add_argument("--synthetic", action="store_true", help="use synthetic features instead of parquet files")
    p.add_argument("--label-shuffle", action="store_true",
                   help="also run the L8 label-shuffle sanity check (expect AUC near 0.5)")
    return p.parse_args(argv)


def _load_features(cfg: dict, synthetic: bool) -> pd.DataFrame:
    if synthetic:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `tests.fixtures` importable
        from tests.fixtures.synth_features import make_synthetic_features
        return make_synthetic_features(cfg)
    files = sorted(Path(cfg["paths"]["features"]).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no feature parquet files in {cfg['paths']['features']} (run stage 04 first)")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main(argv=None) -> int:
    args = _parse(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = {"evaluation": {"tuning": {"mode": args.tuning}}} if args.tuning else None
    cfg = load_config(args.config, overrides)
    set_global_seed(int(cfg["project"]["seed"]))
    arms = args.arms or list(cfg["evaluation"]["arms"])
    models = args.models or list(cfg["evaluation"]["models"])

    df = _load_features(cfg, args.synthetic)
    patient_map = cfg["dataset"]["patient_map"]
    assert_patient_grouping(df, patient_map)                       # L1
    if args.patients:
        wanted = {patient_map.get(p, p) for p in args.patients}     # chb21 -> chb01
        df = df[df["patient"].isin(wanted)]
    if args.max_patients:
        keep = sorted(df["patient"].unique())[: args.max_patients]
        df = df[df["patient"].isin(keep)]
    df = df.reset_index(drop=True)
    feat_cols = feature_columns(df)
    assert_finite_features(df, feat_cols)                          # contract C5
    n_pat = df["patient"].nunique()
    logger.info("loaded %d windows, %d patients, %d feature columns; tuning=%s",
                len(df), n_pat, len(feat_cols), cfg["evaluation"]["tuning"]["mode"])
    if n_pat < 2:
        raise SystemExit("LOSO needs at least two patients")

    prepared = {arm: apply_arm(df, arm, cfg) for arm in arms}       # same rows + same is_calibration mask
    for arm in arms:
        for model in models:
            logger.info("=== arm=%s model=%s ===", arm, model)
            pred, per_patient = run_loso(prepared[arm], feat_cols, model, cfg, arm)
            save_predictions(pred, arm, model, cfg)
            save_tables(per_patient, arm, model, cfg)

    summary = make_summary_table(cfg)
    print(summary.to_string(index=False))

    if args.label_shuffle:
        for model in models:
            auc = label_shuffle_check(prepared["raw"] if "raw" in prepared else prepared[arms[0]],
                                      feat_cols, model, cfg)
            verdict = "OK" if 0.35 <= auc <= 0.65 else "SUSPICIOUS: investigate leakage"
            logger.info("L8 label-shuffle %s: mean AUC %.3f (%s)", model, auc, verdict)
            print(f"label-shuffle AUC ({model}): {auc:.3f}  [{verdict}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
