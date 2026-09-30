"""CLI: run the LOSO baseline for each arm x model combination (Member C, Step 12).

Usage
-----
    python scripts/05_run_baseline.py --config configs/config.yaml
    python scripts/05_run_baseline.py --arms raw --models svm --max-patients 3
    python scripts/05_run_baseline.py --tuning fixed  # fast fallback, see README Section 12.3
    python scripts/05_run_baseline.py --synthetic     # no real data needed (see caveat below)
"""
from __future__ import annotations

import argparse

import pandas as pd

from eegpipe.config import load_config
from eegpipe.evaluation.leakage_checks import assert_finite_features, assert_patient_grouping
from eegpipe.evaluation.loso import run_loso
from eegpipe.evaluation.reporting import save_predictions, save_tables
from eegpipe.models.normalisation import apply_arm
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import features_path, file_index_csv

logger = get_logger(__name__)


def _load_real_features(cfg: dict, patients: list[str] | None, max_patients: int | None) -> pd.DataFrame:
    """Concatenate every included case's Contract C5 parquet (written by scripts/04)."""
    file_index = pd.read_csv(file_index_csv(cfg))
    included = file_index.loc[file_index["include"].astype(bool)]
    cases = sorted(included["case"].unique())
    if not cases:
        raise RuntimeError("No included cases in file_index.csv; nothing to run the baseline on.")

    frames = []
    for case in cases:
        path = features_path(cfg, case)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found: run scripts/04_extract_features.py before "
                "scripts/05_run_baseline.py (or pass --synthetic)."
            )
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    return _filter_patients(df, patients, max_patients)


def _filter_patients(df: pd.DataFrame, patients: list[str] | None, max_patients: int | None) -> pd.DataFrame:
    if patients is not None:
        df = df[df["patient"].isin(patients)]
    if max_patients is not None:
        keep = sorted(df["patient"].unique())[:max_patients]
        df = df[df["patient"].isin(keep)]
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LOSO baseline for each arm x model.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--arms", nargs="*", default=None, help="default: evaluation.arms")
    parser.add_argument("--models", nargs="*", default=None, help="default: evaluation.models")
    parser.add_argument("--patients", nargs="*", default=None, help="restrict to these patients")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--tuning", choices=["nested", "fixed"], default=None,
                         help="overrides evaluation.tuning.mode (see README Section 12.3)")
    parser.add_argument("--synthetic", action="store_true",
                         help="use tests.fixtures.synth_features instead of parquet files. Only "
                              "works with the repo root on PYTHONPATH (as pytest sets it via "
                              "pyproject's `pythonpath`) -- not a plain `python scripts/...` "
                              "invocation from elsewhere. The joint integration test does not "
                              "use this flag; it always runs on real (synthetic-project) data.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.tuning is not None:
        cfg = {**cfg, "evaluation": {**cfg["evaluation"],
                                      "tuning": {**cfg["evaluation"]["tuning"], "mode": args.tuning}}}

    arms = args.arms if args.arms is not None else cfg["evaluation"]["arms"]
    models = args.models if args.models is not None else cfg["evaluation"]["models"]

    if args.synthetic:
        from tests.fixtures.synth_features import make_synthetic_features
        df = make_synthetic_features(cfg, seed=cfg.get("project", {}).get("seed", 0))
        df = _filter_patients(df, args.patients, args.max_patients)
    else:
        df = _load_real_features(cfg, args.patients, args.max_patients)

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    assert_patient_grouping(df, cfg["dataset"]["patient_map"])
    assert_finite_features(df, feature_cols)

    logger.info("Baseline run: %d patient(s), arms=%s, models=%s", df["patient"].nunique(), arms, models)

    for arm in arms:
        arm_df = apply_arm(df, arm, cfg)
        for model in models:
            preds, per_patient = run_loso(arm_df, feature_cols, model, cfg, arm=arm)
            save_predictions(preds, arm, model, cfg)
            save_tables(per_patient, arm, model, cfg)
            logger.info("%s/%s: mean AUC=%.3f over %d patient(s)",
                        arm, model, per_patient["auc"].mean(), len(per_patient))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())