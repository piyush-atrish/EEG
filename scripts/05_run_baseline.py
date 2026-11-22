"""CLI: run the LOSO baseline for each arm x model combination (Step 12).

Usage
-----
    python scripts/05_run_baseline.py --config configs/config.yaml
    python scripts/05_run_baseline.py --arms raw --models svm --max-patients 3
    python scripts/05_run_baseline.py --tuning fixed  # fast fallback, see README Section 12.3
    python scripts/05_run_baseline.py --synthetic     # no real data needed (see caveat below)
    python scripts/05_run_baseline.py --patients chb01 chb02 chb03 chb04 chb05 --tag dev5
    python scripts/05_run_baseline.py --setting rls_L8_lam0.9999 --patients chb01 chb02 chb03 chb04 chb05 --tag dev5

With --setting the features come from data/processed/features_adaptive/<setting>/ and the arms are
named adaptive_<setting> and adaptive_<setting>_standardised. --tag is appended to every output arm
name (e.g. raw_dev5), so runs on different patient sets never overwrite each other. Compare arms
that were run on the same patients and the same tag only.
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from eegpipe.adaptive import adaptive_features_dir, list_settings
from eegpipe.config import load_config
from eegpipe.evaluation.leakage_checks import assert_finite_features, assert_patient_grouping
from eegpipe.evaluation.loso import run_loso
from eegpipe.evaluation.reporting import save_predictions, save_tables
from eegpipe.models.normalisation import apply_arm
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import features_path, file_index_csv

logger = get_logger(__name__)


def arm_label(base_arm: str, setting: str | None, tag: str | None) -> str:
    """Name under which an arm's results are saved (``__`` is reserved as the arm/model separator)."""
    if setting is None:
        label = base_arm
    elif base_arm == "raw":
        label = f"adaptive_{setting}"
    elif base_arm == "subject_standardised":
        label = f"adaptive_{setting}_standardised"
    else:
        label = f"adaptive_{setting}_{base_arm}"
    if tag:
        if not re.fullmatch(r"[A-Za-z0-9-]+", tag):
            raise ValueError(f"--tag may only contain letters, digits and '-', got {tag!r}")
        label = f"{label}_{tag}"
    if "__" in label:
        raise ValueError(f"arm name {label!r} contains '__', which separates arm and model in file names")
    return label


def _load_real_features(
    cfg: dict, patients: list[str] | None, max_patients: int | None, setting: str | None = None
) -> pd.DataFrame:
    """Concatenate every included case's Contract C5 parquet (written by scripts/04).

    Only cases of the requested ``patients`` are read. With ``setting`` the tables come from that
    adaptive setting's feature folder.
    """
    file_index = pd.read_csv(file_index_csv(cfg))
    included = file_index.loc[file_index["include"].astype(bool)]
    if patients is not None:
        included = included.loc[included["patient"].isin(patients)]
    cases = sorted(included["case"].unique())
    if not cases:
        raise RuntimeError("No included cases in file_index.csv; nothing to run the baseline on.")

    frames = []
    for case in cases:
        path = features_path(cfg, case) if setting is None else adaptive_features_dir(cfg, setting) / f"{case}.parquet"
        if not path.exists():
            hint = "scripts/04_extract_features.py" + (f" --setting {setting}" if setting else "")
            raise FileNotFoundError(
                f"{path} not found: run {hint} before scripts/05_run_baseline.py (or pass --synthetic)."
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
    parser.add_argument("--setting", type=str, default=None,
                        help="adaptive setting folder name: read its features, name arms adaptive_<setting>[_standardised]")
    parser.add_argument("--tag", type=str, default=None,
                        help="suffix for every output arm name, e.g. dev5 (keeps runs on different patient sets apart)")
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
    if args.setting is not None and not (adaptive_features_dir(cfg, args.setting)).is_dir():
        logger.error("No features for adaptive setting '%s'. Available: %s", args.setting,
                     ", ".join(p for p in list_settings(cfg)) or "(none; run 02b then 04 --setting)")
        return 1
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
        df = _load_real_features(cfg, args.patients, args.max_patients, args.setting)

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    assert_patient_grouping(df, cfg["dataset"]["patient_map"])
    assert_finite_features(df, feature_cols)

    logger.info("Baseline run: %d patient(s), arms=%s, models=%s", df["patient"].nunique(), arms, models)
    logger.info("Tuning mode: %s (patients: %s)", cfg["evaluation"]["tuning"]["mode"],
                ", ".join(sorted(df["patient"].unique())))

    for arm in arms:
        label = arm_label(arm, args.setting, args.tag)
        arm_df = apply_arm(df, arm, cfg)
        for model in models:
            preds, per_patient = run_loso(arm_df, feature_cols, model, cfg, arm=label)
            save_predictions(preds, label, model, cfg)
            save_tables(per_patient, label, model, cfg)
            logger.info("%s/%s: mean AUC=%.3f over %d patient(s)",
                        label, model, per_patient["auc"].mean(), len(per_patient))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
