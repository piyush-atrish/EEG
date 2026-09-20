"""CLI: extract Contract C5 features for the cohort, one process per case (Member B, Step 9).

Usage
-----
    python scripts/04_extract_features.py --config configs/config.yaml
    python scripts/04_extract_features.py --patients chb01 chb02 --n-jobs 4
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from joblib import Parallel, delayed

from eegpipe.config import load_config
from eegpipe.features.extract import extract_case_features
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import features_path, file_index_csv, windows_path

logger = get_logger(__name__)


def _process_one_case(case: str, file_index: pd.DataFrame, cfg: dict) -> tuple[str, float, int]:
    """Extract and write one case's Contract C5 parquet. Runs in its own process."""
    window_table = pd.read_parquet(windows_path(cfg, case))

    t0 = time.perf_counter()
    features_df = extract_case_features(case, file_index, window_table, cfg)
    elapsed = time.perf_counter() - t0

    out_path = features_path(cfg, case)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(out_path, index=False, engine="pyarrow")

    return case, elapsed, len(features_df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract Contract C5 features for the cohort.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--n-jobs", type=int, default=None, help="Overrides project.n_jobs from config.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    n_jobs = args.n_jobs if args.n_jobs is not None else cfg["project"].get("n_jobs", -1)

    file_index = pd.read_csv(file_index_csv(cfg))
    included = file_index.loc[file_index["include"].astype(bool)]
    if args.patients is not None:
        included = included.loc[included["patient"].isin(args.patients)]

    cases = sorted(included["case"].unique())
    if not cases:
        logger.warning("No cases to process (empty cohort or --patients filter matched nothing).")
        return 0

    logger.info("Extracting features for %d case(s) with n_jobs=%s", len(cases), n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(_process_one_case)(case, file_index, cfg) for case in cases)

    for case, elapsed, n_rows in results:
        logger.info("case=%s: %d windows, %.2fs", case, n_rows, elapsed)

    total_case_time = sum(elapsed for _, elapsed, _ in results)
    total_windows = sum(n_rows for _, _, n_rows in results)
    logger.info(
        "Done: %d cases, %d windows, %.2fs summed per-case time (wall time depends on n_jobs)",
        len(results), total_windows, total_case_time,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Extract features (contract C5) (owner: Member B, Step 9). Stub created by Member A at kickoff."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("Owner implements this script (accepts --config and --patients).")


if __name__ == "__main__":
    sys.exit(main())

