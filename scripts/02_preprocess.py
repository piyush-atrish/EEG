"""Preprocess every included EDF into a filtered float32 .npy array (contract C3).

Exit code: 0 if every file succeeded (or was skipped as already valid), 1 otherwise.
Workers are capped (default 4, ``project.max_workers`` in the config, or ``--workers``) because
long recordings need several GB of RAM each.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from eegpipe.config import load_config
from eegpipe.preprocessing.pipeline import preprocess_all
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import file_index_csv
from eegpipe.utils.seed import set_global_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--patients", nargs="+", help="Subset of patients to process")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing .npy files")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])
    logger = get_logger("02_preprocess", log_file=Path(cfg["paths"]["logs"]) / "02_preprocess.log")

    index_path = file_index_csv(cfg)
    if not index_path.exists():
        logger.error("File index not found at %s. Run scripts/01_download_and_index.py first.",
                     index_path)
        return 1
    index = pd.read_csv(index_path)
    if args.patients:
        unknown = sorted(set(args.patients) - set(index["patient"]))
        if unknown:
            logger.error("Patient(s) not in file index: %s", unknown)
            return 1

    result = preprocess_all(
        index, cfg, patients=args.patients, n_jobs=args.workers, overwrite=args.overwrite
    )
    if result.empty:
        logger.error("No included files to process")
        return 1
    counts = result["status"].value_counts().to_dict()
    logger.info("Status counts: %s", counts)
    failed = counts.get("failed", 0)
    if failed:
        logger.error("Pipeline finished with %d failed file(s); see results/logs/preprocess_status.csv",
                     failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())