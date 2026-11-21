"""Apply one adaptive filter setting to every preprocessed recording (chunk 3.1).

Reads data/interim/preprocessed/<case>/<file>.npy and writes the filtered arrays to
data/interim/preprocessed_adaptive/<setting>/<case>/<file>.npy, one filter state stream per case
(files in chronological order, 60 s warm start). Resumable; cases run in parallel.

Examples:
    python scripts/02b_adaptive.py --patients chb01 chb02
    python scripts/02b_adaptive.py --method nlms --param length=8 --param mu=0.005
    python scripts/02b_adaptive.py --method kalman --param q_over_r=1e-4 --workers 2
    python scripts/02b_adaptive.py --method nlms --warm-start-s 600 --patients chb01

Exit code: 0 if no file failed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

from eegpipe.adaptive import METHODS
from eegpipe.adaptive.runner import resolve_run, run_adaptive
from eegpipe.config import load_config
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import file_index_csv
from eegpipe.utils.seed import set_global_seed


def parse_overrides(items: list[str] | None) -> dict:
    """``["mu=0.01", "length=8"]`` -> ``{"mu": 0.01, "length": 8}``."""
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--param expects key=value, got '{item}'")
        key, value = item.split("=", 1)
        try:
            overrides[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            overrides[key.strip()] = value.strip()
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--method", choices=METHODS, default=None, help="default: adaptive.method")
    parser.add_argument("--param", action="append", metavar="KEY=VALUE",
                        help="override a parameter of the method (repeatable)")
    parser.add_argument("--warm-start-s", type=float, default=None,
                        help="warm-start length in seconds (default: adaptive.warm_start_s); "
                             "a non-default value gets its own output folder")
    parser.add_argument("--patients", nargs="+", help="subset of patients to process")
    parser.add_argument("--workers", type=int, default=None, help="max parallel cases")
    parser.add_argument("--overwrite", action="store_true", help="recompute existing outputs")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])
    logger = get_logger("02b_adaptive", log_file=Path(cfg["paths"]["logs"]) / "02b_adaptive.log")

    try:
        overrides = parse_overrides(args.param)
        method, params = resolve_run(cfg, args.method, overrides)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Method %s with %s", method, params)

    index_path = file_index_csv(cfg)
    if not index_path.exists():
        logger.error("File index not found at %s. Run scripts/01_download_and_index.py first.", index_path)
        return 1
    index = pd.read_csv(index_path)
    if args.patients:
        unknown = sorted(set(args.patients) - set(index["patient"]))
        if unknown:
            logger.error("Patient(s) not in file index: %s", unknown)
            return 1

    result = run_adaptive(index, cfg, method=method, overrides=overrides, patients=args.patients,
                          n_jobs=args.workers, overwrite=args.overwrite,
                          warm_start_s=args.warm_start_s)
    if result.empty:
        logger.error("No included files to process")
        return 1
    counts = result["status"].value_counts().to_dict()
    logger.info("Status counts: %s", counts)
    if counts.get("failed", 0) or counts.get("not_run", 0):
        logger.error("Some files failed or were not run; see results/logs/adaptive_status.csv")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
