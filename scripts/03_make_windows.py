"""CLI: build Contract C4 window tables for the cohort (Member B, Step 6).

Usage
-----
    python scripts/03_make_windows.py --config configs/config.yaml
    python scripts/03_make_windows.py --patients chb01 chb02
"""
from __future__ import annotations

import argparse

import pandas as pd

from eegpipe.config import load_config
from eegpipe.segmentation.windows import build_window_table
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import annotations_csv, file_index_csv, windows_path

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build labelled windows (Contract C4) from preprocessed EEG."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="config (default: <repo>/configs/config.yaml)"
    )
    parser.add_argument("--patients", nargs="*", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    file_index = pd.read_csv(file_index_csv(cfg))
    annotations = pd.read_csv(annotations_csv(cfg))

    table = build_window_table(file_index, annotations, cfg, patients=args.patients)

    if table.empty:
        logger.warning("No windows produced (empty cohort or filter). Nothing written.")
        return 0

    for case, case_df in table.groupby("case"):
        out_path = windows_path(cfg, case)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        case_df.to_parquet(out_path, index=False)
        logger.info("Wrote %d windows to %s", len(case_df), out_path)

    total = len(table)
    n_ictal = int((table["label"] == 1).sum())
    logger.info(
        "Total: %d windows, %d cases, %d patients, ictal fraction %.4f",
        total,
        table["case"].nunique(),
        table["patient"].nunique(),
        n_ictal / total,
    )

    per_patient = table.groupby("patient")["label"].agg(n_windows="size", n_ictal="sum")
    per_patient["ictal_fraction"] = per_patient["n_ictal"] / per_patient["n_windows"]
    print(per_patient.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
