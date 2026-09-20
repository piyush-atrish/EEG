"""Download CHB-MIT metadata and cohort EDFs, then build annotations.csv (C1) and file_index.csv (C2).

Flow (repeated until the cohort is stable, at most ``--max-iter`` rounds):

1. select a cohort from the current index (:func:`eegpipe.io.cohort.select_cohort`);
2. download the selected EDFs that are missing;
3. header-check every selected file (readable, 256 Hz, all 18 channels) and take its real
   duration and sample count;
4. files that fail (download, unreadable, missing channel...) are excluded with a reason and
   the next round back-fills the cap with usable files.

Running with ``--patients`` re-generates only those patients' rows and keeps the others that
are already in ``file_index.csv``.

Exit code: 0 on success, 1 on a hard problem (no metadata, unknown patient, a patient left
without a usable seizure file, or a cohort that did not stabilise).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from eegpipe.config import load_config
from eegpipe.io.annotations import build_annotations, build_summary_durations
from eegpipe.io.cohort import (
    INDEX_COLUMNS,
    build_file_index,
    cohort_summary,
    select_cohort,
    validate_cohort,
)
from eegpipe.io.download import download_edfs, download_metadata
from eegpipe.io.loader import check_edf
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import annotations_csv, ensure_parent, file_index_csv
from eegpipe.utils.seed import set_global_seed


def _merge_with_existing(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Keep other patients' rows from an existing index; replace this run's patients."""
    if not path.exists():
        return new
    try:
        old = pd.read_csv(path)
    except Exception:
        return new
    old = old[~old["patient"].isin(new["patient"].unique())]
    merged = pd.concat([old, new], ignore_index=True)
    return merged.sort_values(["patient", "file_order"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--patients", nargs="+", help="Only (re)build these patients")
    parser.add_argument("--skip-download", action="store_true", help="Use files already on disk")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads")
    parser.add_argument("--max-iter", type=int, default=5)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])
    logger = get_logger("01_download", log_file=Path(cfg["paths"]["logs"]) / "01_download.log")
    raw_dir = Path(cfg["paths"]["raw"])
    aliases = cfg["dataset"].get("channel_aliases", {})
    channels = cfg["dataset"]["channels"]
    fs = int(cfg["dataset"]["fs"])

    # 1. metadata ---------------------------------------------------------------------
    if not args.skip_download:
        failed_meta = download_metadata(cfg, args.workers)
        if "RECORDS" in failed_meta:
            logger.error("RECORDS could not be downloaded; cannot continue")
            return 1
    if not (raw_dir / "RECORDS").exists():
        logger.error("RECORDS not found in %s", raw_dir)
        return 1

    # 2. annotations and index -----------------------------------------------------------
    annotations = build_annotations(raw_dir, cfg["dataset"].get("patient_map", {}))
    ensure_parent(annotations_csv(cfg))
    annotations.to_csv(annotations_csv(cfg), index=False)
    logger.info("annotations.csv: %d seizures in %d files", len(annotations),
                annotations["file"].nunique())

    index = build_file_index(cfg, annotations, build_summary_durations(raw_dir))
    if args.patients:
        unknown = sorted(set(args.patients) - set(index["patient"]))
        if unknown:
            logger.error("Unknown patient(s): %s", unknown)
            return 1
        index = index[index["patient"].isin(args.patients)].reset_index(drop=True)

    # 3. select / download / check until stable -------------------------------------------
    unusable: dict[str, str] = {}
    checked: dict[str, dict] = {}
    previous: set[str] | None = None
    cohort = index
    for round_no in range(1, args.max_iter + 1):
        working = index.copy()
        for name, header in checked.items():
            mask = working["file"] == name
            working.loc[mask, "duration_s"] = header["duration_s"]
            working.loc[mask, "n_samples"] = header["n_samples"]
        cohort = select_cohort(working, cfg, unusable)
        selected = cohort[cohort["include"]]

        missing = [
            f"{r.case}/{r.file}" for r in selected.itertuples() if not (raw_dir / r.case / r.file).exists()
        ]
        if missing and not args.skip_download:
            result = download_edfs(cfg, missing, args.workers)
            for rel in result["failed"]:
                unusable[Path(rel).name] = "download_failed"

        changed = False
        for row in selected.itertuples():
            if row.file in unusable or row.file in checked:
                continue
            edf = raw_dir / row.case / row.file
            if not edf.exists():
                unusable[row.file] = "download_failed"
                changed = True
                continue
            ok, reason, header = check_edf(edf, channels, aliases, fs)
            if ok:
                checked[row.file] = header
                changed = True  # real duration may change the cap arithmetic
            else:
                unusable[row.file] = reason
                logger.warning("Excluding %s: %s", row.file, reason)
                changed = True
        current = set(selected["file"])
        logger.info("Round %d: %d files selected, %d unusable so far", round_no, len(current),
                    len(unusable))
        if not changed and current == previous:
            break
        previous = current
    else:
        logger.error("Cohort did not stabilise after %d rounds", args.max_iter)
        return 1

    # 4. final validation, save ---------------------------------------------------------------
    cohort, warnings = validate_cohort(cohort)
    for message in warnings:
        logger.warning(message)
    final = cohort[INDEX_COLUMNS]
    out = ensure_parent(file_index_csv(cfg))
    _merge_with_existing(final, out).to_csv(out, index=False)

    summary = cohort_summary(cohort, cfg)
    logger.info("Cohort per patient:\n%s", summary.to_string(index=False, float_format="%.2f"))
    logger.info(
        "Saved file_index.csv: %d files, %.2f h, ~%.2f GB of preprocessed cache",
        len(cohort[cohort["include"]]),
        summary["total_hours"].sum() if len(summary) else 0.0,
        summary["cache_gb"].sum() if len(summary) else 0.0,
    )
    reasons = cohort.loc[~cohort["include"], "exclude_reason"].value_counts()
    if len(reasons):
        logger.info("Exclusions:\n%s", reasons.to_string())

    no_seizure = [w for w in warnings if "no usable seizure file" in w]
    if no_seizure:
        return 1
    if args.patients and not cohort["include"].any():
        logger.error("No files included for %s", args.patients)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
