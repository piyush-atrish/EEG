"""Standalone diagnostic: validate a real data delivery against Contracts C1-C3
before running scripts/03_make_windows.py or scripts/04_extract_features.py.

This does NOT modify anything and does NOT need the raw data to leave your
machine -- it only reads and prints a small, pasteable report. Run it
locally against the full dataset once you've extracted A's delivery into
data/interim/ (see the README's frozen `paths` in configs/config.yaml).

Usage
-----
    python check_data_contract.py --config configs/config.yaml
    python check_data_contract.py --config configs/config.yaml --sample-n 6
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from eegpipe.config import load_config
from eegpipe.segmentation.windows import file_stem
from eegpipe.utils.paths import annotations_csv, file_index_csv, preprocessed_path

REQUIRED_FILE_INDEX_COLS = ["patient", "case", "file", "include"]
REQUIRED_ANNOTATION_COLS = ["patient", "case", "file", "seizure_start_s", "seizure_end_s"]


def check(cfg_path: str, sample_n: int) -> int:
    issues: list[str] = []
    notes: list[str] = []

    cfg = load_config(cfg_path)
    n_channels_expected = len(cfg["dataset"]["channels"])
    patient_map = cfg["dataset"].get("patient_map", {})

    fi_path = file_index_csv(cfg)
    ann_path = annotations_csv(cfg)

    print(f"Looking for file_index.csv  at: {fi_path}")
    print(f"Looking for annotations.csv at: {ann_path}")

    if not fi_path.exists():
        print(f"\nMISSING: {fi_path} -- stopping here.")
        return 1
    if not ann_path.exists():
        issues.append(f"MISSING: {ann_path}")

    file_index = pd.read_csv(fi_path)
    annotations = pd.read_csv(ann_path) if ann_path.exists() else pd.DataFrame()

    print(f"\nfile_index.csv: {len(file_index)} rows, columns = {list(file_index.columns)}")
    for col in REQUIRED_FILE_INDEX_COLS:
        if col not in file_index.columns:
            issues.append(f"file_index.csv missing required column: {col!r}")

    if "include" in file_index.columns:
        notes.append(
            f"file_index['include'] dtype = {file_index['include'].dtype} "
            f"(our code does .astype(bool) defensively regardless, so this is informational, not a blocker)"
        )

    print(f"\nannotations.csv: {len(annotations)} rows, columns = {list(annotations.columns)}")
    for col in REQUIRED_ANNOTATION_COLS:
        if col not in annotations.columns:
            issues.append(f"annotations.csv missing required column: {col!r}")

    # --- patient/case pairs and the chb21 -> chb01 patient_map check ---
    if {"patient", "case"}.issubset(file_index.columns):
        pairs = file_index[["patient", "case"]].drop_duplicates()
        print(f"\nPatient/case pairs found:\n{pairs.to_string(index=False)}")
        for case, patient in zip(pairs["case"], pairs["patient"]):
            mapped = patient_map.get(case)
            if mapped is not None and mapped != patient:
                issues.append(
                    f"patient_map says case {case!r} -> patient {mapped!r}, but file_index.csv "
                    f"has patient={patient!r} for that case (LOSO grouping would be WRONG)"
                )
            elif case in patient_map:
                notes.append(f"case {case!r} correctly grouped under patient {patient!r} per patient_map")

    # --- resolve a sample of included files through preprocessed_path() ---
    included = file_index.loc[file_index["include"].astype(bool)] if "include" in file_index.columns else file_index

    print(f"\nSpot-checking preprocessed .npy resolution for up to {sample_n} included files...")
    checked = 0
    for _, row in included.iterrows():
        if checked >= sample_n:
            break
        case, file = row.get("case"), row.get("file")
        if case is None or file is None:
            continue
        stem = file_stem(str(file))
        npy_path = preprocessed_path(cfg, case, stem)
        exists = npy_path.exists()
        print(f"  case={case} file={file} -> {npy_path}  {'FOUND' if exists else 'MISSING'}")
        if not exists:
            issues.append(f"preprocessed_path() for case={case}, file={file} does not resolve: {npy_path}")
        else:
            arr = np.load(npy_path, mmap_mode="r")
            print(f"      shape={arr.shape} dtype={arr.dtype}")
            if arr.shape[0] != n_channels_expected:
                issues.append(f"{npy_path}: first dim (channels) = {arr.shape[0]}, expected {n_channels_expected}")
        checked += 1

    print("\n" + "=" * 60)
    if notes:
        print("NOTES:")
        for note in notes:
            print(f"  - {note}")
    if issues:
        print(f"\n{len(issues)} ISSUE(S) FOUND:")
        for issue in issues:
            print(f"  ! {issue}")
        return 1

    print("\nNo issues found against Contracts C1-C3. Safe to run scripts/03_make_windows.py.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a real data delivery against Contracts C1-C3.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sample-n", type=int, default=3, help="Number of files to spot-check for .npy resolution.")
    args = parser.parse_args()
    return check(args.config, args.sample_n)


if __name__ == "__main__":
    sys.exit(main())