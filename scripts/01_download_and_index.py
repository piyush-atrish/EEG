import argparse
import pandas as pd
from pathlib import Path

from eegpipe.config import load_config
from eegpipe.io.download import download_metadata, download_edfs
from eegpipe.io.annotations import build_annotations
from eegpipe.io.cohort import select_cohort
from eegpipe.io.loader import read_edf_header
from eegpipe.utils.paths import file_index_csv, annotations_csv
from eegpipe.utils.logging_utils import get_logger

logger = get_logger("01_download")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--patients", nargs="+", help="Subset of patients to process")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = Path(cfg["paths"]["raw"])

    # 1. Metadata
    logger.info("Downloading metadata...")
    download_metadata(cfg)

    # 2. Annotations & Index Building
    logger.info("Building annotations...")
    df_annotations = build_annotations(raw_dir, cfg["dataset"]["patient_map"])

    logger.info("Scanning RECORDS to build file index...")
    records_path = raw_dir / "RECORDS"
    with open(records_path, "r") as f:
        all_files = [line.strip() for line in f if line.strip()]

    rows = []
    for file_path in all_files:
        case = file_path.split("/")[0]
        patient = cfg["dataset"]["patient_map"].get(case, case)
        file_name = file_path.split("/")[1]

        seizures = df_annotations[df_annotations["file"] == file_name]
        n_seizures = len(seizures)

        rows.append({
            "patient": patient,
            "case": case,
            "file": file_name,
            "duration_s": 3600.0,
            "n_samples": 3600 * cfg["dataset"]["fs"],
            "n_seizures": n_seizures,
            "role": "seizure" if n_seizures > 0 else "seizure_free",
            "include": False,
            "exclude_reason": ""
        })

    df_index = pd.DataFrame(rows)
    df_index["file_order"] = df_index.groupby("patient").cumcount()

    # 3. Apply cohort selection
    logger.info("Filtering index and applying data caps...")
    df_cohort = select_cohort(df_index, cfg, patients=args.patients)

    files_to_download = [f"{row['case']}/{row['file']}" for _, row in df_cohort[df_cohort["include"]].iterrows()]
    logger.info(f"Cohort selection complete. {len(files_to_download)} files marked for download.")

    # 4. Download EDFs
    failed_downloads = download_edfs(cfg, files_to_download)
    for failed_path in failed_downloads:
        file_name = Path(failed_path).name
        mask = df_cohort["file"] == file_name
        df_cohort.loc[mask, "include"] = False
        df_cohort.loc[mask, "exclude_reason"] = "download_failed"
        logger.error(f"File {file_name} excluded due to download failure.")

    # 5. Correct the durations using the fixed MNE math
    logger.info("Verifying channels and durations in downloaded files...")
    for idx, row in df_cohort[df_cohort["include"]].iterrows():
        edf_path = raw_dir / row["case"] / row["file"]
        if edf_path.exists():
            header = read_edf_header(edf_path)
            df_cohort.loc[idx, "duration_s"] = header["duration_s"]
            df_cohort.loc[idx, "n_samples"] = int(header["duration_s"] * cfg["dataset"]["fs"])

    # 6. Save
    out_ann = annotations_csv(cfg)
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    df_annotations.to_csv(out_ann, index=False)

    out_idx = file_index_csv(cfg)
    out_idx.parent.mkdir(parents=True, exist_ok=True)
    df_cohort.to_csv(out_idx, index=False)

    total_hours = df_cohort[df_cohort["include"]]["duration_s"].sum() / 3600.0
    logger.info(f"Saved file_index.csv. Final included data volume: {total_hours:.2f} hours.")

if __name__ == "__main__":
    main()