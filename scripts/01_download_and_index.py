import argparse
import sys
from pathlib import Path
from eegpipe.config import load_config
from eegpipe.io.download import download_metadata
from eegpipe.io.annotations import build_annotations
from eegpipe.utils.paths import annotations_csv
from eegpipe.utils.logging_utils import get_logger

logger = get_logger("01_download")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Download CHB-MIT metadata and build annotations")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config.yaml")
    parser.add_argument("--patients", nargs="+", help="Subset of patients (not used until Step 3 EDF download)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    
    # 1. Download metadata (RECORDS and summary text files)
    logger.info("Starting metadata download phase...")
    download_metadata(cfg)
    
    # 2. Build annotations.csv
    logger.info("Parsing summary files to build annotations.csv...")
    raw_dir = Path(cfg["paths"]["raw"])
    patient_map = cfg["dataset"]["patient_map"]
    
    df_annotations = build_annotations(raw_dir, patient_map)
    
    out_path = annotations_csv(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_annotations.to_csv(out_path, index=False)
    
    logger.info(f"Saved annotations for {len(df_annotations)} seizures to {out_path}")
    logger.info("Script 01 stopping at annotation parsing (Step 3 will append the cohort selection and EDF download).")

if __name__ == "__main__":
    main()