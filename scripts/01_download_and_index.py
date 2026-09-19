import argparse
import sys
import pandas as pd
from pathlib import Path
from eegpipe.config import load_config
from eegpipe.io.download import download_metadata, download_edfs
from eegpipe.io.annotations import build_annotations
from eegpipe.io.cohort import build_file_index, select_cohort
from eegpipe.io.loader import read_edf_header
from eegpipe.utils.paths import annotations_csv, file_index_csv
from eegpipe.utils.logging_utils import get_logger

logger = get_logger("01_download")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Download CHB-MIT and build indices")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config.yaml")
    parser.add_argument("--patients", nargs="+", help="Subset of patients to process (e.g., chb01 chb02)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = Path(cfg["paths"]["raw"])
    
    # 1. Metadata and Annotations (from Step 2)
    logger.info("Downloading metadata...")
    download_metadata(cfg)
    
    logger.info("Building annotations...")
    df_annotations = build_annotations(raw_dir, cfg["dataset"]["patient_map"])
    df_annotations.to_csv(annotations_csv(cfg), index=False)
    
    # 2. Cohort Selection
    logger.info("Scanning RECORDS to build file index...")
    df_index = build_file_index(cfg, df_annotations)
    
    if args.patients:
        logger.info(f"Filtering index to requested patients: {args.patients}")
        df_index = df_index[df_index["patient"].isin(args.patients)].copy()

    df_cohort = select_cohort(df_index, cfg)
    
    # 3. Download Included EDFs
    files_to_download = []
    for _, row in df_cohort[df_cohort["include"]].iterrows():
        files_to_download.append(f"{row['case']}/{row['file']}")
        
    logger.info(f"Cohort selection complete. {len(files_to_download)} files marked for download.")
    download_edfs(cfg, files_to_download)
    
    # 4. Post-Download Channel Verification
    logger.info("Verifying channels in downloaded files...")
    required_channels = cfg["dataset"]["channels"]
    aliases = cfg["dataset"].get("channel_aliases", {})
    
    for idx, row in df_cohort[df_cohort["include"]].iterrows():
        edf_path = raw_dir / row["case"] / row["file"]
        if edf_path.exists():
            hdr = read_edf_header(edf_path)
            # Resolve aliases in header to see what we actually have
            actual_channels = [aliases.get(c, c) for c in hdr["ch_names"]]
            missing = [c for c in required_channels if c not in actual_channels]
            
            if missing:
                df_cohort.loc[idx, "include"] = False
                df_cohort.loc[idx, "exclude_reason"] = f"missing_channel:{missing[0]}"
                logger.warning(f"File {row['file']} excluded: missing {missing[0]}")
            else:
                # Update accurate duration now that we have the real file
                df_cohort.loc[idx, "duration_s"] = hdr["duration_s"]
                df_cohort.loc[idx, "n_samples"] = int(round(hdr["duration_s"] * cfg["dataset"]["fs"]))
                
    # 5. Save Contract C2
    out_path = file_index_csv(cfg)
    df_cohort.to_csv(out_path, index=False)
    
    included_hours = df_cohort[df_cohort['include']]['duration_s'].sum() / 3600.0
    logger.info(f"Saved file_index.csv. Final included data volume: {included_hours:.2f} hours.")

if __name__ == "__main__":
    main()