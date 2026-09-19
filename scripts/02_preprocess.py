import sys
import argparse
import pandas as pd
from eegpipe.config import load_config
from eegpipe.preprocessing.pipeline import preprocess_all
from eegpipe.utils.paths import file_index_csv
from eegpipe.utils.logging_utils import get_logger

logger = get_logger("02_preprocess")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess included EDF files into .npy arrays")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--patients", nargs="+", help="Subset of patients to process")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing .npy files")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    
    index_path = file_index_csv(cfg)
    if not index_path.exists():
        logger.error(f"File index not found at {index_path}. Run script 01 first.")
        sys.exit(1)
        
    df_index = pd.read_csv(index_path)
    
    # Cap memory usage: Do not spawn more than 4 workers regardless of CPU count
    safe_n_jobs = min(cfg["project"]["n_jobs"], 4) if cfg["project"]["n_jobs"] > 0 else 4
    
    res_df = preprocess_all(
        file_index=df_index, 
        cfg=cfg, 
        patients=args.patients, 
        n_jobs=safe_n_jobs, 
        overwrite=args.overwrite
    )
    
    failed = len(res_df[res_df["status"] == "failed"])
    if failed > 0:
        logger.error(f"Pipeline finished with {failed} failed files.")
        sys.exit(1)

if __name__ == "__main__":
    main()