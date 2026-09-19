import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from eegpipe.io.loader import load_edf_channels
from eegpipe.preprocessing.filters import apply_filters
from eegpipe.utils.paths import preprocessed_path
from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

def preprocess_file(row: pd.Series, cfg: dict, overwrite: bool = False) -> dict:
    """Load, filter, and save an EDF file to a cached .npy array."""
    case = row["case"]
    file_stem = Path(row["file"]).stem
    out_path = preprocessed_path(cfg, case, file_stem)
    
    result = {
        "file": row["file"],
        "status": "success",
        "n_samples": 0,
        "seconds": 0.0,
        "error": ""
    }
    
    if out_path.exists() and not overwrite:
        result["status"] = "skipped_existing"
        result["n_samples"] = row.get("n_samples", 0)
        return result
        
    try:
        edf_path = Path(cfg["paths"]["raw"]) / case / row["file"]
        channels = cfg["dataset"]["channels"]
        aliases = cfg["dataset"].get("channel_aliases", {})
        fs_expected = cfg["dataset"]["fs"]
        
        # 1. Load
        x, fs = load_edf_channels(edf_path, channels, aliases, fs_expected)
        
        # Log flat channels
        flat_channels = np.sum(np.std(x, axis=1) < 1e-6)
        if flat_channels > 0:
            logger.warning(f"{row['file']} has {flat_channels} flat channels.")
        
        # 2. Filter
        x_filtered = apply_filters(x, fs, cfg)
        
        # 3. Save Contract C3
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, x_filtered)
        
        result["n_samples"] = x_filtered.shape[1]
        result["seconds"] = x_filtered.shape[1] / fs
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Failed to preprocess {row['file']}: {e}")
        
    return result

def preprocess_all(file_index: pd.DataFrame, cfg: dict, patients: list[str] | None = None, 
                   n_jobs: int = -1, overwrite: bool = False) -> pd.DataFrame:
    """Run preprocessing in parallel across the cohort."""
    df = file_index[file_index["include"]].copy()
    
    if patients:
        df = df[df["patient"].isin(patients)]
        
    logger.info(f"Preprocessing {len(df)} files using {n_jobs} workers...")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(preprocess_file)(row, cfg, overwrite) for _, row in df.iterrows()
    )
    
    res_df = pd.DataFrame(results)
    
    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(log_dir / "preprocess_status.csv", index=False)
    
    failed = len(res_df[res_df["status"] == "failed"])
    logger.info(f"Preprocessing complete. Failed: {failed}/{len(df)}")
    
    return res_df