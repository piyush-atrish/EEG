import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eegpipe.io.loader import load_edf_channels
from eegpipe.preprocessing.filters import apply_filters
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import preprocessed_path

logger = get_logger(__name__)

def preprocess_file(row: pd.Series, cfg: dict, overwrite: bool = False) -> dict:
    case = row["case"]
    file_stem = Path(row["file"]).stem
    out_path = preprocessed_path(cfg, case, file_stem)

    result = {"file": row["file"], "status": "success", "n_samples": 0, "seconds": 0.0, "error": ""}

    if out_path.exists() and not overwrite:
        result["status"] = "skipped_existing"
        result["n_samples"] = row.get("n_samples", 0)
        return result

    try:
        edf_path = Path(cfg["paths"]["raw"]) / case / row["file"]
        x, fs = load_edf_channels(edf_path, cfg["dataset"]["channels"], cfg["dataset"].get("channel_aliases", {}), cfg["dataset"]["fs"])

        flat_channels = np.sum(np.std(x, axis=1) < 1e-6)
        if flat_channels > 0:
            logger.warning(f"{row['file']} has {flat_channels} flat channels.")

        x_filtered = apply_filters(x, fs, cfg)

        # Validation checks
        if np.isnan(x_filtered).any() or np.isinf(x_filtered).any():
            raise ValueError("Filtered signal contains NaN or Inf values")

        expected_samples = row.get("n_samples", 0)
        if expected_samples > 0 and x_filtered.shape[1] != expected_samples:
            raise ValueError(f"Length mismatch: got {x_filtered.shape[1]}, expected {expected_samples}")

        # ATOMIC WRITE: write to temp file then replace to prevent corrupted caching if killed
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix('.npy.tmp')

        # Write via file handle so numpy doesn't auto-append another .npy
        with open(tmp_path, 'wb') as f:
            np.save(f, x_filtered)

        os.replace(tmp_path, out_path)

        result["n_samples"] = x_filtered.shape[1]
        result["seconds"] = x_filtered.shape[1] / fs

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Failed to preprocess {row['file']}: {e}")

    return result

def preprocess_all(file_index: pd.DataFrame, cfg: dict, patients: list[str] | None = None,
                   n_jobs: int = -1, overwrite: bool = False) -> pd.DataFrame:
    df = file_index[file_index["include"]].copy()
    if patients:
        df = df[df["patient"].isin(patients)]

    logger.info(f"Preprocessing {len(df)} files using {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs)(delayed(preprocess_file)(row, cfg, overwrite) for _, row in df.iterrows())
    res_df = pd.DataFrame(results)

    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocess_status.csv"
    if log_file.exists():
        res_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        res_df.to_csv(log_file, index=False)

    failed = len(res_df[res_df["status"] == "failed"])
    logger.info(f"Preprocessing complete. Failed: {failed}/{len(df)}")
    return res_df
