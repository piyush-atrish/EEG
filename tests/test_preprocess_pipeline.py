import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from pathlib import Path
from eegpipe.preprocessing.pipeline import preprocess_file, preprocess_all

def dummy_loader(edf_path, channels, aliases, fs_expected):
    # Simulate a 2-second EDF load returning (18, 512)
    x = np.random.randn(18, int(fs_expected * 2)).astype(np.float32)
    return x, fs_expected

@patch("eegpipe.preprocessing.pipeline.load_edf_channels", side_effect=dummy_loader)
def test_preprocess_file_and_pipeline(mock_loader, tmp_path):
    cfg = {
        "paths": {
            "raw": str(tmp_path / "raw"),
            "preprocessed": str(tmp_path / "preprocessed"),
            "logs": str(tmp_path / "logs")
        },
        "dataset": {"channels": [f"CH{i}" for i in range(18)], "fs": 256},
        "preprocessing": {
            "notch_hz": 60.0, "notch_q": 30.0, "bandpass_hz": [0.5, 45.0],
            "fir": {"window": "kaiser", "ripple_db": 60.0, "transition_hz": 1.0}
        }
    }
    
    df = pd.DataFrame([
        {"patient": "chb01", "case": "chb01", "file": "chb01_01.edf", "include": True, "n_samples": 512},
        {"patient": "chb01", "case": "chb01", "file": "chb01_02.edf", "include": True, "n_samples": 512}
    ])
    
    # 1. Test parallel execution and directory creation
    res_df = preprocess_all(df, cfg, n_jobs=1)
    
    assert len(res_df) == 2
    assert all(res_df["status"] == "success")
    assert (Path(cfg["paths"]["preprocessed"]) / "chb01" / "chb01_01.npy").exists()
    assert (Path(cfg["paths"]["logs"]) / "preprocess_status.csv").exists()
    
    # 2. Test Contract C3 shape and dtype
    out_arr = np.load(Path(cfg["paths"]["preprocessed"]) / "chb01" / "chb01_01.npy")
    assert out_arr.shape == (18, 512)
    assert out_arr.dtype == np.float32
    
    # 3. Test resumability
    res_skipped = preprocess_file(df.iloc[0], cfg, overwrite=False)
    assert res_skipped["status"] == "skipped_existing"

@patch("eegpipe.preprocessing.pipeline.load_edf_channels", side_effect=Exception("Corrupted Header"))
def test_preprocess_error_isolation(mock_loader, tmp_path):
    # Ensure a corrupt file logs a failure but doesn't crash the batch
    cfg = {
        "paths": {"raw": "", "preprocessed": str(tmp_path), "logs": str(tmp_path)},
        "dataset": {"channels": [], "fs": 256}
    }
    row = pd.Series({"case": "chb01", "file": "bad.edf", "n_samples": 0})
    
    res = preprocess_file(row, cfg, overwrite=True)
    assert res["status"] == "failed"
    assert "Corrupted Header" in res["error"]