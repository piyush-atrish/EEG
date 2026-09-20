import pytest
import numpy as np
import pandas as pd
import mne
from unittest.mock import patch
from eegpipe.io.loader import load_edf_channels, ChannelMissingError
from eegpipe.io.cohort import select_cohort

def test_load_edf_channels_and_aliases():
    info = mne.create_info(ch_names=["FP1-F7", "T8-P8-0"], sfreq=256, ch_types="eeg")
    # Use 1.0 Volts to ensure the loader correctly scales it to 1,000,000 microvolts
    raw = mne.io.RawArray(np.ones((2, 256)), info) 
    
    with patch("mne.io.read_raw_edf", return_value=raw):
        with pytest.raises(ChannelMissingError, match="Missing channel: MISSING"):
            load_edf_channels("fake.edf", channels=["FP1-F7", "MISSING"], aliases={})
            
        data, fs = load_edf_channels(
            "fake.edf", 
            channels=["FP1-F7", "T8-P8"], 
            aliases={"T8-P8-0": "T8-P8"}
        )
        assert fs == 256
        assert data.shape == (2, 256)
        assert data.dtype == np.float32
        assert np.allclose(data, 1e6) 

def test_select_cohort_cap_rules():
    # Setup test with a 2-hour cap limit under the correct new config section
    cfg = {"dataset": {"max_seizure_free_hours_per_patient": 2.0, "fs": 256}}
    
    # Synthetic index simulation with the required 'file_order' column
    df_index = pd.DataFrame([
        {"patient": "chb01", "case": "chb01", "file": "chb01_01.edf", "file_order": 1, "duration_s": 3600.0, "role": "seizure_free"},
        {"patient": "chb01", "case": "chb01", "file": "chb01_02.edf", "file_order": 2, "duration_s": 3600.0, "role": "seizure"},
        {"patient": "chb01", "case": "chb21", "file": "chb21_01.edf", "file_order": 3, "duration_s": 3600.0, "role": "seizure_free"},
        {"patient": "chb01", "case": "chb21", "file": "chb21_02.edf", "file_order": 4, "duration_s": 3600.0, "role": "seizure_free"},
    ])
    
    res = select_cohort(df_index, cfg)
    
    # 01_01 is the calibration file for case chb01 -> Must be included
    assert res.loc[res["file"] == "chb01_01.edf", "include"].values[0]
    
    # 01_02 contains a seizure -> Must be included (seizure files ignore the cap)
    assert res.loc[res["file"] == "chb01_02.edf", "include"].values[0]
    
    # 21_01 is the calibration file for the SECOND session (case chb21) -> Must be included!
    assert res.loc[res["file"] == "chb21_01.edf", "include"].values[0]

    # The two calibration files (01_01 and 21_01) used up the 2.0 hour cap exactly.
    # Therefore, no extra bisection files are added, and 21_02 is excluded.
    assert not res.loc[res["file"] == "chb21_02.edf", "include"].values[0]

def test_select_cohort_no_seizures():
    # New test ensuring patients without ANY seizures are dropped entirely
    cfg = {"dataset": {"max_seizure_free_hours_per_patient": 2.0}}
    df = pd.DataFrame([
        {"patient": "chb01", "case": "chb01", "file": "f1.edf", "file_order": 1, "role": "seizure", "duration_s": 3600},
        {"patient": "chb02", "case": "chb02", "file": "f2.edf", "file_order": 1, "role": "seizure_free", "duration_s": 3600}
    ])
    res = select_cohort(df, cfg)
    
    # chb01 has a seizure -> included
    assert res.loc[res["patient"] == "chb01", "include"].values[0]
    
    # chb02 has no seizures at all -> excluded
    assert not res.loc[res["patient"] == "chb02", "include"].values[0]
    assert res.loc[res["patient"] == "chb02", "exclude_reason"].values[0] == "no_seizure_patient"