import pytest
import numpy as np
import pandas as pd
import mne
from unittest.mock import patch
from eegpipe.io.cohort import select_cohort
from eegpipe.io.loader import load_edf_channels, ChannelMissingError

def test_select_cohort_rules():
    cfg = {"dataset": {"max_seizure_free_hours_per_patient": 2.0}, "project": {"seed": 42}}
    # Synthetic file index
    df = pd.DataFrame([
        {"patient": "chb01", "case": "chb01", "file": "chb01_01.edf", "duration_s": 3600, "role": "seizure_free"},
        {"patient": "chb01", "case": "chb01", "file": "chb01_02.edf", "duration_s": 3600, "role": "seizure"},
        {"patient": "chb01", "case": "chb21", "file": "chb21_01.edf", "duration_s": 3600, "role": "seizure_free"},
        {"patient": "chb01", "case": "chb21", "file": "chb21_02.edf", "duration_s": 3600, "role": "seizure_free"},
    ])
    
    res = select_cohort(df, cfg)
    
    # 01_01 is the first file of chb01 -> Included (1 hour background data)
    assert res.loc[res["file"] == "chb01_01.edf", "include"].values[0]
    
    # 01_02 contains a seizure -> Included (doesn't count towards the background cap)
    assert res.loc[res["file"] == "chb01_02.edf", "include"].values[0]
    
    # 21_01 is the first file of chb21 -> Included (1 hour background data, bringing total to 2 hours)
    assert res.loc[res["file"] == "chb21_01.edf", "include"].values[0]
    
    # 21_02 is seizure-free and the 2.0 hour cap is reached -> Excluded
    assert not res.loc[res["file"] == "chb21_02.edf", "include"].values[0]
    assert res.loc[res["file"] == "chb21_02.edf", "exclude_reason"].values[0] == "over_cap"

def test_load_edf_channels_and_aliases():
    # Create an in-memory MNE Raw object to simulate reading an EDF
    info = mne.create_info(ch_names=["FP1-F7", "T8-P8-0"], sfreq=256, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 256)), info)
    
    with patch("mne.io.read_raw_edf", return_value=raw):
        # 1. Missing channel must fail loudly
        with pytest.raises(ChannelMissingError, match="Missing channel: MISSING"):
            load_edf_channels("fake.edf", channels=["FP1-F7", "MISSING"], aliases={})
            
        # 2. Alias resolution and conversion to microvolts
        data, fs = load_edf_channels(
            "fake.edf", 
            channels=["FP1-F7", "T8-P8"], 
            aliases={"T8-P8-0": "T8-P8"}
        )
        assert fs == 256
        assert data.shape == (2, 256)
        assert data.dtype == np.float32