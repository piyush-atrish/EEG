from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import pytest

from eegpipe.io.cohort import select_cohort
from eegpipe.io.loader import ChannelMissingError, load_edf_channels


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
        # Verify MNE Volts to uV scaling (1.0 V * 1e6 = 1000000.0 uV)
        assert np.allclose(data, 1e6)

def test_select_cohort_cap_rules():
    # Setup test with a 2-hour cap limit
    cfg = {"project": {"max_hours_per_patient": 2.0}, "dataset": {"fs": 256}}

    # Synthetic index simulation
    df_index = pd.DataFrame([
        {"patient": "chb01", "case": "chb01", "file": "chb01_01.edf", "duration_s": 3600.0, "role": "seizure_free"},
        {"patient": "chb01", "case": "chb01", "file": "chb01_02.edf", "duration_s": 3600.0, "role": "seizure"},
        {"patient": "chb01", "case": "chb21", "file": "chb21_01.edf", "duration_s": 3600.0, "role": "seizure_free"},
        {"patient": "chb01", "case": "chb21", "file": "chb21_02.edf", "duration_s": 3600.0, "role": "seizure_free"},
    ])

    res = select_cohort(df_index, cfg)

    # 01_01 is the calibration file -> Must be included
    assert res.loc[res["file"] == "chb01_01.edf", "include"].values[0]

    # 01_02 contains a seizure -> Must be included
    assert res.loc[res["file"] == "chb01_02.edf", "include"].values[0]

    # 01_01 (1h) + 01_02 (1h) = 2 hours. Cap is reached.
    # Therefore, 21_01 and 21_02 must be excluded.
    assert not res.loc[res["file"] == "chb21_01.edf", "include"].values[0]
    assert not res.loc[res["file"] == "chb21_02.edf", "include"].values[0]

def test_select_cohort():
    cfg = {"project": {"max_hours_per_patient": 2.0}}
    df = pd.DataFrame({"patient": ["chb01", "chb02", "chb03"], "role": "seizure_free", "duration_s": 3600})
    res = select_cohort(df, cfg, patients=["chb01", "chb02"])
    assert len(res) == 2
    assert "chb03" not in res["patient"].values
