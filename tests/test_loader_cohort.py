from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import pytest

from eegpipe.io.cohort import select_cohort
from eegpipe.io.loader import (
    ChannelMissingError,
    check_edf,
    load_edf_channels,
    read_edf_header,
    resolve_channel_names,
)
from tests.fixtures.synth_signals import write_synthetic_edf

CH = ["FP1-F7", "F7-T7", "T8-P8"]

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


def _edf(tmp_path, labels, fs=256, seconds=20, seed=0, name="x.edf"):
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 30, size=(len(labels), fs * seconds))
    path = tmp_path / name
    write_synthetic_edf(path, labels, data, fs)
    return path, data


# ------------------------------------------------------------------ loader on real EDFs
def test_loader_returns_microvolts_in_requested_order(tmp_path):
    path, data = _edf(tmp_path, ["F7-T7", "-", "FP1-F7", "T8-P8", "T8-P8"])
    out, fs = load_edf_channels(path, CH, {"T8-P8-0": "T8-P8"})
    assert fs == 256 and out.dtype == np.float32 and out.shape == (3, 256 * 20)
    assert np.abs(out[0] - data[2]).max() < 0.02      # FP1-F7 first although stored 3rd
    assert np.abs(out[1] - data[0]).max() < 0.02      # F7-T7
    assert np.abs(out[2] - data[3]).max() < 0.02      # first T8-P8 duplicate (-0)


def test_loader_duplicate_fallback_without_alias(tmp_path):
    path, data = _edf(tmp_path, ["FP1-F7", "F7-T7", "T8-P8", "T8-P8"])
    out, _ = load_edf_channels(path, CH, aliases={})
    assert np.abs(out[2] - data[2]).max() < 0.02


def test_loader_chunking_gives_identical_result(tmp_path):
    path, _ = _edf(tmp_path, ["FP1-F7", "F7-T7", "T8-P8", "T8-P8"], seconds=30)
    whole, _ = load_edf_channels(path, CH, {}, chunk_s=600)
    chunked, _ = load_edf_channels(path, CH, {}, chunk_s=7)
    np.testing.assert_array_equal(whole, chunked)


def test_loader_missing_channel_and_wrong_fs(tmp_path):
    path, _ = _edf(tmp_path, ["FP1-F7", "F7-T7"])
    with pytest.raises(ChannelMissingError, match="Missing channel: T8-P8"):
        load_edf_channels(path, CH, {})
    path128, _ = _edf(tmp_path, ["FP1-F7", "F7-T7", "T8-P8"], fs=128, name="lo.edf")
    with pytest.raises(ValueError, match="Expected fs=256"):
        load_edf_channels(path128, CH, {})


def test_header_has_exact_sample_count(tmp_path):
    path, _ = _edf(tmp_path, ["FP1-F7"], seconds=17)
    h = read_edf_header(path)
    assert h["n_samples"] == 17 * 256 and h["duration_s"] == 17.0   # no off-by-one


def test_check_edf_reasons(tmp_path):
    ok_path, _ = _edf(tmp_path, ["FP1-F7", "F7-T7", "T8-P8", "T8-P8"])
    assert check_edf(ok_path, CH, {}, 256)[:2] == (True, "")
    miss, _ = _edf(tmp_path, ["FP1-F7", "F7-T7"], name="m.edf")
    assert check_edf(miss, CH, {}, 256)[:2] == (False, "missing_channel:T8-P8")
    bad = tmp_path / "corrupt.edf"
    bad.write_bytes(b"not an edf")
    ok, reason, _ = check_edf(bad, CH, {}, 256)
    assert not ok and reason.startswith("unreadable_edf")


def test_resolve_channel_names_prefers_exact_then_alias():
    assert resolve_channel_names(["A", "B-0"], ["A", "B"], {}) == {"A": "A", "B": "B-0"}
    assert resolve_channel_names(["A", "Z"], ["A", "B"], {"Z": "B"}) == {"A": "A", "B": "Z"}
    with pytest.raises(ChannelMissingError):
        resolve_channel_names(["A"], ["A", "B"], {})


# ------------------------------------------------------------------ cohort rules
def _index(rows):
    df = pd.DataFrame(rows)
    df["duration_s"] = df.get("duration_s", 3600.0)
    df["n_samples"] = 921600
    df["n_seizures"] = (df["role"] == "seizure").astype(int)
    return df


def _rows(patient, roles, case=None, start=1):
    case = case or patient
    return [{"patient": patient, "case": case, "file": f"{case}_{i:02d}.edf",
             "file_order": start + i - 1, "role": r} for i, r in enumerate(roles, 1)]


def test_seizure_files_never_count_towards_the_cap(repo_cfg):
    """The bug in the old logic: 11 seizure files used the whole 2 h cap."""
    roles = ["seizure_free"] + ["seizure"] * 11 + ["seizure_free"] * 30
    res = select_cohort(_index(_rows("chb01", roles)), repo_cfg)
    inc = res[res["include"]]
    assert (inc["role"] == "seizure").sum() == 11
    assert (inc["role"] == "seizure_free").sum() == 2          # 1 calibration + 1 extra = 2 h


def test_cap_counts_calibration_files_of_every_case(repo_cfg):
    rows = _rows("chb01", ["seizure_free", "seizure", "seizure_free"]) \
        + _rows("chb01", ["seizure_free", "seizure_free", "seizure"], case="chb21", start=4)
    res = select_cohort(_index(rows), repo_cfg)
    included = set(res.loc[res["include"], "file"])
    assert {"chb01_01.edf", "chb21_01.edf"} <= included        # a calibration file per case
    assert "chb21_02.edf" not in included                       # cap (2 h) already used


def test_extra_files_are_spread_evenly(repo_cfg):
    cfg = {"dataset": {"max_seizure_free_hours_per_patient": 4.0}}
    roles = ["seizure_free", "seizure"] + ["seizure_free"] * 8       # 1 cal + 3 extras
    res = select_cohort(_index(_rows("chb09", roles)), cfg)
    picked = sorted(res.loc[res["include"] & (res["role"] == "seizure_free"), "file_order"])
    assert picked[0] == 1 and len(picked) == 4
    gaps = np.diff(picked[1:])
    assert gaps.min() >= 2                                         # not clustered


def test_real_durations_drive_the_cap():
    cfg = {"dataset": {"max_seizure_free_hours_per_patient": 2.0}}
    rows = _rows("chb04", ["seizure_free", "seizure", "seizure_free", "seizure_free"])
    df = _index(rows)
    df["duration_s"] = 4 * 3600.0                                  # 4-hour files
    res = select_cohort(df, cfg)

    # chb01 has a seizure -> included
    assert res.loc[res["patient"] == "chb01", "include"].values[0]

    # chb02 has no seizures at all -> excluded
    assert not res.loc[res["patient"] == "chb02", "include"].values[0]
    assert res.loc[res["patient"] == "chb02", "exclude_reason"].values[0] == "no_seizure_patient"
