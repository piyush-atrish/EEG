import logging

import pandas as pd
import pytest

from eegpipe.io.annotations import (
    ANNOTATION_COLUMNS,
    build_annotations,
    build_summary_durations,
    parse_summary_file,
)
from tests.fixtures.synth_signals import make_synthetic_summary_text


def _write(tmp_path, case, text):
    d = tmp_path / case
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{case}-summary.txt").write_text(text)


def test_parse_both_formats(tmp_path):
    p = tmp_path / "chb01-summary.txt"
    p.write_text("Data Sampling Rate: 256 Hz\n\n" + make_synthetic_summary_text("v1")
                 + "\nFile Name: chb01_04.edf\nNumber of Seizures in File: 0\n"
                 + "\n" + make_synthetic_summary_text("v2"))
    recs = parse_summary_file(p)
    assert [r["file"] for r in recs] == ["chb01_03.edf", "chb01_04.edf", "chb04_05.edf"]
    assert recs[0]["seizures"] == [(2996, 3036)]
    assert recs[1]["seizures"] == []
    assert recs[2]["seizures"] == [(7804, 7853), (9081, 9196)]
    assert recs[0]["n_declared"] == 1 and recs[2]["n_declared"] == 2
    assert recs[0]["duration_s"] == 3600.0


def test_build_annotations_maps_patient_and_sorts(tmp_path):
    _write(tmp_path, "chb21", make_synthetic_summary_text("v1").replace("chb01_03", "chb21_03"))
    _write(tmp_path, "chb01", make_synthetic_summary_text("v1"))
    df = build_annotations(tmp_path, {"chb21": "chb01"})
    assert list(df.columns) == ANNOTATION_COLUMNS
    assert list(df["case"]) == ["chb01", "chb21"]           # deterministic order
    assert set(df["patient"]) == {"chb01"}
    assert (df["seizure_end_s"] > df["seizure_start_s"]).all()


def test_zero_seizure_files_do_not_appear(tmp_path):
    _write(tmp_path, "chb01", "File Name: chb01_01.edf\nNumber of Seizures in File: 0\n")
    df = build_annotations(tmp_path, {})
    assert df.empty and list(df.columns) == ANNOTATION_COLUMNS


def test_count_mismatch_warns_and_strict_raises(tmp_path, caplog):
    text = ("File Name: chb01_01.edf\nNumber of Seizures in File: 2\n"
            "Seizure Start Time: 10 seconds\nSeizure End Time: 20 seconds\n")
    _write(tmp_path, "chb01", text)
    with caplog.at_level(logging.WARNING):
        df = build_annotations(tmp_path, {})
    assert len(df) == 1
    assert any("declares 2" in r.message for r in caplog.records)
    with pytest.raises(ValueError, match="annotation problems"):
        build_annotations(tmp_path, {}, strict=True)


def test_invalid_interval_is_skipped(tmp_path):
    text = ("File Name: chb01_01.edf\nNumber of Seizures in File: 1\n"
            "Seizure Start Time: 50 seconds\nSeizure End Time: 40 seconds\n")
    _write(tmp_path, "chb01", text)
    assert build_annotations(tmp_path, {}).empty


def test_durations_handle_midnight_and_hours_over_24(tmp_path):
    text = ("File Name: a.edf\nFile Start Time: 23:59:30\nFile End Time: 00:00:30\n"
            "Number of Seizures in File: 0\n\n"
            "File Name: b.edf\nFile Start Time: 24:10:00\nFile End Time: 25:10:00\n"
            "Number of Seizures in File: 0\n")
    _write(tmp_path, "chb05", text)
    d = build_summary_durations(tmp_path)
    assert d == {"a.edf": 60.0, "b.edf": 3600.0}


def test_result_is_a_dataframe_with_int_seizure_idx(tmp_path):
    _write(tmp_path, "chb04", make_synthetic_summary_text("v2"))
    df = build_annotations(tmp_path, {})
    assert isinstance(df, pd.DataFrame)
    assert list(df["seizure_idx"]) == [0, 1]
