"""Tests for eegpipe.segmentation.windows (Member B, Step 6).

Uses the frozen Milestone-1 settings (fs=256, window_s=4.0, overlap=0.5,
so W=1024 samples, step=512 samples) throughout, and hand-computed
expectations rather than re-deriving the production formula.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eegpipe.segmentation.windows import (
    build_window_table,
    make_windows,
    seizure_intervals_for_file,
)

FS = 256.0
WINDOW_S = 4.0
OVERLAP = 0.5
W = 1024  # round(WINDOW_S * FS)
STEP = 512  # round(W * (1 - OVERLAP))


# ---------------------------------------------------------------------------
# seizure_intervals_for_file
# ---------------------------------------------------------------------------


def test_seizure_intervals_for_file_filters_and_orders_by_idx():
    ann = pd.DataFrame(
        {
            "file": ["a.edf", "a.edf", "b.edf"],
            "seizure_idx": [1, 0, 0],
            "seizure_start_s": [500.0, 100.0, 10.0],
            "seizure_end_s": [540.0, 130.0, 20.0],
        }
    )
    assert seizure_intervals_for_file(ann, "a.edf") == [(100.0, 130.0), (500.0, 540.0)]
    assert seizure_intervals_for_file(ann, "b.edf") == [(10.0, 20.0)]


def test_seizure_intervals_for_file_no_seizures_in_file():
    ann = pd.DataFrame(
        {
            "file": ["a.edf"],
            "seizure_idx": [0],
            "seizure_start_s": [1.0],
            "seizure_end_s": [2.0],
        }
    )
    assert seizure_intervals_for_file(ann, "c.edf") == []


def test_seizure_intervals_for_file_empty_annotations():
    ann = pd.DataFrame({"file": [], "seizure_start_s": [], "seizure_end_s": []})
    assert seizure_intervals_for_file(ann, "a.edf") == []


# ---------------------------------------------------------------------------
# make_windows: window-count formula, hand-computed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_samples,expected_n",
    [
        (W - 1, 0),           # shorter than one window -> no windows
        (W, 1),                # exactly one window fits
        (W + STEP - 1, 1),     # one sample short of a second window
        (W + STEP, 2),         # exactly two windows
        (W + 5 * STEP, 6),     # six windows
        (W + 19 * STEP, 20),   # twenty windows
    ],
)
def test_window_count_formula(n_samples, expected_n):
    df = make_windows(n_samples, FS, WINDOW_S, OVERLAP, seizure_intervals=[])
    assert len(df) == expected_n


def test_no_overlap_gives_label_zero_for_all_windows():
    n_samples = W + 4 * STEP  # 5 windows
    df = make_windows(n_samples, FS, WINDOW_S, OVERLAP, seizure_intervals=[])
    assert len(df) == 5
    assert (df["label"] == 0).all()


# ---------------------------------------------------------------------------
# Labelling: single seizure, known onset/offset -> hand-counted ictal windows
# ---------------------------------------------------------------------------


def test_single_seizure_known_ictal_count():
    # 10 windows total: starts at samples 0, 512, ..., 4608.
    n_samples = W + 9 * STEP
    # Seizure fully contains windows starting at samples 2*STEP and 3*STEP
    # (i.e. [1024, 1536+1024)=[1024,2560) and [1536,1536+1024)=[1536,2560)):
    # on_sample = 2*STEP = 1024, off_sample = 3*STEP + W = 1536 + 1024 = 2560.
    on_sample, off_sample = 2 * STEP, 3 * STEP + W
    on_s, off_s = on_sample / FS, off_sample / FS

    df = make_windows(n_samples, FS, WINDOW_S, OVERLAP, seizure_intervals=[(on_s, off_s)])

    # Hand-verified: windows starting at 1024 ([1024,2048)) and 1536 ([1536,2560))
    # are both fully inside [1024, 2560) -> exactly 2 ictal windows.
    assert (df["label"] == 1).sum() == 2
    ictal_starts = set(df.loc[df["label"] == 1, "start_sample"])
    assert ictal_starts == {2 * STEP, 3 * STEP}


def test_two_seizures_in_one_file():
    n_samples = W + 19 * STEP  # 20 windows: starts 0..9728 step 512
    # Seizure A fully contains only the window starting at sample 0: [0, 1024).
    seizure_a = (0.0, W / FS)
    # Seizure B fully contains only the window starting at sample 15*STEP=7680: [7680, 8704).
    seizure_b = (15 * STEP / FS, (15 * STEP + W) / FS)

    df = make_windows(n_samples, FS, WINDOW_S, OVERLAP, seizure_intervals=[seizure_a, seizure_b])

    ictal_starts = set(df.loc[df["label"] == 1, "start_sample"])
    assert ictal_starts == {0, 15 * STEP}


def test_seizure_at_very_start_of_file():
    n_samples = W + 9 * STEP
    df = make_windows(n_samples, FS, WINDOW_S, OVERLAP, seizure_intervals=[(0.0, W / FS)])
    first = df.iloc[0]
    assert first["start_sample"] == 0
    assert first["label"] == 1


def test_seizure_at_very_end_of_file():
    n_samples = W + 9 * STEP
    last_start = 9 * STEP
    on_s, off_s = last_start / FS, n_samples / FS
    df = make_windows(n_samples, FS, WINDOW_S, OVERLAP, seizure_intervals=[(on_s, off_s)])
    last = df.iloc[-1]
    assert last["start_sample"] == last_start
    assert last["label"] == 1


def test_straddling_window_is_dropped_by_default():
    n_samples = W + 9 * STEP  # windows at 0,512,...,4608
    # Seizure boundary falls mid-window: on at sample 1200 (inside window
    # starting at 512: [512,1536)), off at sample 2200 (inside window
    # starting at 1536: [1536,2560)).
    on_sample, off_sample = 1200, 2200
    on_s, off_s = on_sample / FS, off_sample / FS

    df = make_windows(n_samples, FS, WINDOW_S, OVERLAP, seizure_intervals=[(on_s, off_s)])

    # Window starting at 512 -> [512,1536): overlaps [1200,2200) but not fully
    # inside -> dropped. Window starting at 1024 -> [1024,2048): fully inside
    # [1200? no -- 1024 < 1200, so NOT fully inside either -> dropped too
    # (partial overlap). Window starting at 1536 -> [1536,2560): overlaps but
    # extends past 2200 -> dropped.
    assert 512 not in set(df["start_sample"])
    assert 1024 not in set(df["start_sample"])
    assert 1536 not in set(df["start_sample"])
    # Window starting at 0 -> [0,1024): no overlap with [1200,2200) -> label 0, kept.
    assert df.loc[df["start_sample"] == 0, "label"].iloc[0] == 0


def test_window_spanning_gap_between_two_seizures_is_dropped():
    n_samples = W + 9 * STEP  # windows at 0,512,...,4608
    # Two seizures with a short seizure-free gap between them, positioned so
    # that the window starting at sample 1536 ([1536,2560)) spans the gap:
    # seizure 1 ends at sample 2000, seizure 2 starts at sample 2100.
    seizure1 = (0.0, 2000 / FS)
    seizure2 = (2100 / FS, n_samples / FS)

    df = make_windows(n_samples, FS, WINDOW_S, OVERLAP, seizure_intervals=[seizure1, seizure2])

    # Window [1536, 2560) overlaps seizure1 (ends 2000, so overlap [1536,2000))
    # and seizure2 (starts 2100, overlap [2100,2560)), but is fully inside
    # neither -> dropped.
    assert 1536 not in set(df["start_sample"])
    # Window [0, 1024) is fully inside seizure1 ([0,2000)) -> kept, label 1.
    assert df.loc[df["start_sample"] == 0, "label"].iloc[0] == 1
    # Window [2560, 3584) is fully inside seizure2 ([2100, n_samples)) -> kept, label 1.
    assert df.loc[df["start_sample"] == 2560, "label"].iloc[0] == 1


def test_drop_boundary_false_retains_all_windows_and_labels_partials_ictal():
    n_samples = W + 9 * STEP
    on_sample, off_sample = 1200, 2200
    on_s, off_s = on_sample / FS, off_sample / FS

    df_dropped = make_windows(
        n_samples, FS, WINDOW_S, OVERLAP, seizure_intervals=[(on_s, off_s)], drop_boundary=True
    )
    df_kept = make_windows(
        n_samples, FS, WINDOW_S, OVERLAP, seizure_intervals=[(on_s, off_s)], drop_boundary=False
    )

    n_total_possible = (n_samples - W) // STEP + 1
    assert len(df_kept) == n_total_possible
    assert len(df_kept) > len(df_dropped)
    # The three straddling windows (512, 1024, 1536) are retained and marked ictal.
    for s in (512, 1024, 1536):
        assert df_kept.loc[df_kept["start_sample"] == s, "label"].iloc[0] == 1
    # A clean negative (window at 0) is still label 0.
    assert df_kept.loc[df_kept["start_sample"] == 0, "label"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Output contract: dtypes and sorting
# ---------------------------------------------------------------------------


def test_output_dtypes_and_sorting():
    n_samples = W + 5 * STEP
    df = make_windows(n_samples, FS, WINDOW_S, OVERLAP, seizure_intervals=[(1.0, 3.0)])
    assert list(df.columns) == ["start_sample", "t_start_s", "label"]
    assert df["start_sample"].dtype == np.int64
    assert df["t_start_s"].dtype == np.float64
    assert df["label"].dtype == np.int8
    assert (df["start_sample"].diff().dropna() > 0).all()
    # t_start_s = start_sample / fs
    assert np.allclose(df["t_start_s"].to_numpy(), df["start_sample"].to_numpy() / FS)


def test_empty_recording_returns_typed_empty_frame():
    df = make_windows(W - 1, FS, WINDOW_S, OVERLAP, seizure_intervals=[])
    assert len(df) == 0
    assert list(df.columns) == ["start_sample", "t_start_s", "label"]
    assert df["start_sample"].dtype == np.int64
    assert df["label"].dtype == np.int8


# ---------------------------------------------------------------------------
# build_window_table: local synthetic fixture (does not depend on Member A's
# tests/fixtures/synth_signals.py, which is A's own file to own/write)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_project(tmp_path: Path):
    """Minimal local stand-in for a preprocessed cohort: one patient, two files.

    Writes the .npy arrays via the same `preprocessed_path` helper that
    `build_window_table` reads through, so this fixture stays correct
    regardless of Member A's actual on-disk layout convention — it never
    assumes that layout independently.
    """
    from eegpipe.utils.paths import preprocessed_path

    n_samples_1 = W + 9 * STEP  # 10 windows
    n_samples_2 = W + 4 * STEP  # 5 windows

    cfg = {
        "paths": {"preprocessed": str(tmp_path / "preprocessed")},
        "dataset": {"fs": FS},
        "segmentation": {"window_s": WINDOW_S, "overlap": OVERLAP, "drop_boundary_windows": True},
    }

    for file_stem, n in [("chb01_01", n_samples_1), ("chb01_02", n_samples_2)]:
        path = preprocessed_path(cfg, "chb01", file_stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.zeros((18, n), dtype=np.float32))

    file_index = pd.DataFrame(
        {
            "patient": ["chb01", "chb01"],
            "case": ["chb01", "chb01"],
            "file": ["chb01_01.edf", "chb01_02.edf"],
            "include": [True, True],
        }
    )
    annotations = pd.DataFrame(
        {
            "patient": ["chb01"],
            "case": ["chb01"],
            "file": ["chb01_01.edf"],
            "seizure_idx": [0],
            "seizure_start_s": [0.0],
            "seizure_end_s": [W / FS],
        }
    )

    return file_index, annotations, cfg


def test_build_window_table_contract_columns_and_sorting(tiny_project):
    file_index, annotations, cfg = tiny_project
    table = build_window_table(file_index, annotations, cfg)

    assert list(table.columns) == ["patient", "case", "file", "start_sample", "t_start_s", "label"]
    assert set(table["file"].unique()) == {"chb01_01.edf", "chb01_02.edf"}
    for _, group in table.groupby("file"):
        assert (group["start_sample"].diff().dropna() > 0).all()

    file1 = table.loc[table["file"] == "chb01_01.edf"]
    # 10 windows possible; the seizure (samples [0,1024)) exactly matches the
    # window length, so the window starting at 512 (50% overlap) straddles
    # the seizure boundary and is dropped -> 9 remain.
    assert len(file1) == 9
    assert (file1["label"] == 1).sum() == 1  # seizure at the very start -> one ictal window

    file2 = table.loc[table["file"] == "chb01_02.edf"]
    assert len(file2) == 5
    assert (file2["label"] == 0).all()  # no seizure annotated in this file


def test_build_window_table_respects_patient_filter(tiny_project):
    file_index, annotations, cfg = tiny_project
    table = build_window_table(file_index, annotations, cfg, patients=["chb99"])
    assert table.empty
    assert list(table.columns) == ["patient", "case", "file", "start_sample", "t_start_s", "label"]


def test_build_window_table_excludes_non_included_files(tiny_project):
    file_index, annotations, cfg = tiny_project
    file_index = file_index.copy()
    file_index.loc[file_index["file"] == "chb01_02.edf", "include"] = False

    table = build_window_table(file_index, annotations, cfg)
    assert set(table["file"].unique()) == {"chb01_01.edf"}


def test_build_window_table_handles_int_include_column(tiny_project):
    # Defends against Member A's file_index.csv encoding `include` as 0/1
    # (int64, as pd.read_csv would infer) rather than True/False (bool).
    # Without explicit `.astype(bool)`, pandas' `.loc[int_array]` silently
    # does LABEL-based indexing instead of boolean masking -- a silent
    # correctness bug, not a crash -- so this must stay green.
    file_index, annotations, cfg = tiny_project
    file_index = file_index.copy()
    file_index["include"] = file_index["include"].astype(int)  # -> 1, 1

    table = build_window_table(file_index, annotations, cfg)
    assert set(table["file"].unique()) == {"chb01_01.edf", "chb01_02.edf"}


def test_build_window_table_warns_on_zero_ictal_patient(tiny_project, caplog):
    file_index, annotations, cfg = tiny_project
    annotations = annotations.iloc[0:0]  # no seizures at all -> zero ictal windows

    with caplog.at_level("WARNING"):
        table = build_window_table(file_index, annotations, cfg)

    assert (table["label"] == 1).sum() == 0
    assert any("zero ictal windows" in message for message in caplog.messages)
