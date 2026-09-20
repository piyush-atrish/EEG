"""Windowing and labelling of preprocessed EEG recordings (Contract C4).

Member B, Step 6. Converts each included, preprocessed recording (Contract C3)
into fixed-length, overlapping windows labelled against the seizure
annotations (Contract C1), producing one row per window (Contract C4).

Design decision worth flagging for the viva: the README's Contract C4 spec
defines the label rule only for `drop_boundary=True` ("partial overlap ->
DROPPED"). It leaves the label for a partially-overlapping window
undefined when `drop_boundary=False`. Since that flag exists for
completeness/debugging (the frozen pipeline always runs with
`drop_boundary_windows: true`), we made an explicit choice here: a
partially-overlapping window that is *kept* is labelled ictal (1), on the
reasoning that any window containing seizure activity should not be
counted as a clean negative. This only affects `drop_boundary=False`
callers; it does not change Contract C4 behaviour for the frozen config.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from eegpipe.utils.paths import preprocessed_path

logger = logging.getLogger(__name__)

_WINDOW_COLUMNS = ["start_sample", "t_start_s", "label"]


def file_stem(file: str) -> str:
    """Strip the extension from a Contract C1/C2 `file` value, e.g. `"chb01_01.edf"` -> `"chb01_01"`.

    Shared by `build_window_table` and `extract_case_features` (both need
    it to build a Contract-C3 `.npy` path via `preprocessed_path`) so the
    file-naming convention only needs to change in one place.
    """
    return file.rsplit(".", 1)[0]


def seizure_intervals_for_file(annotations: pd.DataFrame, file: str) -> list[tuple[float, float]]:
    """Return this file's seizure intervals from Contract C1.

    Parameters
    ----------
    annotations : pd.DataFrame
        Contract C1 dataframe (columns include `file`, `seizure_start_s`,
        `seizure_end_s`, and usually `seizure_idx`).
    file : str
        File name to filter on, e.g. ``"chb21_03.edf"``.

    Returns
    -------
    list[tuple[float, float]]
        ``[(start_s, end_s), ...]``, ordered by `seizure_idx` when present,
        otherwise by start time. Empty list if the file has no seizures.
    """
    rows = annotations.loc[annotations["file"] == file]
    if rows.empty:
        return []
    sort_col = "seizure_idx" if "seizure_idx" in rows.columns else "seizure_start_s"
    rows = rows.sort_values(sort_col)
    return list(zip(rows["seizure_start_s"].astype(float), rows["seizure_end_s"].astype(float)))


def _label_window(start_sample: int, end_sample: int, intervals_samples: list[tuple[int, int]]) -> int | None:
    """Label one window against seizure intervals, all in sample space.

    Returns 1 if fully inside a seizure, 0 if it has no overlap with any
    seizure, or None if it partially overlaps a seizure (boundary window).
    """
    overlaps_any = False
    for on_sample, off_sample in intervals_samples:
        if start_sample >= on_sample and end_sample <= off_sample:
            return 1
        if end_sample > on_sample and start_sample < off_sample:
            overlaps_any = True
    return None if overlaps_any else 0


def make_windows(
    n_samples: int,
    fs: float,
    window_s: float,
    overlap: float,
    seizure_intervals: list[tuple[float, float]],
    drop_boundary: bool = True,
) -> pd.DataFrame:
    """Segment one recording into fixed-length windows with seizure labels.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the recording.
    fs : float
        Sampling rate in Hz.
    window_s : float
        Window length in seconds.
    overlap : float
        Fractional overlap between consecutive windows, in [0, 1).
    seizure_intervals : list[tuple[float, float]]
        ``[(start_s, end_s), ...]`` for this file, as returned by
        `seizure_intervals_for_file`.
    drop_boundary : bool, default True
        If True (the frozen Milestone-1 setting), windows that partially
        overlap a seizure are dropped. If False, they are kept and labelled
        1 (see module docstring).

    Returns
    -------
    pd.DataFrame
        Columns `start_sample` (int64), `t_start_s` (float64), `label`
        (int8). Sorted by `start_sample`. Empty (but correctly typed) if no
        window fits.
    """
    W = round(window_s * fs)
    step = round(W * (1 - overlap))
    if step <= 0:
        raise ValueError(f"Non-positive step size ({step}) from window_s={window_s}, overlap={overlap}")

    if n_samples < W:
        return pd.DataFrame(
            {
                "start_sample": pd.array([], dtype="int64"),
                "t_start_s": pd.array([], dtype="float64"),
                "label": pd.array([], dtype="int8"),
            }
        )

    n_windows = (n_samples - W) // step + 1
    starts = np.arange(n_windows, dtype=np.int64) * step

    intervals_samples = [(round(on * fs), round(off * fs)) for on, off in seizure_intervals]

    raw_labels: list[int | None] = [
        _label_window(int(s), int(s) + W, intervals_samples) for s in starts
    ]

    if drop_boundary:
        keep = np.array([lbl is not None for lbl in raw_labels], dtype=bool)
        starts = starts[keep]
        labels = np.array([lbl for lbl in raw_labels if lbl is not None], dtype=np.int8)
    else:
        labels = np.array([1 if lbl is None else lbl for lbl in raw_labels], dtype=np.int8)

    t_start_s = starts.astype(np.float64) / fs
    return pd.DataFrame(
        {
            "start_sample": starts.astype(np.int64),
            "t_start_s": t_start_s,
            "label": labels,
        }
    )


def build_window_table(
    file_index: pd.DataFrame,
    annotations: pd.DataFrame,
    cfg: dict,
    patients: list[str] | None = None,
) -> pd.DataFrame:
    """Build Contract C4 for every included file (optionally restricted to `patients`).

    For every row of `file_index` with `include=True` (and, if given, whose
    `patient` is in `patients`), reads `n_samples` from the Contract-C3
    array's header (via `np.load(mmap_mode="r")`, so the array itself is
    never fully loaded here), windows and labels it, and tags the result
    with `patient`/`case`/`file`.

    Parameters
    ----------
    file_index : pd.DataFrame
        Contract C2.
    annotations : pd.DataFrame
        Contract C1.
    cfg : dict
        Loaded config (uses `dataset.fs` and `segmentation.*`).
    patients : list[str] | None
        Restrict to these patients if given.

    Returns
    -------
    pd.DataFrame
        Contract C4: `patient, case, file, start_sample, t_start_s, label`,
        sorted by (`file`, `start_sample`).
    """
    fs = cfg["dataset"]["fs"]
    seg_cfg = cfg["segmentation"]
    window_s = seg_cfg["window_s"]
    overlap = seg_cfg["overlap"]
    drop_boundary = seg_cfg.get("drop_boundary_windows", True)

    included = file_index.loc[file_index["include"].astype(bool)].copy()
    if patients is not None:
        included = included.loc[included["patient"].isin(patients)]

    frames: list[pd.DataFrame] = []
    for _, row in included.iterrows():
        file_stem_value = file_stem(row["file"])
        npy_path = preprocessed_path(cfg, row["case"], file_stem_value)
        arr = np.load(npy_path, mmap_mode="r")
        n_samples = arr.shape[-1]

        intervals = seizure_intervals_for_file(annotations, row["file"])
        for on_s, off_s in intervals:
            if (off_s - on_s) < window_s:
                logger.warning(
                    "Seizure shorter than window_s in %s (%s to %ss < %ss): "
                    "no ictal window can be fully contained in it.",
                    row["file"], on_s, off_s, window_s,
                )

        win_df = make_windows(n_samples, fs, window_s, overlap, intervals, drop_boundary)
        win_df.insert(0, "file", row["file"])
        win_df.insert(0, "case", row["case"])
        win_df.insert(0, "patient", row["patient"])
        frames.append(win_df)

    if not frames:
        return pd.DataFrame(columns=["patient", "case", "file"] + _WINDOW_COLUMNS)

    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(["file", "start_sample"]).reset_index(drop=True)

    for patient, group in table.groupby("patient"):
        if (group["label"] == 1).sum() == 0:
            logger.warning("Patient %s has zero ictal windows after windowing.", patient)

    return table
