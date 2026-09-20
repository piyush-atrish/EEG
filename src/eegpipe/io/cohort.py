"""File index (contract C2) and cohort selection.

Selection rules (README Step 3), in order:

1. every file that contains a seizure is included (seizure files are never capped);
2. for every case, its first *usable* file is included as the calibration file;
3. further seizure-free files are added, spread evenly across the recording period, until the
   patient's **seizure-free** hours reach ``dataset.max_seizure_free_hours_per_patient``
   (calibration files count towards that cap; seizure files do not);
4. files listed in ``unusable`` (unreadable, missing channel, failed download...) are excluded
   *before* the above, so the cap is back-filled with usable files;
5. a patient with no usable seizure file is excluded entirely.

Selection is deterministic (no randomness): the same index always gives the same cohort.
"""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path

import pandas as pd

from eegpipe.io.loader import read_edf_header
from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_DURATION_S = 3600.0
INDEX_COLUMNS = [
    "patient",
    "case",
    "file",
    "file_order",
    "duration_s",
    "n_samples",
    "n_seizures",
    "role",
    "include",
    "exclude_reason",
]


def _suffix_number(file_name: str) -> int:
    """Numeric recording index of ``chb17a_03.edf`` -> 3 (robust to lettered series)."""
    match = re.search(r"_(\d+)", Path(file_name).stem)
    return int(match.group(1)) if match else 0


def read_records(raw_dir: Path) -> list[str]:
    """Relative EDF paths listed in the dataset's ``RECORDS`` file."""
    path = Path(raw_dir) / "RECORDS"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the metadata download first")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_file_index(
    cfg: dict,
    annotations: pd.DataFrame,
    summary_durations: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build contract C2 for every EDF listed in ``RECORDS``.

    Durations come from the EDF header when the file is present, otherwise from the summary
    clock times, otherwise a 3600 s guess. ``file_order`` is the chronological rank *within a
    patient* (cases sorted, then the numeric file suffix), starting at 1. Every row starts with
    ``include=False``; call :func:`select_cohort` next.
    """
    raw_dir = Path(cfg["paths"]["raw"])
    fs = float(cfg["dataset"]["fs"])
    patient_map = cfg["dataset"].get("patient_map", {})
    seizure_counts = annotations.groupby("file").size().to_dict() if len(annotations) else {}
    summary_durations = summary_durations or {}

    rows = []
    for rel in read_records(raw_dir):
        case, file_name = rel.split("/", 1)
        duration, n_samples = None, None
        edf = raw_dir / case / file_name
        if edf.exists():
            try:
                header = read_edf_header(edf)
                duration, n_samples = header["duration_s"], header["n_samples"]
            except Exception as exc:  # unreadable files are handled by check_edf later
                logger.warning("Could not read header of %s: %s", edf.name, exc)
        if duration is None:
            duration = summary_durations.get(file_name, DEFAULT_DURATION_S)
            n_samples = int(round(duration * fs))
        n_seizures = int(seizure_counts.get(file_name, 0))
        rows.append(
            {
                "patient": patient_map.get(case, case),
                "case": case,
                "file": file_name,
                "_suffix": _suffix_number(file_name),
                "duration_s": float(duration),
                "n_samples": int(n_samples),
                "n_seizures": n_seizures,
                "role": "seizure" if n_seizures > 0 else "seizure_free",
                "include": False,
                "exclude_reason": "",
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["patient", "case", "_suffix", "file"]).reset_index(drop=True)
    df["file_order"] = df.groupby("patient").cumcount() + 1
    return df[INDEX_COLUMNS]


def bisection_order(n: int) -> list[int]:
    """A permutation of ``range(n)`` that yields evenly spread positions first.

    Midpoint first, then the midpoints of each half, and so on. Taking the first ``k`` entries
    always gives ``k`` positions that are spread across the whole range.
    """
    order: list[int] = []
    queue: deque[tuple[int, int]] = deque([(0, n)])
    while queue:
        lo, hi = queue.popleft()
        if lo >= hi:
            continue
        mid = (lo + hi) // 2
        order.append(mid)
        queue.append((lo, mid))
        queue.append((mid + 1, hi))
    return order


def _set_included(df: pd.DataFrame, idx, included: bool, reason: str) -> None:
    """Set ``include`` and ``exclude_reason`` for the rows ``idx`` (label or list of labels)."""
    df.loc[idx, "include"] = included
    df.loc[idx, "exclude_reason"] = reason


def select_cohort(
    index: pd.DataFrame, cfg: dict, unusable: dict[str, str] | None = None
) -> pd.DataFrame:
    """Apply the selection rules (see module docstring); returns a new frame.

    Parameters
    ----------
    unusable : dict, optional
        ``{file name: reason}`` for files that cannot be used. They are excluded first, with
        that reason, so the cap is filled from usable files only.
    """
    cap_hours = float(cfg["dataset"]["max_seizure_free_hours_per_patient"])
    unusable = unusable or {}
    df = index.copy().reset_index(drop=True)
    df["include"] = False
    df["exclude_reason"] = "over_cap"
    bad = df["file"].isin(unusable)
    df.loc[bad, "exclude_reason"] = df.loc[bad, "file"].map(unusable)

    for _patient, pdf in df.groupby("patient", sort=False):
        usable = pdf[~bad.loc[pdf.index]]
        seizure_rows = usable[usable["role"] == "seizure"]
        if seizure_rows.empty:
            df.loc[usable.index, "exclude_reason"] = "no_seizure_patient"
            continue

        _set_included(df, seizure_rows.index, True, "")

        seizure_free_hours = 0.0
        for _case, cdf in usable.groupby("case", sort=False):
            first = cdf.sort_values("file_order").index[0]
            if not df.loc[first, "include"]:
                _set_included(df, first, True, "")
                seizure_free_hours += df.loc[first, "duration_s"] / 3600.0

        candidates = usable[(usable["role"] == "seizure_free") & (~df.loc[usable.index, "include"])]
        candidates = candidates.sort_values("file_order")
        for position in bisection_order(len(candidates)):
            if seizure_free_hours >= cap_hours:
                break
            idx = candidates.index[position]
            _set_included(df, idx, True, "")
            seizure_free_hours += df.loc[idx, "duration_s"] / 3600.0
    return df


def validate_cohort(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Final consistency pass. Returns ``(cohort, warnings)``.

    * A patient that *has* seizure files but none of them is usable (download failure,
      missing channel...) is reported as data loss: ``"<patient>: no usable seizure file"``.
      Such a patient is excluded entirely, since it could not contribute a positive class.
    * A patient that never had a seizure file (for example chb03 in the fake tree) is simply
      excluded without a warning.
    """
    df = df.copy()
    warnings: list[str] = []
    for patient, pdf in df.groupby("patient", sort=False):
        has_seizure_files = (pdf["role"] == "seizure").any()
        included = pdf[pdf["include"]]
        included_seizure = (included["role"] == "seizure").any()
        if has_seizure_files and not included_seizure:
            _set_included(df, included.index, False, "no_seizure_patient")
            df.loc[pdf.index[~pdf["include"] & (pdf["exclude_reason"] == "over_cap")],
                   "exclude_reason"] = "no_seizure_patient"
            warnings.append(f"{patient}: no usable seizure file, patient excluded")
        elif included.empty:
            continue
        elif not (included["role"] == "seizure_free").any():
            warnings.append(f"{patient}: no seizure-free file included (calibration is short)")
    return df, warnings


def cohort_summary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-patient table of included files and hours, plus an estimate of the cache size."""
    n_ch = len(cfg["dataset"]["channels"])
    fs = float(cfg["dataset"]["fs"])
    inc = df[df["include"]]
    rows = []
    for patient, pdf in inc.groupby("patient", sort=True):
        hours = pdf["duration_s"].sum() / 3600.0
        rows.append(
            {
                "patient": patient,
                "files": len(pdf),
                "seizure_files": int((pdf["role"] == "seizure").sum()),
                "seizure_free_hours": pdf.loc[pdf["role"] == "seizure_free", "duration_s"].sum()
                / 3600.0,
                "total_hours": hours,
                "cache_gb": hours * 3600 * fs * n_ch * 4 / 1e9,
            }
        )
    return pd.DataFrame(rows)
