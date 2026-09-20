"""Parse CHB-MIT ``chbXX-summary.txt`` files into the annotations table (contract C1).

Two summary layouts exist and both are supported::

    Seizure Start Time: 2996 seconds          (single seizure)
    Seizure 1 Start Time: 1467 seconds        (numbered, several seizures)

The parser cross-checks the number of parsed seizures against the ``Number of Seizures in File``
line, because a silent miscount would corrupt the labels of every downstream window.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

ANNOTATION_COLUMNS = [
    "patient",
    "case",
    "file",
    "seizure_idx",
    "seizure_start_s",
    "seizure_end_s",
]

_RE_BLOCK = re.compile(r"(?m)^File Name:\s*")
_RE_TIME = r"(\d+):(\d+):(\d+)"
_RE_START_CLOCK = re.compile(rf"File Start Time:\s*{_RE_TIME}")
_RE_END_CLOCK = re.compile(rf"File End Time:\s*{_RE_TIME}")
_RE_DECLARED = re.compile(r"Number of Seizures in File:\s*(\d+)")
_RE_SZ_START = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds")
_RE_SZ_END = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds")


def _clock_seconds(match: re.Match | None) -> int | None:
    """Seconds since midnight of an ``H:M:S`` match (hours may exceed 23 in some summaries)."""
    if match is None:
        return None
    h, m, s = (int(g) for g in match.groups())
    return h * 3600 + m * 60 + s


def _duration_from_clock(start: int | None, end: int | None) -> float | None:
    """File duration from start/end clock times, handling the midnight wrap."""
    if start is None or end is None:
        return None
    if end < start:
        end += 86400
    return float(end - start)


def parse_summary_file(path: Path) -> list[dict]:
    """Parse one summary file.

    Returns one dict per recording with keys ``file``, ``seizures`` (list of ``(start_s, end_s)``),
    ``n_declared`` (int or None) and ``duration_s`` (float or None, from the clock times).
    """
    text = Path(path).read_text(errors="replace")
    records = []
    for block in _RE_BLOCK.split(text)[1:]:  # element 0 is the header before the first file
        file_name = block.splitlines()[0].strip()
        starts = [int(x) for x in _RE_SZ_START.findall(block)]
        ends = [int(x) for x in _RE_SZ_END.findall(block)]
        if len(starts) != len(ends):
            logger.warning(
                "%s: %d seizure start times but %d end times", file_name, len(starts), len(ends)
            )
        declared = _RE_DECLARED.search(block)
        records.append(
            {
                "file": file_name,
                "seizures": list(zip(starts, ends, strict=False)),
                "n_declared": int(declared.group(1)) if declared else None,
                "duration_s": _duration_from_clock(
                    _clock_seconds(_RE_START_CLOCK.search(block)),
                    _clock_seconds(_RE_END_CLOCK.search(block)),
                ),
            }
        )
    return records


def _summary_files(raw_dir: Path) -> list[Path]:
    return sorted(Path(raw_dir).rglob("*-summary.txt"))


def build_annotations(raw_dir: Path, patient_map: dict, strict: bool = False) -> pd.DataFrame:
    """Concatenate every summary into contract C1 (one row per seizure, deterministic order).

    Parameters
    ----------
    strict : bool
        Raise ``ValueError`` on any count mismatch or invalid interval instead of logging a
        warning and continuing.
    """
    rows: list[dict] = []
    problems: list[str] = []
    for summary in _summary_files(raw_dir):
        case = summary.stem.split("-")[0]
        patient = patient_map.get(case, case)
        for rec in parse_summary_file(summary):
            valid = []
            for start, end in rec["seizures"]:
                if end <= start:
                    problems.append(f"{rec['file']}: invalid seizure interval {start}-{end} s")
                    continue
                valid.append((start, end))
            if rec["n_declared"] is not None and rec["n_declared"] != len(rec["seizures"]):
                problems.append(
                    f"{rec['file']}: summary declares {rec['n_declared']} seizures, "
                    f"parsed {len(rec['seizures'])}"
                )
            for idx, (start, end) in enumerate(sorted(valid)):
                rows.append(
                    {
                        "patient": patient,
                        "case": case,
                        "file": rec["file"],
                        "seizure_idx": idx,
                        "seizure_start_s": float(start),
                        "seizure_end_s": float(end),
                    }
                )
    for message in problems:
        logger.warning("annotation check: %s", message)
    if problems and strict:
        raise ValueError(f"{len(problems)} annotation problems, first: {problems[0]}")
    df = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
    return df.sort_values(["case", "file", "seizure_idx"]).reset_index(drop=True)


def build_summary_durations(raw_dir: Path) -> dict[str, float]:
    """Map ``file name -> duration in seconds`` from summary clock times (midnight-safe).

    Used to estimate lengths of files that have not been downloaded yet, so that the data cap
    is not computed from a blanket 3600 s guess (some CHB-MIT cases have 2 h or 4 h files).
    """
    durations: dict[str, float] = {}
    for summary in _summary_files(raw_dir):
        for rec in parse_summary_file(summary):
            if rec["duration_s"] is not None:
                durations[rec["file"]] = rec["duration_s"]
    return durations