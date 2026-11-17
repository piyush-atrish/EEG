"""Event-level (seizure-level) evaluation of window predictions.

Window-level sensitivity and false positives are dominated by how many windows a seizure
spans. Clinically what matters is: *was each seizure detected, how late, and how many false
alarms per hour were raised.* This module turns the per-window predictions of Contract C6
(``patient, case, file, start_sample, y_true, y_pred, arm, model``) into those numbers.

Definitions (all in sample space, per recording file; files are never merged):

* **Seizure event** -- a run of consecutive ictal windows (``y_true == 1``, start samples
  exactly one step apart). Boundary windows were dropped at windowing time, so two seizures
  never merge. Onset is the start of the first ictal window, which is at most one step
  (2 s) after the annotated onset, because only windows fully inside a seizure are labelled
  ictal. Latency therefore slightly *under*-states the delay measured from the annotation.
* **Alarm** -- predicted-positive windows (``y_pred == 1``) that pass the optional
  ``min_consecutive`` rule (an alarm needs that many consecutive positive windows), then runs
  separated by at most ``merge_gap_s`` are merged into one alarm event. An alarm is *raised*
  at the end of its first qualifying window, because a causal detector has seen the full
  window only then (so the minimum possible latency is ``window_s``).
* **Refractory period** -- after an alarm is raised, further alarms within ``refractory_s``
  (same file) are suppressed.
* **Detected seizure** -- some alarm is raised between the seizure's onset and its end
  (inclusive). Latency = raise time minus onset, of the first such alarm.
* **False alarm** -- an alarm raised outside every seizure. Extra alarms raised inside an
  already-detected seizure are neither detections nor false alarms.
* **False alarms per hour** -- false alarms divided by the scored time, taken (like the
  window-level metric) as ``n_windows * step_s / 3600``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("case", "file", "start_sample", "y_true", "y_pred")
GROUP_COLUMNS = ("arm", "model", "patient")
SUMMARY_COLUMNS = [
    "n_seizures",
    "n_detected",
    "event_sensitivity",
    "n_alarms",
    "n_false_alarms",
    "hours_scored",
    "fa_per_hour_event",
    "median_latency_s",
    "mean_latency_s",
]


@dataclass(frozen=True)
class EventResult:
    """Result for one patient / arm / model: a summary dict and the two detail tables."""

    summary: dict
    seizures: pd.DataFrame  # one row per true seizure
    alarms: pd.DataFrame  # one row per alarm event


def _runs(mask: np.ndarray, starts: np.ndarray, step: int) -> list[tuple[int, int]]:
    """Inclusive index ranges of consecutive True entries whose windows are exactly one step apart."""
    runs: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1] and starts[j + 1] - starts[j] == step:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _alarms(
    starts: np.ndarray,
    positive: np.ndarray,
    step: int,
    window: int,
    min_consecutive: int,
    merge_gap: int,
    refractory: int,
) -> list[tuple[int, int]]:
    """Alarm events of one file as ``(raise_sample, end_sample)`` pairs, in time order."""
    eligible = [(i + min_consecutive - 1, j) for i, j in _runs(positive, starts, step)
                if j - i + 1 >= min_consecutive]
    merged: list[tuple[int, int]] = []
    for first, last in eligible:
        if merged and starts[first] - starts[merged[-1][1]] <= merge_gap:
            merged[-1] = (merged[-1][0], last)
        else:
            merged.append((first, last))
    alarms: list[tuple[int, int]] = []
    last_raised: int | None = None
    for first, last in merged:
        raised = int(starts[first] + window)
        if last_raised is not None and raised - last_raised < refractory:
            continue
        alarms.append((raised, int(starts[last] + window)))
        last_raised = raised
    return alarms


def _validate(pred: pd.DataFrame, step_s: float, min_consecutive: int, refractory_s: float) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in pred.columns]
    if missing:
        raise ValueError(f"predictions are missing column(s): {missing}")
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be >= 1")
    if refractory_s < 0:
        raise ValueError("refractory_s must be >= 0")
    if pred.duplicated(["case", "file", "start_sample"]).any():
        raise ValueError("duplicate (case, file, start_sample) rows in predictions")


def event_metrics(
    pred: pd.DataFrame,
    fs: float,
    window_s: float,
    step_s: float,
    min_consecutive: int = 1,
    merge_gap_s: float | None = None,
    refractory_s: float = 0.0,
) -> EventResult:
    """Event-level metrics for ONE patient / arm / model (one Contract C6 slice).

    ``merge_gap_s`` defaults to one step (only directly adjacent alarm windows merge).
    """
    _validate(pred, step_s, min_consecutive, refractory_s)
    window = int(round(window_s * fs))
    step = int(round(step_s * fs))
    merge_gap = step if merge_gap_s is None else max(step, int(round(merge_gap_s * fs)))
    refractory = int(round(refractory_s * fs))

    seizure_rows: list[dict] = []
    alarm_rows: list[dict] = []
    ordered = pred.sort_values(["case", "file", "start_sample"])
    for (case, file), group in ordered.groupby(["case", "file"], sort=False):
        starts = group["start_sample"].to_numpy(dtype=np.int64)
        y_true = group["y_true"].to_numpy(dtype=int)
        y_pred = group["y_pred"].to_numpy(dtype=int)

        events = [(int(starts[i]), int(starts[j] + window)) for i, j in _runs(y_true == 1, starts, step)]
        alarms = _alarms(starts, y_pred == 1, step, window, min_consecutive, merge_gap, refractory)

        for onset, end in events:
            hits = [a for a in alarms if onset <= a[0] <= end]
            first_hit = hits[0][0] if hits else None
            seizure_rows.append(
                {
                    "case": case,
                    "file": file,
                    "onset_sample": onset,
                    "end_sample": end,
                    "detected": first_hit is not None,
                    "alarm_raise_sample": first_hit,
                    "latency_s": (first_hit - onset) / fs if first_hit is not None else np.nan,
                }
            )
        for raised, end in alarms:
            inside = any(onset <= raised <= e_end for onset, e_end in events)
            alarm_rows.append(
                {
                    "case": case,
                    "file": file,
                    "raise_sample": raised,
                    "end_sample": end,
                    "in_seizure": inside,
                }
            )

    seizures = pd.DataFrame(
        seizure_rows,
        columns=["case", "file", "onset_sample", "end_sample", "detected", "alarm_raise_sample", "latency_s"],
    )
    alarm_table = pd.DataFrame(
        alarm_rows, columns=["case", "file", "raise_sample", "end_sample", "in_seizure"]
    )

    n_seizures = len(seizures)
    n_detected = int(seizures["detected"].sum()) if n_seizures else 0
    n_false = int((~alarm_table["in_seizure"]).sum()) if len(alarm_table) else 0
    hours = len(pred) * step_s / 3600.0
    latencies = seizures["latency_s"].dropna().to_numpy(dtype=float)

    summary = {
        "n_seizures": n_seizures,
        "n_detected": n_detected,
        "event_sensitivity": (n_detected / n_seizures) if n_seizures else float("nan"),
        "n_alarms": len(alarm_table),
        "n_false_alarms": n_false,
        "hours_scored": hours,
        "fa_per_hour_event": (n_false / hours) if hours > 0 else float("nan"),
        "median_latency_s": float(np.median(latencies)) if len(latencies) else float("nan"),
        "mean_latency_s": float(np.mean(latencies)) if len(latencies) else float("nan"),
    }
    return EventResult(summary=summary, seizures=seizures, alarms=alarm_table)


def event_metrics_table(
    predictions: pd.DataFrame, fs: float, window_s: float, step_s: float, **options
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event metrics for every (arm, model, patient) present in ``predictions``.

    Returns ``(summary, seizures)``. ``summary`` has one row per group with the key columns
    followed by :data:`SUMMARY_COLUMNS`; ``seizures`` has one row per true seizure.
    ``options`` are passed to :func:`event_metrics`.
    """
    keys = [c for c in GROUP_COLUMNS if c in predictions.columns]
    if not keys:
        result = event_metrics(predictions, fs, window_s, step_s, **options)
        return pd.DataFrame([result.summary]), result.seizures

    summary_rows: list[dict] = []
    seizure_rows: list[dict] = []
    for values, group in predictions.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        labels = dict(zip(keys, values))
        result = event_metrics(group, fs, window_s, step_s, **options)
        summary_rows.append({**labels, **result.summary})
        # plain records (not DataFrame concatenation) avoid dtype surprises with all-NaN latencies
        seizure_rows.extend({**labels, **row} for row in result.seizures.to_dict("records"))
    summary = pd.DataFrame(summary_rows, columns=[*keys, *SUMMARY_COLUMNS])
    seizures = pd.DataFrame(
        seizure_rows,
        columns=[*keys, "case", "file", "onset_sample", "end_sample",
                 "detected", "alarm_raise_sample", "latency_s"],
    )
    return summary, seizures