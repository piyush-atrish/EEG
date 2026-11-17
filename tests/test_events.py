"""Tests for evaluation/events.py (event-level metrics) on hand-built predictions."""

import math

import pandas as pd
import pytest

from eegpipe.evaluation.events import event_metrics, event_metrics_table

FS, WINDOW_S, STEP_S = 256, 4.0, 2.0
STEP, WIN = 512, 1024  # samples


def bits(n, on):
    return [1 if i in set(on) else 0 for i in range(n)]


def make_file(y_true, y_pred, case="chb01", file="f1.edf", patient="p1", arm="raw",
              model="svm", drop=()):
    keep = [i for i in range(len(y_true)) if i not in set(drop)]
    return pd.DataFrame(
        {
            "patient": patient,
            "case": case,
            "file": file,
            "start_sample": [i * STEP for i in keep],
            "y_true": [y_true[i] for i in keep],
            "y_pred": [y_pred[i] for i in keep],
            "arm": arm,
            "model": model,
        }
    )


def ev(df, **kwargs):
    return event_metrics(df, FS, WINDOW_S, STEP_S, **kwargs)


N = 30
SEIZURE = [10, 11, 12, 13, 14]


def test_perfect_detection_has_latency_of_one_window():
    truth = bits(N, SEIZURE)
    res = ev(make_file(truth, truth))
    s = res.summary
    assert (s["n_seizures"], s["n_detected"], s["n_false_alarms"]) == (1, 1, 0)
    assert s["event_sensitivity"] == 1.0
    assert s["median_latency_s"] == pytest.approx(4.0)  # the first window must be completed


def test_late_detection_latency():
    res = ev(make_file(bits(N, SEIZURE), bits(N, [12, 13, 14])))
    assert res.summary["median_latency_s"] == pytest.approx(8.0)


def test_missed_seizure():
    res = ev(make_file(bits(N, SEIZURE), bits(N, [])))
    s = res.summary
    assert (s["n_seizures"], s["n_detected"]) == (1, 0)
    assert s["event_sensitivity"] == 0.0
    assert math.isnan(s["median_latency_s"])


def test_false_alarms_are_merged_and_rate_is_per_hour():
    res = ev(make_file(bits(N, SEIZURE), bits(N, [2, 3, 4, 20, 21])))
    s = res.summary
    assert s["n_false_alarms"] == 2  # two bursts, not five windows
    assert s["hours_scored"] == pytest.approx(30 * 2 / 3600)
    assert s["fa_per_hour_event"] == pytest.approx(2 / (60 / 3600))


def test_min_consecutive_removes_isolated_alarm_and_delays_detection():
    truth = bits(N, SEIZURE)
    pred = bits(N, [3, 10, 11, 12, 13, 14, 20, 21])
    base = ev(make_file(truth, pred)).summary
    assert base["n_false_alarms"] == 2
    strict = ev(make_file(truth, pred), min_consecutive=2).summary
    assert strict["n_false_alarms"] == 1  # the isolated window at 3 is gone
    assert strict["median_latency_s"] == pytest.approx(6.0)  # alarm raised one step later


def test_refractory_suppresses_close_alarms():
    truth = bits(N, [])
    pred = bits(N, [2, 12])  # raised 20 s apart
    assert ev(make_file(truth, pred), refractory_s=30).summary["n_false_alarms"] == 1
    assert ev(make_file(truth, pred), refractory_s=20).summary["n_false_alarms"] == 2
    assert ev(make_file(truth, pred), refractory_s=10).summary["n_false_alarms"] == 2


def test_merge_gap_joins_nearby_alarm_runs():
    truth = bits(N, [])
    pred = bits(N, [2, 4])  # starts are 4 s apart
    assert ev(make_file(truth, pred)).summary["n_false_alarms"] == 2
    assert ev(make_file(truth, pred), merge_gap_s=4.0).summary["n_false_alarms"] == 1


def test_seizures_separated_by_dropped_windows_stay_separate():
    truth = bits(20, [5, 6, 7, 10, 11])
    res = ev(make_file(truth, truth, drop=(8, 9)))
    assert res.summary["n_seizures"] == 2 and res.summary["n_detected"] == 2


def test_second_alarm_inside_a_seizure_is_not_a_false_alarm():
    res = ev(make_file(bits(N, SEIZURE), bits(N, [10, 11, 13, 14])))
    s = res.summary
    assert (s["n_alarms"], s["n_false_alarms"], s["n_detected"]) == (2, 0, 1)
    assert s["median_latency_s"] == pytest.approx(4.0)  # first alarm counts


def test_files_are_evaluated_independently():
    a = make_file(bits(N, []), bits(N, [3]), case="chb01", file="a.edf")
    b = make_file(bits(N, []), bits(N, [3]), case="chb02", file="a.edf")
    s = ev(pd.concat([a, b], ignore_index=True)).summary
    assert s["n_false_alarms"] == 2
    assert s["hours_scored"] == pytest.approx(60 * 2 / 3600)


def test_patient_without_seizures_has_undefined_sensitivity():
    s = ev(make_file(bits(N, []), bits(N, [5]))).summary
    assert s["n_seizures"] == 0 and math.isnan(s["event_sensitivity"])
    assert s["n_false_alarms"] == 1


def test_invalid_input_is_rejected():
    df = make_file(bits(N, SEIZURE), bits(N, SEIZURE))
    with pytest.raises(ValueError):
        ev(df.drop(columns=["y_pred"]))
    with pytest.raises(ValueError):
        ev(pd.concat([df, df.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError):
        ev(df, min_consecutive=0)
    with pytest.raises(ValueError):
        event_metrics(df, FS, WINDOW_S, 0.0)


def test_table_groups_by_arm_model_patient():
    hit = bits(N, SEIZURE)
    frames = [
        make_file(hit, hit, patient="p1", arm="raw"),
        make_file(hit, hit, patient="p1", arm="subject_standardised"),
        make_file(hit, bits(N, []), patient="p2", arm="raw"),
        make_file(hit, bits(N, []), patient="p2", arm="subject_standardised"),
    ]
    summary, seizures = event_metrics_table(pd.concat(frames, ignore_index=True), FS, WINDOW_S, STEP_S)
    assert len(summary) == 4 and len(seizures) == 4
    by = summary.set_index(["arm", "patient"])["n_detected"]
    assert by[("raw", "p1")] == 1 and by[("raw", "p2")] == 0
