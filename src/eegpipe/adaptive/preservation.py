"""Does an adaptive setting keep a seizure-like burst visible? (chunk 2.6 screen)

Two views of the same synthetic recording (background + one burst), input vs filter output:

* **Feature separability (the pass rule).** Windows (4 s, 50% overlap, as in the pipeline) inside the
  burst are compared with background windows using cheap features (log power, line length, band
  powers, relative band powers, spectral entropy, Hjorth); ``sep = mean 2|AUC - 0.5|``. The fraction
  kept is ``sep(output) / sep(input)``. This is what a downstream classifier can still use.
* **Broadband power contrast (diagnostic only).** ``contrast = burst power / pre-burst power`` and its
  kept fraction. It is *not* used for the pass rule: a whitening filter (NLMS/RLS) removes low-frequency
  power by design, so this number falls even when the burst stays perfectly separable (see
  docs/decisions.md, 2026-11-21).

Reported per run: ``kept_sep``, ``kept_mean`` (power, whole burst), ``kept_tail`` (power, last seconds)
and ``post_ratio`` (power 5-35 s after the burst relative to the pre-burst background, output over
input; values well above 1 mean adaptation to the seizure would raise false alarms afterwards).

The screen removes settings that clearly destroy the burst. It does not choose the winner; that is
the job of the real-data pilot on the development patients (chunk 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata

from eegpipe.adaptive.factory import apply_adaptive, params_tag, resolve_params
from eegpipe.adaptive.synthetic import KINDS, make_recording

DEFAULT_LENGTHS = (5, 15, 30, 60, 120)
RULE_LENGTHS = (15, 30, 60)  # a setting must keep the separability for these burst durations...
THRESHOLD = 0.85  # ...at least this fraction of the feature separability, for every burst kind


def _power(x: np.ndarray, start: int, stop: int) -> float:
    return float(np.mean(x[:, start:stop].astype(np.float64) ** 2))


def contrast_metrics(
    x_in: np.ndarray,
    x_out: np.ndarray,
    onset: int,
    end: int,
    fs: float,
    bg_s: float = 60.0,
    tail_s: float = 10.0,
    post_gap_s: float = 5.0,
    post_s: float = 30.0,
) -> dict:
    """Contrast fractions kept; see the module docstring."""
    bg = (max(0, onset - int(bg_s * fs)), onset)
    tail_len = min(int(tail_s * fs), (end - onset) // 2)
    post = (end + int(post_gap_s * fs), min(x_in.shape[1], end + int((post_gap_s + post_s) * fs)))

    def ratio(a: tuple[int, int]) -> float:
        out = _power(x_out, *a) / _power(x_out, *bg)
        inp = _power(x_in, *a) / _power(x_in, *bg)
        return out / inp

    return {
        "kept_mean": ratio((onset, end)),
        "kept_tail": ratio((end - tail_len, end)),
        "post_ratio": ratio(post) if post[1] > post[0] else float("nan"),
    }


WINDOW_S, STEP_S = 4.0, 2.0
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}


def _window_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Cheap per-window features for one array of windows ``x`` shaped (n_windows, n_samples)."""
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    power = np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)) ** 2 + 1e-20
    in_band = (freqs >= 0.5) & (freqs <= 45.0)
    total = power[:, in_band].sum(axis=1)
    band = np.stack([power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in BANDS.values()], axis=1)
    prob = power[:, in_band] / total[:, None]
    dx = np.diff(x, axis=1)
    ddx = np.diff(dx, axis=1)
    var_x, var_dx, var_ddx = x.var(axis=1) + 1e-20, dx.var(axis=1) + 1e-20, ddx.var(axis=1) + 1e-20
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / mobility
    columns = [
        np.log(var_x),
        np.log(np.abs(dx).mean(axis=1) + 1e-20),  # line length
        *np.log(band).T,
        *(band / band.sum(axis=1, keepdims=True)).T,
        -(prob * np.log(prob)).sum(axis=1),  # spectral entropy
        mobility,
        complexity,
    ]
    return np.stack(columns, axis=1)


def _auc_rows(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    ranks = rankdata(features, axis=0)
    return (ranks[labels == 1].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def window_separability(
    x: np.ndarray, onset: int, end: int, fs: float, bg_s: float = 60.0
) -> float:
    """Mean ``2 |AUC - 0.5|`` over channels and cheap window features, burst vs background windows.

    Burst windows lie fully inside ``[onset, end)``, background windows fully inside the ``bg_s``
    seconds before ``onset`` (4 s windows, 50% overlap, as in the pipeline). 1.0 is perfect
    separation, 0.0 none. Returns NaN if either class has no window.
    """
    win, step = int(WINDOW_S * fs), int(STEP_S * fs)
    values = []
    for channel in x:
        windows = sliding_window_view(channel.astype(np.float64), win)[::step]
        starts = np.arange(windows.shape[0]) * step
        stops = starts + win
        labels = np.full(len(starts), -1)
        labels[(starts >= onset) & (stops <= end)] = 1
        labels[(starts >= onset - int(bg_s * fs)) & (stops <= onset)] = 0
        keep = labels >= 0
        if (labels == 1).sum() < 1 or (labels == 0).sum() < 1:
            return float("nan")
        feats = _window_features(windows[keep], fs)
        values.append(2.0 * np.abs(_auc_rows(feats, labels[keep]) - 0.5))
    return float(np.mean(values))


def run_one(
    method: str,
    params: dict | None,
    burst_s: float,
    kind: str,
    seed: int = 0,
    fs: float = 256.0,
    warm_start_s: float = 60.0,
    **recording_options,
) -> dict:
    """One synthetic recording through one filter setting."""
    recording_options.setdefault("ratio", 2.0)  # not saturated: a 4x burst is separable at any setting
    rec = make_recording(burst_s, kind=kind, fs=fs, seed=seed, **recording_options)
    y, _ = apply_adaptive(rec["x"], method, params, fs=fs, warm_start_s=warm_start_s)
    metrics = contrast_metrics(rec["x"], y, rec["onset"], rec["end"], fs)
    sep_in = window_separability(rec["x"], rec["onset"], rec["end"], fs)
    sep_out = window_separability(y, rec["onset"], rec["end"], fs)
    metrics["sep_in"], metrics["sep_out"] = sep_in, sep_out
    metrics["kept_sep"] = sep_out / sep_in if sep_in > 0 else float("nan")
    return {"method": method, "tag": params_tag(method, params), "burst_s": burst_s,
            "kind": kind, "seed": seed, **metrics}


def screen(
    settings: list[tuple[str, dict | None]],
    burst_lengths=DEFAULT_LENGTHS,
    kinds=KINDS,
    seeds=(0, 1, 2),
    threshold: float = THRESHOLD,
    rule_lengths=RULE_LENGTHS,
    fs: float = 256.0,
    warm_start_s: float = 60.0,
    **recording_options,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every setting over bursts, kinds and seeds. Returns ``(detail, summary)``.

    ``summary`` has one row per setting: ``worst_kept`` is the lowest seed-averaged ``kept_sep``
    over the rule lengths and kinds, and ``passes`` is ``worst_kept >= threshold``.
    """
    rows = []
    for method, params in settings:
        resolved = resolve_params(method, params)
        for burst_s in burst_lengths:
            for kind in kinds:
                for seed in seeds:
                    rows.append(run_one(method, resolved, burst_s, kind, seed, fs, warm_start_s,
                                        **recording_options))
    detail = pd.DataFrame(rows)
    averaged = detail.groupby(["method", "tag", "burst_s", "kind"], as_index=False)[
        ["kept_sep", "kept_mean", "kept_tail", "post_ratio", "sep_in", "sep_out"]
    ].mean()
    in_rule = averaged[averaged["burst_s"].isin(rule_lengths)]
    summary = (
        in_rule.groupby(["method", "tag"], as_index=False)["kept_sep"].min()
        .rename(columns={"kept_sep": "worst_kept"})
    )
    summary["passes"] = summary["worst_kept"] >= threshold
    return averaged, summary.sort_values(["method", "worst_kept"], ascending=[True, False]).reset_index(drop=True)
