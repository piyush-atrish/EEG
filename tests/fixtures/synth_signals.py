"""Synthetic data generators for Member A's modules (and reused by B and C).

Everything is written in the exact contract layout of the README (C1, C2, C3), so downstream
code can be developed and tested without any real CHB-MIT data.

Guarantees of :func:`make_synthetic_project` (all covered by ``tests/test_synth_signals.py``):

* every patient has 1-2 seizures, never overlapping within a file, separated by >= 15 s;
* the first file of every case is seizure-free (calibration file);
* patient ``chb01`` consists of TWO cases (``chb01`` and ``chb21``), like the real dataset;
* ``file_order`` is a chronological rank *within patient* (unique across that patient's cases);
* patients differ in spectral colour (AR(1) coefficient) and gain (inter-subject variability);
* everything is written under ``root`` (nothing leaks into the current directory).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from eegpipe.config import find_repo_root, load_config
from eegpipe.utils.paths import annotations_csv, ensure_parent, file_index_csv, preprocessed_path

EXTRA_CHANNELS = ["P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"]  # present in real 23-channel files


# --------------------------------------------------------------------------------------
# Config helper
# --------------------------------------------------------------------------------------
def make_test_cfg(root: Path, base_cfg: dict | None = None) -> dict:
    """Return a config whose every ``paths`` entry lives under ``root`` (e.g. ``tmp_path``)."""
    root = Path(root)
    base = base_cfg or load_config()
    repo = find_repo_root()
    new_paths = {}
    for key, value in base["paths"].items():
        try:
            rel = Path(value).relative_to(repo)
        except ValueError:
            rel = Path(key)
        new_paths[key] = str(root / rel)
    return load_config(overrides={"paths": new_paths})


# --------------------------------------------------------------------------------------
# Summary text and raw-array generators
# --------------------------------------------------------------------------------------
def make_synthetic_summary_text(kind: str = "v1") -> str:
    """Two real-world CHB-MIT summary layouts. ``v1``: single seizure, ``v2``: numbered."""
    if kind == "v1":
        return (
            "File Name: chb01_03.edf\n"
            "File Start Time: 13:43:04\n"
            "File End Time: 14:43:04\n"
            "Number of Seizures in File: 1\n"
            "Seizure Start Time: 2996 seconds\n"
            "Seizure End Time: 3036 seconds\n"
        )
    if kind == "v2":
        return (
            "File Name: chb04_05.edf\n"
            "File Start Time: 13:43:04\n"
            "File End Time: 14:43:04\n"
            "Number of Seizures in File: 2\n"
            "Seizure 1 Start Time: 7804 seconds\n"
            "Seizure 1 End Time: 7853 seconds\n"
            "Seizure 2 Start Time: 9081 seconds\n"
            "Seizure 2 End Time: 9196 seconds\n"
        )
    raise ValueError("kind must be 'v1' or 'v2'")


def make_synthetic_raw_array(
    fs: int = 256,
    seconds: float = 30.0,
    tones_hz: tuple[float, ...] = (10.0, 60.0),
    n_ch: int = 18,
    seed: int = 0,
) -> np.ndarray:
    """Sum of unit sinusoids at ``tones_hz`` plus small Gaussian noise; (n_ch, n) float32."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    tone = np.zeros_like(t)
    for f in tones_hz:
        tone += np.sin(2 * np.pi * f * t)
    noise = rng.normal(0.0, 0.1, size=(n_ch, t.size))
    return (tone + noise).astype(np.float32)


# --------------------------------------------------------------------------------------
# Synthetic project in contract layout (C1, C2, C3)
# --------------------------------------------------------------------------------------
def _place_seizures(
    rng: np.random.Generator,
    n: int,
    duration_s: float,
    min_len: float = 10.0,
    max_len: float = 20.0,
    margin: float = 10.0,
    min_gap: float = 15.0,
) -> list[tuple[float, float]]:
    """Draw ``n`` sorted, non-overlapping seizure intervals inside a file (rejection sampling)."""
    if n == 0:
        return []
    if duration_s < 2 * margin + n * max_len + (n - 1) * min_gap:
        raise ValueError("minutes_per_file is too short to place the requested seizures")
    for _ in range(2000):
        starts = np.sort(rng.uniform(margin, duration_s - margin - max_len, size=n))
        lengths = rng.uniform(min_len, max_len, size=n)
        ends = starts + lengths
        if all(starts[i + 1] >= ends[i] + min_gap for i in range(n - 1)):
            return [(float(s), float(e)) for s, e in zip(starts, ends, strict=True)]
    raise RuntimeError("could not place non-overlapping seizures")


def _simulate_file(
    rng: np.random.Generator,
    n_ch: int,
    n_samples: int,
    fs: int,
    ar: float,
    gain: float,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    """AR(1) background (patient-specific colour and gain) plus ~3 Hz spike-wave-like seizures."""
    noise = rng.standard_normal((n_ch, n_samples))
    signal = lfilter([1.0], [1.0, -ar], noise, axis=-1) * gain
    sigma = gain / np.sqrt(1.0 - ar**2)
    for start_s, end_s in intervals:
        i0, i1 = int(round(start_s * fs)), int(round(end_s * fs))
        t = np.arange(i1 - i0) / fs
        phase = rng.uniform(0.0, 0.5, size=(n_ch, 1))
        rhythm = np.sin(2 * np.pi * 3.0 * t + phase) + 0.5 * np.sin(2 * np.pi * 6.0 * t + 2 * phase)
        envelope = np.minimum(1.0, np.minimum(t, t[-1] - t) / 1.0)  # 1 s on/off ramps
        chan_scale = rng.uniform(0.6, 1.0, size=(n_ch, 1))
        amplitude = sigma * rng.uniform(1.5, 3.0)
        signal[:, i0:i1] += amplitude * chan_scale * envelope * rhythm
    return signal.astype(np.float32)


def make_synthetic_project(
    root: Path,
    cfg: dict | None = None,
    n_patients: int = 4,
    files_per_patient: int = 3,
    minutes_per_file: float = 3.0,
    seed: int = 0,
) -> dict:
    """Write annotations.csv (C1), file_index.csv (C2) and preprocessed .npy files (C3).

    Patient 0 (``chb01``) has two cases (``chb01`` and ``chb21``); patients 1.. are ``chb02``...
    ``files_per_patient`` counts files *per case* (at least 2 so a seizure can be placed after
    the seizure-free calibration file). All output goes under ``root``.

    Returns a dict with ``root``, ``cfg`` (re-rooted), ``annotations_csv``, ``file_index_csv``,
    ``preprocessed`` (list of paths), ``patients`` and ``cases``.
    """
    if files_per_patient < 2:
        raise ValueError("files_per_patient must be >= 2")
    if n_patients < 1:
        raise ValueError("n_patients must be >= 1")
    cfg = make_test_cfg(Path(root), cfg)
    rng = np.random.default_rng(seed)
    fs = int(cfg["dataset"]["fs"])
    n_ch = len(cfg["dataset"]["channels"])
    duration_s = minutes_per_file * 60.0
    n_samples = int(round(duration_s * fs))

    patients = ["chb01"] + [f"chb{p + 1:02d}" for p in range(1, n_patients)]
    cases_of = {p: ([p, "chb21"] if p == "chb01" else [p]) for p in patients}

    ann_rows: list[dict] = []
    idx_rows: list[dict] = []
    written: list[Path] = []

    for patient in patients:
        ar = float(rng.uniform(0.6, 0.95))
        gain = float(rng.uniform(0.5, 2.0))
        files = [(case, i) for case in cases_of[patient] for i in range(1, files_per_patient + 1)]
        eligible = [f for f in files if f[1] >= 2]  # first file of each case stays seizure-free
        n_seizures_patient = int(rng.integers(1, 3))  # 1 or 2, guaranteed >= 1
        picks = rng.choice(len(eligible), size=n_seizures_patient, replace=True)
        per_file: dict[tuple[str, int], int] = {}
        for k in picks:
            per_file[eligible[int(k)]] = per_file.get(eligible[int(k)], 0) + 1

        for order, (case, i) in enumerate(files, start=1):
            stem = f"{case}_{i:02d}"
            file_name = f"{stem}.edf"
            intervals = _place_seizures(rng, per_file.get((case, i), 0), duration_s)
            signal = _simulate_file(rng, n_ch, n_samples, fs, ar, gain, intervals)
            out = ensure_parent(preprocessed_path(cfg, case, stem))
            np.save(out, signal)
            written.append(out)
            for s_idx, (start_s, end_s) in enumerate(intervals):
                ann_rows.append(
                    {
                        "patient": patient,
                        "case": case,
                        "file": file_name,
                        "seizure_idx": s_idx,
                        "seizure_start_s": start_s,
                        "seizure_end_s": end_s,
                    }
                )
            idx_rows.append(
                {
                    "patient": patient,
                    "case": case,
                    "file": file_name,
                    "file_order": order,
                    "duration_s": duration_s,
                    "n_samples": n_samples,
                    "n_seizures": len(intervals),
                    "role": "seizure" if intervals else "seizure_free",
                    "include": True,
                    "exclude_reason": "",
                }
            )

    ann_cols = ["patient", "case", "file", "seizure_idx", "seizure_start_s", "seizure_end_s"]
    ann_path = ensure_parent(annotations_csv(cfg))
    idx_path = ensure_parent(file_index_csv(cfg))
    pd.DataFrame(ann_rows, columns=ann_cols).to_csv(ann_path, index=False)
    pd.DataFrame(idx_rows).to_csv(idx_path, index=False)
    return {
        "root": Path(root),
        "cfg": cfg,
        "annotations_csv": ann_path,
        "file_index_csv": idx_path,
        "preprocessed": written,
        "patients": patients,
        "cases": [c for p in patients for c in cases_of[p]],
    }


# --------------------------------------------------------------------------------------
# Fake CHB-MIT raw tree with REAL EDF files (for loader / cohort / script-01 tests)
# --------------------------------------------------------------------------------------
def write_synthetic_edf(
    path: Path, labels: list[str], data_uv: np.ndarray, fs: int = 256
) -> None:
    """Write a valid EDF with the given channel ``labels`` and data in microvolts.

    Labels may repeat (like the duplicated ``T8-P8`` in real CHB-MIT files). The number of
    samples must be a whole number of seconds (EDF data records are 1 s long).
    """
    import edfio

    if data_uv.shape[0] != len(labels):
        raise ValueError("one data row per label is required")
    if data_uv.shape[1] % fs:
        raise ValueError("n_samples must be a multiple of fs")
    signals = [
        edfio.EdfSignal(row.astype(np.float64), fs, label=lab, physical_dimension="uV")
        for lab, row in zip(labels, data_uv, strict=True)
    ]
    ensure_parent(path)
    edfio.Edf(signals).write(path)


def _clock(seconds: int) -> str:
    seconds %= 86400
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def make_fake_chbmit_raw(
    root: Path, cfg: dict | None = None, seed: int = 0, duration_s: int = 60
) -> dict:
    """Create a miniature CHB-MIT tree (summaries, RECORDS lists, real EDFs) under ``root``.

    Cases and what they exercise:

    * ``chb01`` (3 files, seizure in 02) and ``chb21`` (2 files, seizure in 02): same patient;
    * ``chb02`` (3 files, seizure in 03): file 02 lacks channel ``P7-O1`` (must be excluded);
    * ``chb03`` (2 files, no seizure): patient without seizures (must be excluded);
    * every EDF has 18 required + 4 extra channels, a placeholder ``-`` channel and a second
      ``T8-P8`` (duplicate label), in shuffled order, in microvolts;
    * file ``chb03_02`` has a start/end time that wraps past midnight.

    Returns paths plus ``truth``: {edf_path: (18, n) array in canonical channel order, uV}.
    """
    cfg = make_test_cfg(Path(root), cfg)
    rng = np.random.default_rng(seed)
    fs = int(cfg["dataset"]["fs"])
    channels = list(cfg["dataset"]["channels"])
    raw_dir = Path(cfg["paths"]["raw"])
    n = duration_s * fs

    layout = {
        "chb01": {1: [], 2: [(20, 30)], 3: []},
        "chb21": {1: [], 2: [(25, 40)]},
        "chb02": {1: [], 2: [], 3: [(10, 22)]},
        "chb03": {1: [], 2: []},
    }
    missing_channel = {("chb02", 2): "P7-O1"}
    start_clock = {"chb03_02": 23 * 3600 + 59 * 60 + 30}  # wraps past midnight

    truth: dict[Path, np.ndarray] = {}
    records: list[str] = []
    records_seizure: list[str] = []
    for case, files in layout.items():
        summary = ["Data Sampling Rate: 256 Hz", "*" * 20, ""]
        for i, intervals in files.items():
            stem = f"{case}_{i:02d}"
            data = rng.normal(0.0, 30.0, size=(len(channels), n))
            labels = list(channels)
            block = data.copy()
            drop = missing_channel.get((case, i))
            if drop:
                keep = [k for k, lab in enumerate(labels) if lab != drop]
                labels = [labels[k] for k in keep]
                block = block[keep]
            extras = rng.normal(0.0, 30.0, size=(len(EXTRA_CHANNELS) + 2, n))
            all_labels = labels + EXTRA_CHANNELS + ["-", "T8-P8"]  # placeholder + duplicate
            all_data = np.vstack([block, extras])
            perm = rng.permutation(len(all_labels))
            # keep the ORIGINAL T8-P8 first so MNE names it T8-P8-0 (the one we compare against)
            first = all_labels.index("T8-P8")
            perm = [first] + [p for p in perm if p != first]
            order = list(rng.permutation(len(perm)))
            perm = [perm[k] for k in order]
            shuffled_labels = [all_labels[k] for k in perm]
            if shuffled_labels.index("T8-P8") != min(
                k for k, lab in enumerate(shuffled_labels) if lab == "T8-P8"
            ):
                raise AssertionError("unreachable")
            # Make sure the first occurrence of 'T8-P8' in the file is the canonical channel
            positions = [k for k, p in enumerate(perm) if all_labels[p] == "T8-P8"]
            if perm[positions[0]] != first:
                perm[positions[0]], perm[positions[1]] = perm[positions[1]], perm[positions[0]]
                shuffled_labels = [all_labels[k] for k in perm]
            edf_path = raw_dir / case / f"{stem}.edf"
            write_synthetic_edf(edf_path, shuffled_labels, all_data[perm], fs)
            if not drop:
                truth[edf_path] = data
            records.append(f"{case}/{stem}.edf")
            if intervals:
                records_seizure.append(f"{case}/{stem}.edf")
            t0 = start_clock.get(stem, 10 * 3600 + 3600 * (i - 1))
            summary += [
                f"File Name: {stem}.edf",
                f"File Start Time: {_clock(t0)}",
                f"File End Time: {_clock(t0 + duration_s)}",
                f"Number of Seizures in File: {len(intervals)}",
            ]
            for k, (a, b) in enumerate(intervals, start=1):
                tag = "Seizure" if len(intervals) == 1 else f"Seizure {k}"
                summary += [f"{tag} Start Time: {a} seconds", f"{tag} End Time: {b} seconds"]
            summary.append("")
        (raw_dir / case).mkdir(parents=True, exist_ok=True)
        (raw_dir / case / f"{case}-summary.txt").write_text("\n".join(summary))

    (raw_dir / "RECORDS").write_text("\n".join(records) + "\n")
    (raw_dir / "RECORDS-WITH-SEIZURES").write_text("\n".join(records_seizure) + "\n")
    return {
        "cfg": cfg,
        "raw_dir": raw_dir,
        "truth": truth,
        "records": records,
        "seizure_files": records_seizure,
        "duration_s": duration_s,
    }
