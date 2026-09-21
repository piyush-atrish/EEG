"""Synthetic feature tables in Contract C5 format (Member C).

Used to develop and test the LOSO harness before real features exist. Column
names are built locally from the config catalog; Member B's code is NOT imported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eegpipe.config import channel_slug

# The first windows of every patient are guaranteed interictal so that a
# calibration segment exists (README Step 10).
N_LEADING_INTERICTAL = 350


def synthetic_feature_names(cfg: dict) -> list[str]:
    """``f_{feature}_{channel_slug}`` in catalog order (outer) and channel order (inner)."""
    cat = cfg["features"]["catalog"]
    feats = [f for group in ("time", "frequency", "nonlinear", "wavelet") for f in cat[group]]
    if not cfg["features"]["entropy"]["enabled"]:
        feats = [f for f in feats if f not in cat["nonlinear"]]
    slugs = [channel_slug(c) for c in cfg["dataset"]["channels"]]
    return [f"f_{f}_{s}" for f in feats for s in slugs]


def make_synthetic_features(cfg: dict, n_patients: int = 8, windows_per_patient: int = 600,
                            ictal_fraction: float = 0.05, effect: float = 1.0,
                            patient_shift: float = 1.0, n_informative: int = 20,
                            seed: int = 0) -> pd.DataFrame:
    """Build a synthetic Contract C5 DataFrame.

    Parameters
    ----------
    n_patients : int
        Patients ``chb01, chb02, ...``. Patient ``chb01`` consists of TWO cases
        (``chb01`` and ``chb21``) to test patient grouping.
    windows_per_patient : int
        Windows per patient in chronological order.
    ictal_fraction : float
        Approximate fraction of ictal windows, placed as two contiguous
        "seizures" after the leading interictal segment.
    effect : float
        Shift (in within-patient SD units) added to ``n_informative`` features
        during ictal windows. ``effect=0`` means no signal (AUC about 0.5).
    patient_shift : float
        Size of the per-patient additive offset and multiplicative scale of
        every feature (inter-subject variability). ``0`` means none.
    n_informative : int
        Number of features that carry the seizure effect.
    seed : int
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``patient, case, file, start_sample, t_start_s, label`` plus one
        float32 column per feature and channel.
    """
    rng = np.random.default_rng(seed)
    names = synthetic_feature_names(cfg)
    n_feat = len(names)
    fs = float(cfg["dataset"]["fs"])
    seg = cfg["segmentation"]
    step = int(round(round(seg["window_s"] * fs) * (1.0 - seg["overlap"])))
    informative = rng.choice(n_feat, size=min(n_informative, n_feat), replace=False)
    lead = min(N_LEADING_INTERICTAL, int(0.6 * windows_per_patient))
    windows_per_file = max(windows_per_patient // 3, 1)

    frames = []
    for p in range(n_patients):
        patient = f"chb{p + 1:02d}"
        n = windows_per_patient
        label = np.zeros(n, dtype=np.int8)
        n_ictal = max(int(round(ictal_fraction * n)), 2)
        n_ictal = min(n_ictal, n - lead)
        if n_ictal >= 2:  # two contiguous seizures, placed at random after the leading segment
            len_a, len_b = n_ictal // 2, n_ictal - n_ictal // 2
            free = n - lead - n_ictal - 2  # slack for a gap between the two seizures
            slack_a = int(rng.integers(0, max(free, 0) + 1))
            start_a = lead + slack_a
            start_b = start_a + len_a + 1 + int(rng.integers(0, max(free - slack_a, 0) + 1))
            label[start_a:start_a + len_a] = 1
            label[start_b:start_b + len_b] = 1

        offset = rng.normal(0.0, patient_shift, size=n_feat) if patient_shift > 0 else np.zeros(n_feat)
        scale = np.exp(rng.normal(0.0, 0.3 * patient_shift, size=n_feat)) if patient_shift > 0 \
            else np.ones(n_feat)
        x = rng.normal(0.0, 1.0, size=(n, n_feat))
        x[np.ix_(label == 1, informative)] += effect
        x = offset + scale * x

        idx = np.arange(n)
        # patient chb01 has two cases (chb01 for the first half, chb21 for the second half)
        case = np.where(idx < n // 2, "chb01", "chb21") if patient == "chb01" else np.full(n, patient)
        local = np.zeros(n, dtype=np.int64)  # window index within its own case (chronological)
        for c in np.unique(case):
            local[case == c] = np.arange((case == c).sum())
        file_no = local // windows_per_file
        within = local % windows_per_file
        file = np.array([f"{c}_{k + 1:02d}.edf" for c, k in zip(case, file_no)])
        start_sample = within * step

        meta = pd.DataFrame({
            "patient": patient, "case": case, "file": file,
            "start_sample": start_sample.astype(np.int64),
            "t_start_s": start_sample / fs, "label": label,
        })
        feats = pd.DataFrame(x.astype(np.float32), columns=names)
        frames.append(pd.concat([meta, feats], axis=1))
    return pd.concat(frames, ignore_index=True)


def make_test_config(tmp_path, tuning_mode: str = "fixed", overrides: dict | None = None) -> dict:
    """Config for tests: every output path inside ``tmp_path``, small and fast models.

    ``tuning_mode`` is ``"fixed"`` (fast) or ``"nested"``. Extra ``overrides`` are
    deep-merged on top (via ``load_config``).
    """
    from pathlib import Path

    from eegpipe.config import load_config

    tmp_path = Path(tmp_path)
    base = {
        "paths": {k: str(tmp_path / k) for k in
                  ("interim", "preprocessed", "windows", "features", "predictions",
                   "tables", "figures", "logs")},
        "evaluation": {
            "tuning": {"mode": tuning_mode},
            "fixed_params": {"rf": {"n_estimators": 50}},
            "grids": {"rf": {"n_estimators": [50], "max_depth": [None, 10], "min_samples_leaf": [1, 5]}},
        },
    }
    if overrides:
        from copy import deepcopy
        merged = deepcopy(base)
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        base = merged
    return load_config(overrides=base)
