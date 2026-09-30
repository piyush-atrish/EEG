"""Synthetic Contract-C5 (feature table) generator for Member C's own tests.

Everything downstream in ``evaluation/`` and ``models/`` can be developed and tested against
this before Member B's real feature parquet files exist (see the project dependency graph:
C: Step 10 -> Step 11 -> Step 12, synthetic features until B delivers).

Feature names are built locally from ``cfg["features"]["catalog"]`` and
``cfg["dataset"]["channels"]`` (the exact ``f_{feature}_{channel_slug}`` naming rule from
Contract C5, README Section 7) -- this file never imports Member B's ``features.extract``.

Guarantees of :func:`make_synthetic_features`:

* one row per (patient, case, file, window); columns are the Contract C5 keys
  (``patient``, ``case``, ``file``, ``start_sample``, ``t_start_s``, ``label``) followed by
  float32 feature columns, in catalog order then channel order;
* every patient gets its own additive offset and multiplicative scale per feature
  (inter-subject variability, magnitude controlled by ``patient_shift``; 0 disables it);
* ictal windows have ``n_informative`` features shifted by ``effect`` (the classification
  "signal"; 0 disables it, so labels become unpredictable and AUC should sit near 0.5);
* the first 350 windows (chronological, i.e. sorted by (case, file, start_sample)) of every
  patient are interictal, so a 10-minute / 300-window calibration period is always available;
* patient ``chb01`` has two cases, ``chb01`` and ``chb21`` (or whichever pair
  ``cfg["dataset"]["patient_map"]`` specifies), to exercise the patient/case grouping guards
  the same way the real CHB-MIT dataset does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eegpipe.config import channel_slug

# Chronological windows guaranteed interictal before the first ictal window can appear, for
# every patient. Comfortably above the default calibration requirement (ceil(10 min * 60 / step_s)
# with step_s = window_s * (1 - overlap) = 2 s at the frozen defaults -> 300 windows).
_GUARANTEED_INTERICTAL = 350


def _feature_columns(cfg: dict) -> list[str]:
    """Build the Contract-C5 feature column names locally (catalog order, then channel order).

    Mirrors the naming rule in README Section 7 (C5) and Step 9 (``feature_names``), but is a
    self-contained implementation: Member C's tests must not depend on Member B's code.
    """
    catalog = cfg["features"]["catalog"]
    entropy_enabled = cfg["features"].get("entropy", {}).get("enabled", True)
    slugs = [channel_slug(ch) for ch in cfg["dataset"]["channels"]]

    names: list[str] = []
    for group, feats in catalog.items():
        if group == "nonlinear" and not entropy_enabled:
            continue
        for feat in feats:
            for slug in slugs:
                names.append(f"f_{feat}_{slug}")
    return names


def make_synthetic_features(
    cfg: dict,
    n_patients: int = 8,
    windows_per_patient: int = 600,
    ictal_fraction: float = 0.05,
    effect: float = 1.0,
    patient_shift: float = 1.0,
    n_informative: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a synthetic Contract-C5 DataFrame for evaluation and model tests.

    Parameters mirror README Section 11, Step 10. See the module docstring for the
    guarantees this generator makes (calibration prefix, patient/case grouping, tunable
    signal strength and inter-subject variability).
    """
    rng = np.random.default_rng(seed)

    fs = float(cfg["dataset"]["fs"])
    window_s = float(cfg["segmentation"]["window_s"])
    overlap = float(cfg["segmentation"]["overlap"])
    step_samples = round(window_s * (1 - overlap) * fs)

    feature_cols = _feature_columns(cfg)
    n_features = len(feature_cols)
    n_informative = min(n_informative, n_features)
    informative_idx = rng.choice(n_features, size=n_informative, replace=False)

    # Which patient gets a second case, and what it's called. Prefer the real config's
    # patient_map (chb21 -> chb01); fall back to the README's own example so the "one patient,
    # two cases" guarantee holds even if a test passes a stripped-down cfg.
    patient_map = cfg.get("dataset", {}).get("patient_map") or {}
    extra_case_of = {v: k for k, v in patient_map.items()}
    extra_case_of.setdefault("chb01", "chb21")

    patients = [f"chb{i:02d}" for i in range(1, n_patients + 1)]

    parts: list[pd.DataFrame] = []
    for patient in patients:
        cases = sorted({patient, extra_case_of[patient]} if patient in extra_case_of else {patient})

        total = windows_per_patient
        available_for_ictal = max(total - _GUARANTEED_INTERICTAL, 0)
        n_ictal = min(int(round(total * ictal_fraction)), available_for_ictal)

        label_seq = np.zeros(total, dtype=np.int8)
        if n_ictal > 0:
            ictal_positions = rng.choice(
                np.arange(_GUARANTEED_INTERICTAL, total), size=n_ictal, replace=False
            )
            label_seq[ictal_positions] = 1

        # Split the chronological window sequence across this patient's case(s); the first
        # (alphabetically earliest, i.e. earliest-recorded) case gets at least the
        # guaranteed-interictal prefix, so the split never cuts through it.
        if len(cases) == 1:
            split = total
        else:
            split = min(max(_GUARANTEED_INTERICTAL, int(round(total * 0.6))), total)

        # Per-patient inter-subject variability (spec: magnitude = patient_shift; 0 -> none).
        offset_p = rng.normal(0.0, patient_shift, size=n_features)
        scale_p = np.clip(1.0 + patient_shift * rng.normal(0.0, 0.3, size=n_features), 0.2, None)

        cursor = 0
        for case in cases:
            n_case_windows = split if case == cases[0] else total - split
            if n_case_windows <= 0:
                continue
            case_labels = label_seq[cursor : cursor + n_case_windows]
            cursor += n_case_windows

            start_samples = np.arange(n_case_windows, dtype=np.int64) * step_samples
            t_start_s = start_samples / fs

            x = rng.normal(0.0, 1.0, size=(n_case_windows, n_features))
            x = x * scale_p + offset_p
            ictal_rows = np.where(case_labels == 1)[0]
            if len(ictal_rows):
                x[np.ix_(ictal_rows, informative_idx)] += effect

            meta = pd.DataFrame(
                {
                    "patient": patient,
                    "case": case,
                    "file": f"{case}_01.edf",
                    "start_sample": start_samples,
                    "t_start_s": t_start_s,
                    "label": case_labels,
                }
            )
            feats = pd.DataFrame(x.astype(np.float32), columns=feature_cols)
            parts.append(pd.concat([meta, feats], axis=1))

    out = pd.concat(parts, ignore_index=True)
    key_cols = ["patient", "case", "file", "start_sample", "t_start_s", "label"]
    return out[key_cols + feature_cols]