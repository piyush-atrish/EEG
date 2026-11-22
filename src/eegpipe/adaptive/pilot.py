"""Cheap pilot (stage A): screen adaptive settings on the development patients without a classifier.

For each setting the filtered arrays are turned into 15 cheap window features per channel (see
``preservation.FEATURE_NAMES``) on all ictal windows plus a seeded sample of interictal windows, and
two diagnostics from docs/adaptive_design.md are computed:

* **M1, subject invariance:** share of each feature's interictal variance explained by patient
  identity (eta-squared); the mean over features is used for the comparison (the median is also
  reported). Lower is better.
* **M2, contrast kept:** mean within-patient ``2 |AUC - 0.5|`` over features (seizure vs interictal
  windows). Higher is better; reported relative to the raw arrays.

and a third, informative one: **direction agreement**, the share of strong features whose direction
matches the other patients (see ``evaluation/feature_direction.py``).

A setting ``proceeds`` to the expensive stage B (full features + LOSO) if M1 improved on raw and M2 kept
at least ``m2_floor`` of the raw value. Only the development patients may be used here (frozen rule,
docs/decisions.md); the evaluation patients are scored once, with the frozen setting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eegpipe.adaptive.preservation import FEATURE_NAMES, window_features
from eegpipe.config import channel_slug
from eegpipe.evaluation.feature_direction import direction_summary, feature_auc_by_patient
from eegpipe.segmentation.windows import file_stem
from eegpipe.utils.paths import file_index_csv, windows_path

DEV_PATIENTS = ("chb01", "chb02", "chb03", "chb04", "chb05")
M2_FLOOR = 0.9


def missing_arrays(cfg: dict, root: str | Path, patients: list[str]) -> list[str]:
    """Included files of ``patients`` that have no array under ``root`` (as ``case/stem.npy``)."""
    index = pd.read_csv(file_index_csv(cfg))
    index = index[index["include"].astype(bool) & index["patient"].isin(patients)]
    return [f"{row.case}/{file_stem(row.file)}.npy" for row in index.itertuples()
            if not (Path(root) / row.case / f"{file_stem(row.file)}.npy").exists()]


def sample_windows(windows: pd.DataFrame, interictal_cap: int, seed: int) -> pd.DataFrame:
    """All ictal windows plus at most ``interictal_cap`` randomly chosen interictal ones (seeded)."""
    rng = np.random.default_rng(seed)
    ictal = windows[windows["label"] == 1]
    interictal = windows[windows["label"] == 0]
    if len(interictal) > interictal_cap:
        interictal = interictal.iloc[np.sort(rng.choice(len(interictal), interictal_cap, replace=False))]
    return pd.concat([ictal, interictal]).sort_values(["file", "start_sample"]).reset_index(drop=True)


def cheap_case_table(
    windows: pd.DataFrame,
    array_root: str | Path,
    cfg: dict,
    interictal_cap: int = 3000,
    seed: int = 0,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Cheap features of the sampled windows of ONE case, read from ``<array_root>/<case>/<stem>.npy``."""
    fs = float(cfg["dataset"]["fs"])
    win = int(round(cfg["segmentation"]["window_s"] * fs))
    slugs = [channel_slug(c) for c in cfg["dataset"]["channels"]]
    columns = [f"f_{name}_{slug}" for slug in slugs for name in FEATURE_NAMES]
    chosen = sample_windows(windows, interictal_cap, seed)
    frames = []
    for (case, file), group in chosen.groupby(["case", "file"], sort=False):
        arr = np.load(Path(array_root) / case / f"{file_stem(file)}.npy", mmap_mode="r")
        starts = group["start_sample"].to_numpy()
        for i in range(0, len(starts), batch_size):
            batch = np.stack([np.asarray(arr[:, s : s + win], dtype=np.float64) for s in starts[i : i + batch_size]])
            feats = np.concatenate([window_features(batch[:, c, :], fs) for c in range(batch.shape[1])], axis=1)
            meta = group.iloc[i : i + batch_size][["patient", "case", "file", "start_sample", "label"]]
            frames.append(pd.concat([meta.reset_index(drop=True),
                                     pd.DataFrame(feats.astype(np.float32), columns=columns)], axis=1))
    return pd.concat(frames, ignore_index=True)


def patient_variance_fraction(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Eta-squared per feature: between-patient sum of squares over total, interictal windows only."""
    g = df[df["label"] == 0]
    if g["patient"].nunique() < 2:
        raise ValueError("need interictal windows from at least two patients")
    grand = g[feature_cols].mean()
    means = g.groupby("patient")[feature_cols].mean()
    counts = g.groupby("patient").size()
    between = (counts.to_numpy()[:, None] * (means - grand) ** 2).sum()
    total = ((g[feature_cols] - grand) ** 2).sum()
    return (between / total.where(total > 0)).fillna(0.0)


def summarise_table(df: pd.DataFrame, min_ictal: int = 30) -> dict:
    """M1, M2 and direction agreement of one setting's feature table."""
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    eta = patient_variance_fraction(df, feature_cols)
    auc = feature_auc_by_patient(df, feature_cols, min_ictal=min_ictal)
    summary = direction_summary(auc) if auc.shape[1] >= 1 else pd.DataFrame()
    return {
        "n_patients": int(df["patient"].nunique()),
        "n_windows": len(df),
        "m1_median": float(eta.median()),
        "m1_mean": float(eta.mean()),
        "m2": float(summary["separability"].mean()) if len(summary) else float("nan"),
        "agreement": float(summary["agreement"].mean()) if len(summary) else float("nan"),
    }


def separability_by_feature_type(df: pd.DataFrame, min_ictal: int = 30) -> pd.Series:
    """Mean within-patient ``2 |AUC - 0.5|`` per feature type (``FEATURE_NAMES``), over channels and patients."""
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    auc = feature_auc_by_patient(df, feature_cols, min_ictal=min_ictal)
    sep = (2.0 * (auc - 0.5).abs()).mean(axis=1)  # per feature column, averaged over patients
    # longest names first, so "logpow_delta" is never mistaken for a shorter prefix
    ordered = sorted(FEATURE_NAMES, key=len, reverse=True)
    kind = {col: next(n for n in ordered if col.startswith(f"f_{n}_")) for col in feature_cols}
    return sep.groupby(pd.Series(kind)).mean().reindex(FEATURE_NAMES)


def run_stage_a(
    cfg: dict,
    settings: dict[str, str | Path],
    patients: list[str],
    interictal_cap: int = 3000,
    seed: int = 0,
    min_ictal: int = 30,
    m2_floor: float = M2_FLOOR,
    return_breakdown: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Summary table, one row per setting plus the raw arrays (``raw``) as reference.

    With ``return_breakdown=True`` also returns a table of separability per feature type (rows) and
    setting (columns), to see *which* kinds of features gain or lose seizure information.
    """
    index = pd.read_csv(file_index_csv(cfg))
    index = index[index["include"].astype(bool) & index["patient"].isin(patients)]
    cases = sorted(index["case"].unique())
    windows = {case: pd.read_parquet(windows_path(cfg, case)) for case in cases}
    roots = {"raw": Path(cfg["paths"]["preprocessed"]), **{k: Path(v) for k, v in settings.items()}}
    rows, by_type = [], {}
    for name, root in roots.items():
        missing = missing_arrays(cfg, root, patients)
        if missing:
            raise FileNotFoundError(
                f"setting '{name}' is missing {len(missing)} array(s) for {patients}, e.g. {missing[0]}; "
                "run scripts/02b_adaptive.py for those patients first"
            )
        table = pd.concat([cheap_case_table(windows[c], root, cfg, interictal_cap, seed) for c in cases],
                          ignore_index=True)
        rows.append({"setting": name, **summarise_table(table, min_ictal)})
        if return_breakdown:
            by_type[name] = separability_by_feature_type(table, min_ictal)
    out = pd.DataFrame(rows)
    ref = out.loc[out["setting"] == "raw"].iloc[0]
    out["m1_ratio"] = out["m1_mean"] / ref["m1_mean"]
    out["m2_ratio"] = out["m2"] / ref["m2"]
    out["agreement_delta"] = out["agreement"] - ref["agreement"]
    out["proceeds"] = (out["m1_ratio"] < 1.0) & (out["m2_ratio"] >= m2_floor) & (out["setting"] != "raw")
    if return_breakdown:
        return out, pd.DataFrame(by_type)
    return out
