"""Feature runner: assembles the four feature groups into Contract C5.

Member B, Step 9. `feature_names`/`extract_window_features`/
`extract_batch_features` are pure numeric functions (no I/O, no logging,
no non-finite handling) so they stay trivially testable and so
"batch equals per-window" is a meaningful check with no side effects to
account for. `extract_case_features` is the one function whose output IS
Contract C5 ("no NaN or inf values are allowed in a delivered file"), so
non-finite replacement, counting, and logging live there and nowhere else.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from eegpipe.config import channel_slug
from eegpipe.features.frequency_domain import frequency_features
from eegpipe.features.nonlinear import nonlinear_features
from eegpipe.features.time_domain import time_domain_features
from eegpipe.features.wavelet import wavelet_features
from eegpipe.segmentation.windows import file_stem
from eegpipe.utils.paths import preprocessed_path

logger = logging.getLogger(__name__)

_META_COLUMNS = ["patient", "case", "file", "start_sample", "t_start_s", "label"]
_NON_FINITE_WARN_FRACTION = 0.001  # 0.1 %


def _flat_catalog(cfg: dict) -> list[str]:
    """Catalog feature names, group order (`time, frequency, nonlinear, wavelet`) as in config.yaml.

    Omits the `nonlinear` group entirely when `features.entropy.enabled` is
    False, so a disabled-entropy run has 30 features/channel (540 total),
    not 32/576 with dead columns -- matching Section 6's "Derived facts"
    note that downstream code must infer feature columns, not hard-code
    576.
    """
    catalog = cfg["features"]["catalog"]
    entropy_enabled = cfg["features"]["entropy"].get("enabled", True)
    names: list[str] = []
    for group, feats in catalog.items():
        if group == "nonlinear" and not entropy_enabled:
            continue
        names.extend(feats)
    return names


def feature_names(cfg: dict) -> list[str]:
    """Ordered Contract C5 feature column names: `f_{feature}_{channel_slug}`.

    Catalog order (time, frequency, [nonlinear], wavelet) is the outer
    loop; `cfg["dataset"]["channels"]` order is the inner loop -- i.e. all
    18 channels of `mean` first, then all 18 of `std`, and so on.
    """
    channels = cfg["dataset"]["channels"]
    return [f"f_{feat}_{channel_slug(ch)}" for feat in _flat_catalog(cfg) for ch in channels]


def _extract(x: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Shared numeric core for both `extract_window_features` and `extract_batch_features`.

    `x` has shape `(..., n_ch, n_samples)`; the four group functions are
    each already fully vectorised over leading dimensions (Steps 7-8), so
    this calls each exactly once regardless of whether `x` is a single
    window or a batch.
    """
    all_feats: dict[str, np.ndarray] = {}
    all_feats.update(time_domain_features(x))
    all_feats.update(frequency_features(x, fs, cfg))
    all_feats.update(nonlinear_features(x, fs, cfg))  # {} if entropy disabled
    all_feats.update(wavelet_features(x, cfg))

    ordered = _flat_catalog(cfg)
    # Each all_feats[name] has shape (..., n_ch); stack along a new
    # second-to-last axis -> (..., n_features, n_ch), then merge the last
    # two axes. A row-major reshape iterates the LAST axis fastest, which
    # is exactly "channels inner loop, features outer loop".
    stacked = np.stack([all_feats[name] for name in ordered], axis=-2)
    flat_shape = stacked.shape[:-2] + (stacked.shape[-2] * stacked.shape[-1],)
    return stacked.reshape(flat_shape).astype(np.float32)


def extract_window_features(win: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Extract one window's feature vector.

    Parameters
    ----------
    win : np.ndarray
        Shape `(n_ch, n_samples)`.
    fs : float
        Sampling rate in Hz.
    cfg : dict
        Full config.

    Returns
    -------
    np.ndarray
        1-D float32 vector, ordered exactly as `feature_names(cfg)`.
    """
    return _extract(win, fs, cfg)


def extract_batch_features(batch: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Extract a batch of windows' feature vectors in one call.

    Parameters
    ----------
    batch : np.ndarray
        Shape `(n_win, n_ch, n_samples)`.
    fs : float
        Sampling rate in Hz.
    cfg : dict
        Full config.

    Returns
    -------
    np.ndarray
        Shape `(n_win, n_features)` float32, same column order as
        `feature_names(cfg)` / `extract_window_features`.
    """
    return _extract(batch, fs, cfg)


def extract_case_features(
    case: str,
    file_index: pd.DataFrame,
    window_table: pd.DataFrame,
    cfg: dict,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Build Contract C5 for one case: C4 columns plus every feature column.

    For each included file of `case` (in `file_order`, or file-name order
    if `file_order` is absent -- e.g. in a minimal test fixture), loads
    the preprocessed array once (`mmap_mode="r"`), slices its windows in
    batches of `batch_size`, extracts, and concatenates. Any non-finite
    feature value is replaced with `0.0`; replacements are counted per
    feature column and logged, with a warning if more than 0.1 % of all
    feature values in the case were replaced.

    Parameters
    ----------
    case : str
        Case id, e.g. `"chb01"`.
    file_index : pd.DataFrame
        Contract C2.
    window_table : pd.DataFrame
        Contract C4 for (at least) this case.
    cfg : dict
        Full config.
    batch_size : int, default 256
        Windows per `extract_batch_features` call.

    Returns
    -------
    pd.DataFrame
        Contract C5 for this case: `patient, case, file, start_sample,
        t_start_s, label` plus one float32 column per
        `feature_names(cfg)`, no NaN/inf.
    """
    fs = cfg["dataset"]["fs"]
    window_len = round(cfg["segmentation"]["window_s"] * fs)  # must match segmentation/windows.py
    ordered_columns = feature_names(cfg)

    case_files = file_index.loc[(file_index["case"] == case) & (file_index["include"].astype(bool))].copy()
    sort_col = "file_order" if "file_order" in case_files.columns else "file"
    case_files = case_files.sort_values(sort_col)

    case_windows = window_table.loc[window_table["case"] == case]

    frames: list[pd.DataFrame] = []
    non_finite_counts = np.zeros(len(ordered_columns), dtype=np.int64)
    total_values = 0

    for _, file_row in case_files.iterrows():
        file = file_row["file"]
        file_windows = case_windows.loc[case_windows["file"] == file].sort_values("start_sample")
        if file_windows.empty:
            continue

        file_stem_value = file_stem(file)
        arr = np.load(preprocessed_path(cfg, case, file_stem_value), mmap_mode="r")
        starts = file_windows["start_sample"].to_numpy()

        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            batch_windows = np.stack([np.asarray(arr[:, s : s + window_len]) for s in batch_starts], axis=0)
            feats = extract_batch_features(batch_windows, fs, cfg)

            non_finite_mask = ~np.isfinite(feats)
            if non_finite_mask.any():
                non_finite_counts += non_finite_mask.sum(axis=0)
                feats = np.where(non_finite_mask, 0.0, feats).astype(np.float32)
            total_values += feats.size

            batch_meta = (
                file_windows.iloc[batch_start : batch_start + batch_size][_META_COLUMNS]
                .reset_index(drop=True)
            )
            batch_feat_df = pd.DataFrame(feats, columns=ordered_columns)
            frames.append(pd.concat([batch_meta, batch_feat_df], axis=1))

    if not frames:
        return pd.DataFrame(columns=_META_COLUMNS + ordered_columns)

    result = pd.concat(frames, ignore_index=True)

    total_replaced = int(non_finite_counts.sum())
    if total_replaced > 0:
        for col, count in zip(ordered_columns, non_finite_counts):
            if count > 0:
                logger.info("case=%s: replaced %d non-finite value(s) in %r with 0.0", case, int(count), col)
        fraction = total_replaced / total_values if total_values else 0.0
        if fraction > _NON_FINITE_WARN_FRACTION:
            logger.warning(
                "case=%s: %.4f%% of feature values were non-finite and replaced with 0.0 (%d / %d)",
                case, fraction * 100, total_replaced, total_values,
            )

    return result
