"""Tests for eegpipe.features.extract (Member B, Step 9).

Uses a local synthetic fixture (not Member A's tests/fixtures/synth_signals.py,
which is A's file to own) built directly on top of Member B's own Step 6
(`build_window_table`) and Steps 7-8 feature functions, so this test
exercises the real integration across all of B's own modules end to end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eegpipe.features.extract import (
    extract_batch_features,
    extract_case_features,
    extract_window_features,
    feature_names,
)
from eegpipe.features.frequency_domain import frequency_features
from eegpipe.features.time_domain import time_domain_features
from eegpipe.segmentation.windows import build_window_table

FS = 256.0
WINDOW_S = 4.0
OVERLAP = 0.5
W = 1024
STEP = 512

CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]
assert len(CHANNELS) == 18

BASE_CFG = {
    "dataset": {"fs": FS, "channels": CHANNELS},
    "segmentation": {"window_s": WINDOW_S, "overlap": OVERLAP, "drop_boundary_windows": True},
    "features": {
        "welch": {"nperseg_s": 1.0, "noverlap_frac": 0.5},
        "bands": {
            "delta": [0.5, 4.0], "theta": [4.0, 8.0], "alpha": [8.0, 13.0],
            "beta": [13.0, 30.0], "gamma": [30.0, 45.0],
        },
        "entropy": {"enabled": True, "m": 2, "r_factor": 0.2, "decimate": 2},
        "wavelet": {"name": "db4", "level": 5},
        "catalog": {
            "time": ["mean", "std", "skew", "kurt", "line_length",
                     "hjorth_activity", "hjorth_mobility", "hjorth_complexity"],
            "frequency": ["logpow_delta", "logpow_theta", "logpow_alpha", "logpow_beta", "logpow_gamma",
                          "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma"],
            "nonlinear": ["sampen", "apen"],
            "wavelet": ["dwt_logE_d1", "dwt_logE_d2", "dwt_logE_d3", "dwt_logE_d4", "dwt_logE_d5", "dwt_logE_a5",
                        "dwt_std_d1", "dwt_std_d2", "dwt_std_d3", "dwt_std_d4", "dwt_std_d5", "dwt_std_a5"],
        },
    },
}


# ---------------------------------------------------------------------------
# feature_names
# ---------------------------------------------------------------------------


def test_feature_names_length_and_uniqueness():
    names = feature_names(BASE_CFG)
    assert len(names) == 18 * 32 == 576
    assert len(set(names)) == len(names)


def test_feature_names_omits_nonlinear_when_entropy_disabled():
    cfg = {**BASE_CFG, "features": {**BASE_CFG["features"], "entropy": {**BASE_CFG["features"]["entropy"], "enabled": False}}}
    names = feature_names(cfg)
    assert len(names) == 18 * 30 == 540
    assert not any("sampen" in n or "apen" in n for n in names)


def test_feature_names_catalog_outer_channel_inner_order():
    names = feature_names(BASE_CFG)
    # First 18 names: all channels of `mean`, in config channel order.
    assert names[:18] == [f"f_mean_{ch.upper().replace('-', '_')}" for ch in CHANNELS]
    # Next 18: all channels of `std`.
    assert names[18:36] == [f"f_std_{ch.upper().replace('-', '_')}" for ch in CHANNELS]
    # Last 18: all channels of `dwt_std_a5` (final catalog entry).
    assert names[-18:] == [f"f_dwt_std_a5_{ch.upper().replace('-', '_')}" for ch in CHANNELS]


# ---------------------------------------------------------------------------
# extract_window_features / extract_batch_features
# ---------------------------------------------------------------------------


def test_extract_window_features_shape_and_dtype():
    rng = np.random.default_rng(0)
    win = rng.normal(size=(18, W)) * 20.0
    vec = extract_window_features(win, FS, BASE_CFG)
    assert vec.shape == (576,)
    assert vec.dtype == np.float32


def test_extract_window_features_column_order_matches_vector_order():
    rng = np.random.default_rng(1)
    win = rng.normal(size=(18, W)) * 20.0
    vec = extract_window_features(win, FS, BASE_CFG)
    names = feature_names(BASE_CFG)

    time_feats = time_domain_features(win)
    freq_feats = frequency_features(win, FS, BASE_CFG)

    # Spot-check several indices across different catalog groups/channels
    # against the group functions computed independently.
    checks = [
        ("f_mean_" + CHANNELS[0].upper().replace("-", "_"), time_feats["mean"][0]),
        ("f_mean_" + CHANNELS[5].upper().replace("-", "_"), time_feats["mean"][5]),
        ("f_hjorth_complexity_" + CHANNELS[17].upper().replace("-", "_"), time_feats["hjorth_complexity"][17]),
        ("f_logpow_alpha_" + CHANNELS[2].upper().replace("-", "_"), freq_feats["logpow_alpha"][2]),
        ("f_rel_gamma_" + CHANNELS[9].upper().replace("-", "_"), freq_feats["rel_gamma"][9]),
    ]
    for col_name, expected in checks:
        idx = names.index(col_name)
        assert vec[idx] == pytest.approx(float(expected), rel=1e-4, abs=1e-5)


def test_extract_batch_features_shape():
    rng = np.random.default_rng(2)
    batch = rng.normal(size=(5, 18, W)) * 20.0
    mat = extract_batch_features(batch, FS, BASE_CFG)
    assert mat.shape == (5, 576)
    assert mat.dtype == np.float32


def test_extract_batch_equals_per_window():
    rng = np.random.default_rng(3)
    batch = rng.normal(size=(4, 18, W)) * 20.0
    batched = extract_batch_features(batch, FS, BASE_CFG)
    for i in range(4):
        single = extract_window_features(batch[i], FS, BASE_CFG)
        finite = np.isfinite(single)
        assert np.allclose(batched[i][finite], single[finite], atol=1e-4)
        assert np.array_equal(np.isnan(batched[i]), np.isnan(single))


# ---------------------------------------------------------------------------
# extract_case_features: local synthetic project (own Step 6 + 7 + 8 code)
# ---------------------------------------------------------------------------


def _make_signal(n_samples: int, seizure_start_s: float, seizure_end_s: float, seed: int) -> np.ndarray:
    """18-channel synthetic recording: white noise interictally, a large,
    higher-frequency, mostly regular rhythm ictally (mimics rhythmic ictal
    discharge). Chosen (and empirically checked) so BOTH the expected
    seizure signatures hold: line length rises (higher-frequency,
    higher-amplitude content -> bigger per-sample swings than noise) and
    sample entropy falls (a near-periodic signal is far more predictable
    than white noise, regardless of its frequency or amplitude).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / FS
    ictal_mask = (t >= seizure_start_s) & (t < seizure_end_s)

    noise = 5.0 * rng.normal(size=(18, n_samples))
    rhythm = 60.0 * np.sin(2 * np.pi * 14.0 * t)[None, :] * np.ones((18, 1))

    x = np.where(ictal_mask[None, :], rhythm + 0.3 * noise, noise)
    return x.astype(np.float32)


@pytest.fixture
def synthetic_project(tmp_path: Path):
    n_samples = W + 18 * STEP  # 19 possible windows, ~40 s
    seizure_start_s, seizure_end_s = 12.0, 20.0

    signal = _make_signal(n_samples, seizure_start_s, seizure_end_s, seed=42)

    cfg = {
        **BASE_CFG,
        "paths": {"preprocessed": str(tmp_path / "preprocessed")},
    }

    # Write via the same `preprocessed_path` helper that extract_case_features
    # reads through, so this fixture stays correct regardless of Member A's
    # actual on-disk layout convention (see test_windows.py's tiny_project
    # fixture, which applies the same principle for Step 6).
    from eegpipe.utils.paths import preprocessed_path

    path = preprocessed_path(cfg, "chb01", "chb01_01")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, signal)

    file_index = pd.DataFrame({
        "patient": ["chb01"],
        "case": ["chb01"],
        "file": ["chb01_01.edf"],
        "file_order": [0],
        "include": [True],
    })
    annotations = pd.DataFrame({
        "patient": ["chb01"],
        "case": ["chb01"],
        "file": ["chb01_01.edf"],
        "seizure_idx": [0],
        "seizure_start_s": [seizure_start_s],
        "seizure_end_s": [seizure_end_s],
    })

    window_table = build_window_table(file_index, annotations, cfg)
    return file_index, window_table, cfg


def test_extract_case_features_no_nan_inf_and_key_columns(synthetic_project):
    file_index, window_table, cfg = synthetic_project
    result = extract_case_features("chb01", file_index, window_table, cfg)

    for col in ["patient", "case", "file", "start_sample", "t_start_s", "label"]:
        assert col in result.columns

    feature_cols = feature_names(cfg)
    assert all(col in result.columns for col in feature_cols)
    assert len(result) == len(window_table)

    values = result[feature_cols].to_numpy()
    assert np.all(np.isfinite(values))
    assert (result["label"] == 1).sum() > 0  # sanity: the fixture does have ictal windows


def test_extract_case_features_ictal_windows_differ_from_interictal(synthetic_project):
    file_index, window_table, cfg = synthetic_project
    result = extract_case_features("chb01", file_index, window_table, cfg)

    ll_col = f"f_line_length_{'FP1_F7'}"
    se_col = f"f_sampen_{'FP1_F7'}"

    ictal_ll = result.loc[result["label"] == 1, ll_col]
    interictal_ll = result.loc[result["label"] == 0, ll_col]
    ictal_se = result.loc[result["label"] == 1, se_col]
    interictal_se = result.loc[result["label"] == 0, se_col]

    assert len(ictal_ll) > 0 and len(interictal_ll) > 0
    # Rhythmic, large-amplitude ictal segment -> larger line length.
    assert ictal_ll.mean() > interictal_ll.mean()
    # Rhythmic, more regular ictal segment -> lower sample entropy.
    assert ictal_se.mean() < interictal_se.mean()


def test_extract_case_features_empty_case_returns_typed_empty_frame(synthetic_project):
    file_index, window_table, cfg = synthetic_project
    result = extract_case_features("chb99", file_index, window_table, cfg)
    assert result.empty
    assert list(result.columns) == ["patient", "case", "file", "start_sample", "t_start_s", "label"] + feature_names(cfg)


def test_extract_case_features_respects_batch_size(synthetic_project):
    file_index, window_table, cfg = synthetic_project
    full = extract_case_features("chb01", file_index, window_table, cfg, batch_size=256)
    small_batches = extract_case_features("chb01", file_index, window_table, cfg, batch_size=3)
    pd.testing.assert_frame_equal(
        full.sort_values("start_sample").reset_index(drop=True),
        small_batches.sort_values("start_sample").reset_index(drop=True),
        check_exact=False,
        atol=1e-4,
    )


def test_extract_case_features_non_finite_replacement_and_warning(synthetic_project, caplog):
    file_index, window_table, cfg = synthetic_project
    # Force a constant window (std=0) into the recording so its entropy
    # (sampen/apen) comes out `nan` from Step 8 and must be replaced here.
    from eegpipe.utils.paths import preprocessed_path

    bad_signal = np.zeros((18, W + 18 * STEP), dtype=np.float32)
    np.save(preprocessed_path(cfg, "chb01", "chb01_01"), bad_signal)

    with caplog.at_level("INFO"):
        result = extract_case_features("chb01", file_index, window_table, cfg)

    feature_cols = feature_names(cfg)
    assert np.all(np.isfinite(result[feature_cols].to_numpy()))
    assert any("replaced" in message for message in caplog.messages)
