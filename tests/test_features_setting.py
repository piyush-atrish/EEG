"""Features computed from an adaptive setting's arrays instead of the preprocessed cache (chunk 3.2)."""

import copy

import numpy as np
import pandas as pd
import pytest

from eegpipe.adaptive import adaptive_features_dir, adaptive_root
from eegpipe.features.extract import extract_case_features, extract_window_features, feature_names
from eegpipe.segmentation.windows import build_window_table
from eegpipe.utils.paths import features_path, file_index_csv, preprocessed_path, windows_path

FS, W, STEP = 256, 1024, 512
N_SAMPLES = W + 6 * STEP  # 7 windows, 16 s


@pytest.fixture()
def project(tmp_cfg):
    cfg = copy.deepcopy(tmp_cfg)
    cfg["features"]["entropy"]["enabled"] = False  # keeps the test fast; irrelevant to the routing
    rng = np.random.default_rng(0)
    raw = rng.normal(0.0, 10.0, (18, N_SAMPLES)).astype(np.float32)
    raw_path = preprocessed_path(cfg, "chb01", "chb01_01")
    raw_path.parent.mkdir(parents=True)
    np.save(raw_path, raw)

    alt_root = adaptive_root(cfg) / "testset"
    alt = (3.0 * raw).astype(np.float32)  # a "setting" whose arrays differ from the raw ones
    (alt_root / "chb01").mkdir(parents=True)
    np.save(alt_root / "chb01" / "chb01_01.npy", alt)

    file_index = pd.DataFrame({"patient": ["chb01"], "case": ["chb01"], "file": ["chb01_01.edf"],
                               "file_order": [0], "include": [True]})
    annotations = pd.DataFrame({"patient": ["chb01"], "case": ["chb01"], "file": ["chb01_01.edf"],
                                "seizure_idx": [0], "seizure_start_s": [7.0], "seizure_end_s": [14.0]})
    window_table = build_window_table(file_index, annotations, cfg)
    return {"cfg": cfg, "raw": raw, "alt": alt, "alt_root": alt_root,
            "file_index": file_index, "window_table": window_table}


def manual_features(array, window_table, cfg):
    rows = [extract_window_features(array[:, s : s + W], FS, cfg) for s in window_table["start_sample"]]
    return np.stack(rows)


def test_default_reads_the_preprocessed_cache(project):
    p = project
    result = extract_case_features("chb01", p["file_index"], p["window_table"], p["cfg"])
    expected = manual_features(p["raw"], p["window_table"], p["cfg"])
    assert np.allclose(result[feature_names(p["cfg"])].to_numpy(), expected, rtol=1e-4, atol=1e-4)


def test_array_dir_reads_the_other_arrays(project):
    p = project
    default = extract_case_features("chb01", p["file_index"], p["window_table"], p["cfg"])
    other = extract_case_features("chb01", p["file_index"], p["window_table"], p["cfg"],
                                  array_dir=p["alt_root"])
    expected = manual_features(p["alt"], p["window_table"], p["cfg"])
    assert np.allclose(other[feature_names(p["cfg"])].to_numpy(), expected, rtol=1e-4, atol=1e-4)
    ratio = other["f_std_FP1_F7"].to_numpy() / default["f_std_FP1_F7"].to_numpy()
    assert np.allclose(ratio, 3.0, rtol=1e-3)  # the amplitude difference proves which arrays were read
    # windows, labels and metadata do not depend on the array folder
    meta = ["patient", "case", "file", "start_sample", "t_start_s", "label"]
    pd.testing.assert_frame_equal(default[meta], other[meta])


def test_pointing_array_dir_at_the_default_cache_changes_nothing(project):
    p = project
    default = extract_case_features("chb01", p["file_index"], p["window_table"], p["cfg"])
    explicit = extract_case_features("chb01", p["file_index"], p["window_table"], p["cfg"],
                                     array_dir=p["cfg"]["paths"]["preprocessed"])
    pd.testing.assert_frame_equal(default, explicit)


def test_missing_arrays_in_array_dir_raise(project, tmp_path):
    p = project
    with pytest.raises(FileNotFoundError):
        extract_case_features("chb01", p["file_index"], p["window_table"], p["cfg"],
                              array_dir=tmp_path / "empty")


def test_script_setting_writes_to_the_setting_folder(project, tmp_path, write_cfg, load_script):
    p = project
    cfg = p["cfg"]
    file_index_csv(cfg).parent.mkdir(parents=True, exist_ok=True)
    p["file_index"].to_csv(file_index_csv(cfg), index=False)
    windows_path(cfg, "chb01").parent.mkdir(parents=True, exist_ok=True)
    p["window_table"].to_parquet(windows_path(cfg, "chb01"), index=False)
    cfg_path = write_cfg(cfg, tmp_path / "cfg.yaml")
    script = load_script("04_extract_features")

    assert script.main(["--config", str(cfg_path), "--n-jobs", "1", "--setting", "testset"]) == 0
    out = adaptive_features_dir(cfg, "testset") / "chb01.parquet"
    assert out.exists() and not features_path(cfg, "chb01").exists()
    got = pd.read_parquet(out)[feature_names(cfg)].to_numpy()
    assert np.allclose(got, manual_features(p["alt"], p["window_table"], cfg), rtol=1e-4, atol=1e-4)

    assert script.main(["--config", str(cfg_path), "--n-jobs", "1"]) == 0  # the default path still works
    assert features_path(cfg, "chb01").exists()
    assert script.main(["--config", str(cfg_path), "--n-jobs", "1", "--setting", "no_such_setting"]) == 1
