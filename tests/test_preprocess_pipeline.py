import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from eegpipe.io.loader import ChannelMissingError  # noqa: F401  (documented failure type)
from eegpipe.preprocessing import pipeline as pl
from eegpipe.preprocessing.filters import apply_filters
from eegpipe.preprocessing.pipeline import (
    preprocess_all,
    preprocess_file,
    resolve_n_jobs,
    validate_output,
)
from eegpipe.utils.paths import preprocess_status_csv, preprocessed_path
from tests.fixtures.synth_signals import write_synthetic_edf


@pytest.fixture()
def env(fake_chbmit):
    """cfg + a file index covering the 9 fake EDFs that have all 18 channels."""
    cfg = fake_chbmit["cfg"]
    rows = [{"patient": "chb01" if p.parent.name in ("chb01", "chb21") else p.parent.name,
             "case": p.parent.name, "file": p.name, "n_samples": 60 * 256, "include": True}
            for p in sorted(fake_chbmit["truth"])]
    return cfg, pd.DataFrame(rows), fake_chbmit["truth"]


def _out(cfg, name):
    return preprocessed_path(cfg, name.split("_")[0], Path(name).stem)


def test_real_edf_run_matches_reference_filtering(env):
    cfg, index, truth = env
    res = preprocess_all(index, cfg, n_jobs=1)
    assert (res["status"] == "success").all() and len(res) == 9
    for edf, data in truth.items():
        y = np.load(_out(cfg, edf.name))
        assert y.dtype == np.float32 and y.shape == (18, 60 * 256)
        np.testing.assert_allclose(y, apply_filters(data, 256.0, cfg), atol=0.05)


def test_second_run_skips_valid_cache_and_log_is_appended(env):
    cfg, index, _ = env
    preprocess_all(index, cfg, n_jobs=1)
    second = preprocess_all(index, cfg, n_jobs=1)
    assert (second["status"] == "skipped_existing").all()
    log = pd.read_csv(preprocess_status_csv(cfg))
    assert len(log) == 18 and log["run_id"].nunique() == 2         # history kept, runs tagged
    assert {"elapsed_s", "flat_channels", "eeg_seconds", "error"} <= set(log.columns)


def test_truncated_cache_file_is_recomputed(env):
    cfg, index, _ = env
    preprocess_all(index, cfg, n_jobs=1)
    victim = _out(cfg, "chb01_01.edf")
    victim.write_bytes(victim.read_bytes()[:5000])                   # simulate a killed writer
    res = preprocess_all(index, cfg, n_jobs=1).set_index("file")
    assert res.loc["chb01_01.edf", "status"] == "recomputed_invalid_cache"
    assert np.load(victim).shape == (18, 60 * 256)


def test_stale_cache_with_wrong_length_is_recomputed(env):
    cfg, index, _ = env
    preprocess_all(index, cfg, n_jobs=1)
    np.save(_out(cfg, "chb01_01.edf"), np.zeros((18, 100), dtype=np.float32))
    res = preprocess_all(index, cfg, n_jobs=1).set_index("file")
    assert res.loc["chb01_01.edf", "status"] == "recomputed_invalid_cache"


def test_write_is_atomic_no_partial_file_on_crash(env):
    cfg, index, _ = env
    row = index.iloc[0]
    with patch.object(pl.os, "replace", side_effect=OSError("disk full")):
        res = preprocess_file(row, cfg)
    assert res["status"] == "failed" and "disk full" in res["error"]
    out = _out(cfg, row["file"])
    assert not out.exists() and not list(out.parent.glob("*.tmp"))


def test_validate_output_detects_problems():
    ok = np.zeros((18, 100), dtype=np.float32)
    assert validate_output(ok, 18, 100) == []
    assert "NaN" in validate_output(np.full((18, 100), np.nan, np.float32), 18, 100)[0]
    assert "bad shape" in validate_output(np.zeros((17, 100), np.float32), 18, 100)[0]
    msg = validate_output(ok, 18, 101)[0]
    assert "off by one" in msg and "re-run" in msg                    # stale-index hint


def test_stale_index_is_reported_as_failure_not_silently_accepted(env):
    cfg, index, _ = env
    index = index.copy()
    index["n_samples"] = 60 * 256 - 1                                 # the old off-by-one index
    res = preprocess_all(index, cfg, n_jobs=1)
    assert (res["status"] == "failed").all()
    assert "stale" in res["error"].iloc[0]


def test_one_bad_file_does_not_stop_the_batch(env):
    cfg, index, _ = env
    bad = Path(cfg["paths"]["raw"]) / "chb01" / "chb01_01.edf"
    bad.write_bytes(b"corrupt")
    res = preprocess_all(index, cfg, n_jobs=1).set_index("file")
    assert res.loc["chb01_01.edf", "status"] == "failed"
    assert (res.drop("chb01_01.edf")["status"] == "success").all()


def test_flat_channel_is_counted(tmp_cfg):
    cfg = tmp_cfg
    rng = np.random.default_rng(0)
    data = rng.normal(0, 30, size=(18, 5 * 256))
    data[4] = 0.0
    path = Path(cfg["paths"]["raw"]) / "chb01" / "chb01_01.edf"
    write_synthetic_edf(path, list(cfg["dataset"]["channels"]), data)
    row = {"case": "chb01", "file": "chb01_01.edf", "n_samples": 5 * 256}
    assert preprocess_file(row, cfg)["flat_channels"] == 1


def test_resolve_n_jobs_is_capped():
    with patch.object(os, "cpu_count", return_value=16):
        assert resolve_n_jobs(-1, {}) == 4
        assert resolve_n_jobs(None, {"project": {"max_workers": 2}}) == 2
        assert resolve_n_jobs(12, {}) == 4 and resolve_n_jobs(1, {}) == 1
    with patch.object(os, "cpu_count", return_value=2):
        assert resolve_n_jobs(-1, {}) == 2


@pytest.mark.slow
def test_parallel_process_pool_matches_sequential(env, tmp_path):
    cfg, index, _ = env
    with patch.object(os, "cpu_count", return_value=4):
        preprocess_all(index, cfg, n_jobs=2, backend="loky")
    par = {p.name: np.load(p) for p in Path(cfg["paths"]["preprocessed"]).rglob("*.npy")}
    assert len(par) == 9
    from tests.fixtures.synth_signals import make_test_cfg
    cfg2 = make_test_cfg(tmp_path / "seq")
    cfg2["paths"]["raw"] = cfg["paths"]["raw"]
    preprocess_all(index, cfg2, n_jobs=1)
    for p in Path(cfg2["paths"]["preprocessed"]).rglob("*.npy"):
        np.testing.assert_array_equal(par[p.name], np.load(p))


# ---------------------------------------------------------------- script 02
def test_script02_exit_codes(fake_chbmit, tmp_path, write_cfg, load_script):
    cfg = fake_chbmit["cfg"]
    cfg_path = write_cfg(cfg, tmp_path / "cfg.yaml")
    s1, s2 = load_script("01_download_and_index"), load_script("02_preprocess")
    assert s2.main(["--config", str(cfg_path)]) == 1                   # no index yet
    assert s1.main(["--config", str(cfg_path), "--skip-download"]) == 0
    assert s2.main(["--config", str(cfg_path), "--workers", "1"]) == 0
    assert s2.main(["--config", str(cfg_path), "--patients", "chb99"]) == 1
    bad = Path(cfg["paths"]["raw"]) / "chb02" / "chb02_01.edf"
    bad.write_bytes(b"corrupt")
    assert s2.main(["--config", str(cfg_path), "--overwrite", "--workers", "1"]) == 1
