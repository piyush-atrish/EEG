"""Tests for the common adaptive entry point (adaptive/factory.py)."""

import numpy as np
import pytest
import yaml

from eegpipe.adaptive import (
    METHODS,
    adaptive_dir,
    apply_adaptive,
    build_filter,
    expand_pilot_grid,
    params_from_config,
    params_tag,
    resolve_params,
    warm_start_from_config,
)
from eegpipe.adaptive.kalman_gain import KalmanGainTracker
from eegpipe.adaptive.nlms import NLMSFilter
from eegpipe.adaptive.rls import RLSFilter

FS = 256


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return np.cumsum(rng.normal(0.0, 5.0, (3, 40 * FS)), axis=1) * 0.05 + rng.normal(0.0, 10.0, (3, 40 * FS))


def test_resolve_params_overlays_defaults_and_rejects_typos():
    assert resolve_params("nlms")["length"] == 16
    assert resolve_params("nlms", {"mu": 0.01})["mu"] == 0.01
    with pytest.raises(ValueError, match="unknown parameter"):
        resolve_params("rls", {"forgeting": 0.99})  # typo
    with pytest.raises(ValueError, match="unknown adaptive method"):
        resolve_params("lms")


def test_build_filter_returns_the_right_class():
    assert isinstance(build_filter("nlms", 2), NLMSFilter)
    assert isinstance(build_filter("rls", 2), RLSFilter)
    assert isinstance(build_filter("kalman", 2, fs=FS), KalmanGainTracker)


def test_apply_adaptive_matches_using_the_filter_directly(data):
    expected_filter = NLMSFilter(3, length=8, mu=0.01)
    expected_filter.warm_start(data[:, :FS])
    expected = expected_filter.process(data)
    y, _ = apply_adaptive(data, "nlms", {"length": 8, "mu": 0.01}, fs=FS, warm_start_s=1.0)
    assert np.array_equal(y, expected)


def test_every_method_returns_finite_float32_of_the_same_shape(data):
    for method in METHODS:
        y, _ = apply_adaptive(data, method, fs=FS, warm_start_s=2.0)
        assert y.shape == data.shape and y.dtype == np.float32 and np.isfinite(y).all()


def test_chaining_files_with_the_returned_state_equals_one_long_recording(data):
    split = 17 * FS + 123  # inside a block, inside a recording
    for method in METHODS:
        whole, _ = apply_adaptive(data, method, fs=FS, warm_start_s=1.0)
        first, state = apply_adaptive(data[:, :split], method, fs=FS, warm_start_s=1.0)
        second, _ = apply_adaptive(data[:, split:], method, fs=FS, state=state)
        assert np.array_equal(whole, np.concatenate([first, second], axis=1)), method


def test_state_from_another_method_or_parameter_set_is_rejected(data):
    _, state = apply_adaptive(data[:, : 5 * FS], "nlms", {"length": 8}, fs=FS, warm_start_s=1.0)
    with pytest.raises(ValueError, match="different method"):
        apply_adaptive(data, "rls", fs=FS, state=state)
    with pytest.raises(ValueError, match="different method"):
        apply_adaptive(data, "nlms", {"length": 16}, fs=FS, state=state)


def test_no_warm_start_when_disabled(data):
    cold, _ = apply_adaptive(data, "nlms", fs=FS, warm_start_s=0.0)
    expected = NLMSFilter(3).process(data)
    assert np.array_equal(cold, expected)


def test_params_tag_is_stable_and_distinguishes_settings():
    assert params_tag("nlms") == "nlms_L16_mu0.005"
    assert params_tag("rls") == "rls_L8_lam0.9999"
    assert params_tag("rls", {"forgetting": 0.9995}) == "rls_L8_lam0.9995"
    assert params_tag("kalman") == "kalman_qr0.0001_blk1"
    assert params_tag("kalman", {"q_over_r": 1e-3}) == "kalman_qr0.001_blk1"
    assert params_tag("nlms", {"mu": 0.01}) != params_tag("nlms", {"mu": 0.02})
    assert params_tag("rls", {"delta": 5.0}).endswith("_delta5")  # non-identifying but non-default


def test_adaptive_dir_sits_next_to_the_preprocessed_arrays(tmp_path):
    cfg = {"paths": {"interim": str(tmp_path)}}
    assert adaptive_dir(cfg, "nlms") == tmp_path / "preprocessed_adaptive" / "nlms_L16_mu0.005"


def test_a_non_default_warm_start_gets_its_own_folder(tmp_path):
    cfg = {"paths": {"interim": str(tmp_path)}}
    base = tmp_path / "preprocessed_adaptive"
    assert adaptive_dir(cfg, "nlms", warm_start_s=600) == base / "nlms_L16_mu0.005_ws600"
    assert adaptive_dir(cfg, "nlms", warm_start_s=60) == base / "nlms_L16_mu0.005"
    cfg_300 = {"paths": {"interim": str(tmp_path)}, "adaptive": {"warm_start_s": 300}}
    assert adaptive_dir(cfg_300, "nlms").name == "nlms_L16_mu0.005_ws300"


def test_returned_state_records_the_warm_start(data):
    _, state = apply_adaptive(data[:, : 5 * FS], "nlms", fs=FS, warm_start_s=1.0)
    assert state["warm_start_s"] == 1.0


BLOCK = """
adaptive:
  method: nlms
  nlms: {length: 8}
  pilot_grid:
    nlms: {length: [8, 16], mu: [0.001, 0.005, 0.02]}
    kalman: {q_over_r: [1.0e-4, 1.0e-3]}
"""


def test_config_block_selects_method_and_overrides_defaults():
    cfg = yaml.safe_load(BLOCK)
    method, params = params_from_config(cfg)
    assert method == "nlms" and params["length"] == 8 and params["mu"] == 0.005
    assert params_from_config(cfg, "rls")[0] == "rls"
    assert params_from_config({})[0] == "nlms"  # no adaptive block: documented default
    with pytest.raises(ValueError):
        params_from_config({"adaptive": {"method": "lms"}})
    with pytest.raises(ValueError, match="unknown parameter"):
        params_from_config({"adaptive": {"nlms": {"stepsize": 0.1}}})  # typo in another method's block
    with pytest.raises(ValueError, match="unknown key"):
        params_from_config({"adaptive": {"methd": "rls"}})  # typo in a top-level key


def test_warm_start_from_config():
    assert warm_start_from_config({}) == 60.0
    assert warm_start_from_config({"adaptive": {"warm_start_s": 30}}) == 30.0
    with pytest.raises(ValueError):
        warm_start_from_config({"adaptive": {"warm_start_s": -1}})


def test_pilot_grid_expands_to_the_cartesian_product():
    cfg = yaml.safe_load(BLOCK)
    grid = expand_pilot_grid(cfg, "nlms")
    assert len(grid) == 6 and {g["length"] for g in grid} == {8, 16}
    assert all("eps" in g for g in grid)  # fully resolved dicts
    assert len(expand_pilot_grid(cfg, "kalman")) == 2
    assert expand_pilot_grid(cfg, "rls") == []


def test_repo_config_adaptive_block_is_valid(repo_cfg):
    if "adaptive" not in repo_cfg:
        pytest.skip("adaptive block not added to configs/config.yaml yet")
    for method in METHODS:
        assert params_from_config(repo_cfg, method)[0] == method
        assert expand_pilot_grid(repo_cfg, method)
