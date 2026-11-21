"""Tests for adaptive/runner.py and scripts/02b_adaptive.py on synthetic preprocessed arrays."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eegpipe.adaptive import apply_adaptive, resolve_params, warm_start_from_config
from eegpipe.adaptive.factory import adaptive_dir
from eegpipe.adaptive.runner import file_order, run_adaptive, status_csv
from eegpipe.utils.paths import ensure_parent, preprocessed_path

FS, N_CH, SECONDS = 256, 18, 20
METHOD, OVERRIDES = "nlms", {"length": 4}
NAMES = ["chb01_10.edf", "chb01_9.edf", "chb01_2.edf"]  # numeric order 2, 9, 10 (not alphabetical)
ORDERED = ["chb01_2.edf", "chb01_9.edf", "chb01_10.edf"]


def make_inputs(cfg, case, names, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {}
    for name in names:
        x = rng.normal(0.0, 20.0, (N_CH, SECONDS * FS)).astype(np.float32)
        path = ensure_parent(preprocessed_path(cfg, case, name.removesuffix(".edf")))
        np.save(path, x)
        arrays[name] = x
    return arrays


def make_index(rows):
    return pd.DataFrame(
        [{"patient": case[:5], "case": case, "file": name, "n_samples": SECONDS * FS, "include": True}
         for case, name in rows]
    )


def out_file(cfg, case, name):
    params = resolve_params(METHOD, OVERRIDES)
    return adaptive_dir(cfg, METHOD, params) / case / f"{name.removesuffix('.edf')}.npy"


def expected_chain(cfg, arrays, ordered):
    params = resolve_params(METHOD, OVERRIDES)
    state, outputs = None, {}
    for name in ordered:
        y, state = apply_adaptive(arrays[name], METHOD, params, fs=FS,
                                  warm_start_s=warm_start_from_config(cfg), state=state)
        outputs[name] = y
    return outputs


def run(cfg, index, **kwargs):
    return run_adaptive(index, cfg, method=METHOD, overrides=OVERRIDES, n_jobs=1, **kwargs)


def statuses(result):
    return dict(zip(result["file"], result["status"]))


def test_file_order_is_numeric_not_alphabetical():
    assert sorted(NAMES, key=file_order) == ORDERED
    assert file_order("chb02_16+.edf")[0] == 16


def test_files_are_chained_in_chronological_order(tmp_cfg):
    arrays = make_inputs(tmp_cfg, "chb01", NAMES)
    result = run(tmp_cfg, make_index([("chb01", n) for n in NAMES]))
    assert set(result["status"]) == {"success"}
    for name, expected in expected_chain(tmp_cfg, arrays, ORDERED).items():
        assert np.array_equal(np.load(out_file(tmp_cfg, "chb01", name)), expected)


def test_cases_are_independent_state_streams(tmp_cfg):
    a = make_inputs(tmp_cfg, "chb01", ORDERED, seed=1)
    b = make_inputs(tmp_cfg, "chb21", ORDERED, seed=2)
    index = make_index([("chb01", n) for n in ORDERED] + [("chb21", n) for n in ORDERED])
    run(tmp_cfg, index)
    for case, arrays in (("chb01", a), ("chb21", b)):
        for name, expected in expected_chain(tmp_cfg, arrays, ORDERED).items():
            assert np.array_equal(np.load(out_file(tmp_cfg, case, name)), expected)


def test_rerun_skips_finished_files_and_resumes_from_the_saved_state(tmp_cfg):
    make_inputs(tmp_cfg, "chb01", NAMES)
    index = make_index([("chb01", n) for n in NAMES])
    run(tmp_cfg, index)
    first_run = {n: np.load(out_file(tmp_cfg, "chb01", n)) for n in ORDERED}
    out_file(tmp_cfg, "chb01", ORDERED[-1]).unlink()  # an interrupted run: last file missing
    second = run(tmp_cfg, index)
    assert statuses(second) == {ORDERED[0]: "skipped_existing", ORDERED[1]: "skipped_existing",
                                ORDERED[2]: "success"}
    assert np.array_equal(np.load(out_file(tmp_cfg, "chb01", ORDERED[-1])), first_run[ORDERED[-1]])


def test_recomputing_a_file_forces_everything_after_it(tmp_cfg):
    make_inputs(tmp_cfg, "chb01", NAMES)
    index = make_index([("chb01", n) for n in NAMES])
    run(tmp_cfg, index)
    first_run = {n: np.load(out_file(tmp_cfg, "chb01", n)) for n in ORDERED}
    out_file(tmp_cfg, "chb01", ORDERED[1]).unlink()  # middle file missing
    second = run(tmp_cfg, index)
    assert statuses(second) == {ORDERED[0]: "skipped_existing", ORDERED[1]: "success",
                                ORDERED[2]: "recomputed_downstream"}
    for n in ORDERED:
        assert np.array_equal(np.load(out_file(tmp_cfg, "chb01", n)), first_run[n])


def test_state_made_with_other_settings_is_not_trusted(tmp_cfg):
    make_inputs(tmp_cfg, "chb01", NAMES)
    index = make_index([("chb01", n) for n in NAMES])
    run(tmp_cfg, index)
    state_file = out_file(tmp_cfg, "chb01", ORDERED[0]).with_suffix(".state.pkl")
    state = pickle.loads(state_file.read_bytes())
    state["params"]["mu"] = 0.5  # as if produced by a different setting
    state_file.write_bytes(pickle.dumps(state))
    second = run(tmp_cfg, index)
    assert statuses(second)[ORDERED[0]] == "success"  # recomputed, not skipped
    assert statuses(second)[ORDERED[1]] == "recomputed_downstream"


def test_state_made_with_another_warm_start_is_not_trusted(tmp_cfg):
    make_inputs(tmp_cfg, "chb01", NAMES)
    index = make_index([("chb01", n) for n in NAMES])
    run(tmp_cfg, index)
    state_file = out_file(tmp_cfg, "chb01", ORDERED[0]).with_suffix(".state.pkl")
    state = pickle.loads(state_file.read_bytes())
    state["warm_start_s"] = 600.0
    state_file.write_bytes(pickle.dumps(state))
    assert statuses(run(tmp_cfg, index))[ORDERED[0]] == "success"


def test_different_warm_starts_use_separate_folders_and_give_different_output(tmp_cfg):
    arrays = make_inputs(tmp_cfg, "chb01", ORDERED)
    index = make_index([("chb01", n) for n in ORDERED])
    run(tmp_cfg, index)  # default warm start (60 s: the whole 20 s file)
    short = run(tmp_cfg, index, warm_start_s=1.0)
    assert set(short["status"]) == {"success"} and short["setting"].iloc[0].endswith("_ws1")
    params = resolve_params(METHOD, OVERRIDES)
    default_out = np.load(out_file(tmp_cfg, "chb01", ORDERED[0]))
    short_out = np.load(adaptive_dir(tmp_cfg, METHOD, params, 1.0) / "chb01" / "chb01_2.npy")
    assert not np.array_equal(default_out, short_out)
    expected = expected_chain(tmp_cfg, arrays, ORDERED)  # the default-folder result is untouched
    assert np.array_equal(default_out, expected[ORDERED[0]])


def test_overwrite_recomputes_everything(tmp_cfg):
    make_inputs(tmp_cfg, "chb01", NAMES)
    index = make_index([("chb01", n) for n in NAMES])
    run(tmp_cfg, index)
    assert set(run(tmp_cfg, index, overwrite=True)["status"]) == {"success"}


def test_a_failed_file_stops_its_case_but_not_other_cases(tmp_cfg):
    arrays = make_inputs(tmp_cfg, "chb01", NAMES, seed=1)
    other = make_inputs(tmp_cfg, "chb21", ORDERED, seed=2)
    broken = arrays[ORDERED[1]].copy()
    broken[3, 100] = np.nan
    np.save(preprocessed_path(tmp_cfg, "chb01", ORDERED[1].removesuffix(".edf")), broken)
    index = make_index([("chb01", n) for n in NAMES] + [("chb21", n) for n in ORDERED])
    result = run(tmp_cfg, index)
    by_case = {c: statuses(g) for c, g in result.groupby("case")}
    assert by_case["chb01"] == {ORDERED[0]: "success", ORDERED[1]: "failed", ORDERED[2]: "not_run"}
    assert set(by_case["chb21"].values()) == {"success"}
    assert "NaN" in result.loc[result["status"] == "failed", "error"].iloc[0]
    assert not out_file(tmp_cfg, "chb01", ORDERED[1]).exists()  # no output for the broken file
    for name, expected in expected_chain(tmp_cfg, other, ORDERED).items():
        assert np.array_equal(np.load(out_file(tmp_cfg, "chb21", name)), expected)


def test_missing_input_is_reported_as_a_failure(tmp_cfg):
    make_inputs(tmp_cfg, "chb01", ORDERED[:1])
    index = make_index([("chb01", n) for n in ORDERED[:2]])  # the second file has no array
    result = run(tmp_cfg, index)
    assert statuses(result) == {ORDERED[0]: "success", ORDERED[1]: "failed"}
    assert "FileNotFoundError" in result.loc[result["status"] == "failed", "error"].iloc[0]


def test_excluded_files_and_other_patients_are_ignored(tmp_cfg):
    make_inputs(tmp_cfg, "chb01", ORDERED)
    index = make_index([("chb01", n) for n in ORDERED])
    index.loc[0, "include"] = False
    assert len(run(tmp_cfg, index)) == 2
    assert len(run(tmp_cfg, index, patients=["chb99"])) == 0


def test_status_log_is_appended_across_runs(tmp_cfg):
    make_inputs(tmp_cfg, "chb01", ORDERED)
    index = make_index([("chb01", n) for n in ORDERED])
    run(tmp_cfg, index)
    run(tmp_cfg, index)
    log = pd.read_csv(status_csv(tmp_cfg))
    assert len(log) == 6 and log["run_id"].nunique() == 2
    assert set(log["setting"]) == {"nlms_L4_mu0.005"}
    assert {"rms_in", "rms_out", "elapsed_s", "status"} <= set(log.columns)


def test_script_helpers_and_exit_codes(tmp_cfg, tmp_path, write_cfg, load_script):
    script = load_script("02b_adaptive")
    assert script.parse_overrides(["mu=0.01", "length=8", "tag=abc"]) == {"mu": 0.01, "length": 8, "tag": "abc"}
    with pytest.raises(ValueError):
        script.parse_overrides(["mu"])
    cfg_path = write_cfg(tmp_cfg, tmp_path / "cfg.yaml")
    assert script.main(["--config", str(cfg_path), "--method", "nlms", "--param", "bogus=1"]) == 1  # typo
    assert script.main(["--config", str(cfg_path), "--method", "nlms"]) == 1  # no file index yet
    make_inputs(tmp_cfg, "chb01", ORDERED)
    index = make_index([("chb01", n) for n in ORDERED])
    ensure_parent(tmp_cfg["paths"]["interim"] + "/file_index.csv")
    index.to_csv(tmp_cfg["paths"]["interim"] + "/file_index.csv", index=False)
    assert script.main(["--config", str(cfg_path), "--method", "nlms", "--param", "length=4",
                        "--workers", "1"]) == 0
    assert script.main(["--config", str(cfg_path), "--method", "nlms", "--patients", "chb99"]) == 1
    assert script.main(["--config", str(cfg_path), "--method", "nlms", "--param", "length=4",
                        "--workers", "1", "--warm-start-s", "1"]) == 0
    assert any(p.name.endswith("_ws1") for p in (Path(tmp_cfg["paths"]["interim"]) / "preprocessed_adaptive").iterdir())
