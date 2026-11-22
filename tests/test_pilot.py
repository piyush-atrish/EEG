"""Tests for adaptive/pilot.py (stage A) and tools/pilot_stage_a.py on a synthetic 3-patient project."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eegpipe.adaptive.pilot import (
    missing_arrays,
    patient_variance_fraction,
    run_stage_a,
    sample_windows,
)
from eegpipe.adaptive.preservation import FEATURE_NAMES
from eegpipe.segmentation.windows import build_window_table
from eegpipe.utils.paths import file_index_csv, preprocessed_path, windows_path

FS, SECONDS = 256, 240
GAINS = {"chb01": 1.0, "chb02": 6.0, "chb03": 0.25}  # patient-specific amplitude scale


def test_patient_variance_fraction_known_values():
    rng = np.random.default_rng(0)
    rows = []
    for patient, shift in (("p1", 0.0), ("p2", 10.0)):
        n = 400
        rows.append(pd.DataFrame({"patient": patient, "label": 0, "f_shifted": rng.normal(shift, 1, n),
                                  "f_same": rng.normal(0, 1, n)}))
    rows.append(pd.DataFrame({"patient": "p1", "label": 1, "f_shifted": 1e6, "f_same": 1e6}, index=[0]))
    eta = patient_variance_fraction(pd.concat(rows, ignore_index=True), ["f_shifted", "f_same"])
    assert eta["f_shifted"] > 0.95 and eta["f_same"] < 0.05  # ictal rows are ignored
    with pytest.raises(ValueError):
        patient_variance_fraction(rows[0], ["f_same"])


def test_sample_windows_keeps_all_ictal_and_caps_interictal_reproducibly():
    windows = pd.DataFrame({"patient": "p", "case": "c", "file": "f.edf",
                            "start_sample": np.arange(500) * 512, "label": [1] * 20 + [0] * 480})
    a = sample_windows(windows, interictal_cap=50, seed=3)
    b = sample_windows(windows, interictal_cap=50, seed=3)
    assert a.equals(b) and (a["label"] == 1).sum() == 20 and (a["label"] == 0).sum() == 50
    assert not a.equals(sample_windows(windows, interictal_cap=50, seed=4))


@pytest.fixture()
def project(tmp_cfg):
    cfg = copy.deepcopy(tmp_cfg)
    rng = np.random.default_rng(1)
    n = FS * SECONDS
    t = np.arange(n) / FS
    burst = ((t >= 100) & (t < 160)) * np.sin(2 * np.pi * 10 * t)
    root = Path(cfg["paths"]["interim"]) / "preprocessed_adaptive"
    rows, notes = [], []
    for case, gain in GAINS.items():
        raw = (gain * (10.0 * rng.normal(size=(18, n)) + 40.0 * burst[None, :])).astype(np.float32)
        raw_path = preprocessed_path(cfg, case, f"{case}_01")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_path, raw)
        for name, arr in (("norm", raw / gain), ("scrambled", rng.normal(size=raw.shape))):
            (root / name / case).mkdir(parents=True, exist_ok=True)
            np.save(root / name / case / f"{case}_01.npy", arr.astype(np.float32))
        rows.append({"patient": case, "case": case, "file": f"{case}_01.edf", "file_order": 0, "include": True})
        notes.append({"patient": case, "case": case, "file": f"{case}_01.edf", "seizure_idx": 0,
                      "seizure_start_s": 100.0, "seizure_end_s": 160.0})
    file_index, annotations = pd.DataFrame(rows), pd.DataFrame(notes)
    file_index_csv(cfg).parent.mkdir(parents=True, exist_ok=True)
    file_index.to_csv(file_index_csv(cfg), index=False)
    table = build_window_table(file_index, annotations, cfg)
    for case in GAINS:
        windows_path(cfg, case).parent.mkdir(parents=True, exist_ok=True)
        table[table["case"] == case].to_parquet(windows_path(cfg, case), index=False)
    return cfg, root


def test_stage_a_flags_a_good_setting_and_rejects_a_destructive_one(project):
    cfg, root = project
    result = run_stage_a(cfg, {"norm": root / "norm", "scrambled": root / "scrambled"},
                         patients=list(GAINS), interictal_cap=60, seed=0, min_ictal=5)
    by = result.set_index("setting")
    assert by.loc["raw", "m1_ratio"] == pytest.approx(1.0) and not by.loc["raw", "proceeds"]
    # removing each patient's amplitude scale removes the patient effect but keeps the seizure contrast
    assert by.loc["norm", "m1_ratio"] < 0.2
    assert by.loc["norm", "m2_ratio"] > 0.95 and by.loc["norm", "proceeds"]
    # arrays without the burst: the patient effect is gone too, but so is the seizure contrast
    assert by.loc["scrambled", "m2_ratio"] < 0.6 and not by.loc["scrambled", "proceeds"]
    assert {"m1_median", "m1_mean", "m2", "agreement", "agreement_delta", "n_windows"} <= set(result.columns)


def test_tool_refuses_evaluation_patients():
    spec = importlib.util.spec_from_file_location(
        "pilot_stage_a", Path(__file__).resolve().parents[1] / "tools" / "pilot_stage_a.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    with pytest.raises(SystemExit, match="evaluation patients"):
        tool.main(["--patients", "chb01", "chb06"])


def test_an_incomplete_setting_is_reported_clearly(project):
    cfg, root = project
    (root / "norm" / "chb02" / "chb02_01.npy").unlink()  # as if 02b had not been run for chb02
    assert missing_arrays(cfg, root / "norm", list(GAINS)) == ["chb02/chb02_01.npy"]
    assert missing_arrays(cfg, root / "scrambled", list(GAINS)) == []
    with pytest.raises(FileNotFoundError, match="missing 1 array"):
        run_stage_a(cfg, {"norm": root / "norm"}, patients=list(GAINS), interictal_cap=30, min_ictal=5)


def test_tool_skips_incomplete_settings_by_default_but_not_when_named(project, tmp_path, write_cfg, capsys):
    cfg, root = project
    (root / "norm" / "chb02" / "chb02_01.npy").unlink()
    cfg_path = write_cfg(cfg, tmp_path / "cfg.yaml")
    spec = importlib.util.spec_from_file_location(
        "pilot_stage_a", Path(__file__).resolve().parents[1] / "tools" / "pilot_stage_a.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    common = ["--config", str(cfg_path), "--patients", "chb01", "chb02", "chb03", "--cap", "30",
              "--allow-non-dev"]
    with pytest.raises(SystemExit, match="missing 1 array"):
        tool.main([*common, "--settings", "norm"])
    assert tool.main(common) == 0  # default: 'norm' is skipped with a message, 'scrambled' runs
    out = capsys.readouterr().out
    assert "Skipping 'norm'" in out and "scrambled" in out


def test_breakdown_by_feature_type(project):
    cfg, root = project
    result, breakdown = run_stage_a(
        cfg, {"norm": root / "norm"}, patients=list(GAINS), interictal_cap=60, seed=0,
        min_ictal=5, return_breakdown=True,
    )
    assert list(breakdown.index) == FEATURE_NAMES and list(breakdown.columns) == ["raw", "norm"]
    assert breakdown.notna().all().all() and (breakdown >= 0).all().all() and (breakdown <= 1).all().all()
    # per-patient amplitude scaling cannot change within-patient separability of the same features
    assert np.allclose(breakdown["raw"], breakdown["norm"], atol=0.05)
    assert len(result) == 2
