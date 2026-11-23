"""Command-line tools: label-shuffle on an adaptive setting's features, event report arm filter."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eegpipe.adaptive import adaptive_features_dir
from eegpipe.utils.paths import file_index_csv
from tests.fixtures.synth_features import make_synthetic_features

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_shuffle_reads_an_adaptive_setting(tmp_cfg, tmp_path, write_cfg):
    cfg = copy.deepcopy(tmp_cfg)
    cfg["evaluation"]["tuning"]["mode"] = "fixed"
    df = make_synthetic_features(cfg, n_patients=4, windows_per_patient=600, effect=1.5,
                                 patient_shift=0.0, seed=0)
    out_dir = adaptive_features_dir(cfg, "testset")
    out_dir.mkdir(parents=True)
    for case, part in df.groupby("case"):
        part.to_parquet(out_dir / f"{case}.parquet", index=False)
    index = df.groupby(["patient", "case"], as_index=False).agg(file=("file", "first"))
    index["include"] = True
    file_index_csv(cfg).parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(file_index_csv(cfg), index=False)
    cfg_path = write_cfg(cfg, tmp_path / "cfg.yaml")

    # an older log without the "setting" column must survive (the real results/tables has one)
    log_path = Path(cfg["paths"]["tables"]) / "label_shuffle.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "timestamp,model,seed,n_test_patients,n_patients_available,tuning_mode,auc,within_band\n"
        "2026-11-14T23:52:34,svm,0,3,5,nested,0.398,True\n"
    )
    tool = load_tool("check_label_shuffle")
    code = tool.main(["--config", str(cfg_path), "--models", "svm", "--seeds", "0",
                      "--n-test", "2", "--setting", "testset"])
    assert code in (0, 1)  # 0 = within the chance band; the point here is that it ran on the setting
    log = pd.read_csv(log_path)
    assert len(log) == 2 and log["auc"].iloc[0] == 0.398  # the old row is intact
    assert pd.isna(log["setting"].iloc[0]) and log["setting"].iloc[-1] == "testset"
    assert log["n_patients_available"].iloc[-1] == 4


def test_event_report_arm_filter(tmp_cfg, tmp_path, write_cfg, capsys):
    cfg = copy.deepcopy(tmp_cfg)
    pred_dir = Path(cfg["paths"]["predictions"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    starts = np.arange(40) * 512
    for arm in ("arm_a", "arm_b"):
        frame = pd.DataFrame({"patient": "chb01", "case": "chb01", "file": "chb01_01.edf",
                              "start_sample": starts, "y_true": [0] * 20 + [1] * 5 + [0] * 15,
                              "y_pred": [0] * 20 + [1] * 5 + [0] * 15, "arm": arm, "model": "svm"})
        frame.to_parquet(pred_dir / f"{arm}__svm.parquet", index=False)
    cfg_path = write_cfg(cfg, tmp_path / "cfg.yaml")
    tool = load_tool("event_report")
    assert tool.main(["--config", str(cfg_path), "--arms", "arm_a"]) == 0
    out = capsys.readouterr().out
    assert "arm_a" in out and "arm_b" not in out
    with pytest.raises(SystemExit, match="None of"):
        tool.main(["--config", str(cfg_path), "--arms", "nope"])


def test_compare_arms_can_exclude_patients(tmp_cfg, tmp_path, write_cfg):
    cfg = copy.deepcopy(tmp_cfg)
    tables = Path(cfg["paths"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)
    patients = [f"chb{i:02d}" for i in range(1, 6)]
    base = np.linspace(0.6, 0.9, 5)
    for arm, shift in (("ref", 0.0), ("other", 0.1)):
        pd.DataFrame({"patient": patients, "auc": base + shift, "sensitivity": 0.5,
                      "specificity": 0.9, "fa_per_hour": 10.0}).to_csv(
            tables / f"per_patient_{arm}__rf.csv", index=False)
    cfg_path = write_cfg(cfg, tmp_path / "cfg.yaml")
    tool = load_tool("compare_arms")
    assert tool.main(["--config", str(cfg_path), "--reference", "ref", "--exclude", "chb03"]) == 0
    result = pd.read_csv(tables / "arm_comparison.csv")
    assert set(result.loc[result["subset"] == "all", "n_patients"]) == {4}
