"""Third arm: LOSO on the features of an adaptive setting, with separate output names (chunk 3.3)."""

import copy

import pandas as pd
import pytest

from eegpipe.adaptive import adaptive_features_dir
from eegpipe.utils.paths import (
    features_path,
    file_index_csv,
    per_patient_table_path,
    predictions_path,
)
from tests.fixtures.synth_features import make_synthetic_features


@pytest.fixture()
def project(tmp_cfg):
    cfg = copy.deepcopy(tmp_cfg)
    df = make_synthetic_features(cfg, n_patients=5, windows_per_patient=600, effect=1.5,
                                 patient_shift=0.0, seed=0)
    out_dir = adaptive_features_dir(cfg, "testset")
    out_dir.mkdir(parents=True)
    for case, part in df.groupby("case"):
        part.to_parquet(out_dir / f"{case}.parquet", index=False)
    index = df.groupby(["patient", "case"], as_index=False).agg(file=("file", "first"))
    index["include"] = True
    file_index_csv(cfg).parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(file_index_csv(cfg), index=False)
    return cfg


def test_arm_label_naming(load_script):
    arm_label = load_script("05_run_baseline").arm_label
    assert arm_label("raw", None, None) == "raw"
    assert arm_label("raw", None, "dev5") == "raw_dev5"
    assert arm_label("raw", "rls_L8_lam0.9999", "dev5") == "adaptive_rls_L8_lam0.9999_dev5"
    assert arm_label("subject_standardised", "kalman_qr3e-05_blk1", None) == "adaptive_kalman_qr3e-05_blk1_standardised"
    for bad_tag in ("a b", "x__y", "dev/5"):
        with pytest.raises(ValueError):
            arm_label("raw", None, bad_tag)


def test_setting_reads_its_own_features_and_names_the_arms(project, tmp_path, write_cfg, load_script):
    cfg = project
    cfg_path = write_cfg(cfg, tmp_path / "cfg.yaml")
    script = load_script("05_run_baseline")
    code = script.main(["--config", str(cfg_path), "--setting", "testset", "--tag", "dev",
                        "--models", "svm", "--tuning", "fixed"])
    assert code == 0
    assert not features_path(cfg, "chb02").exists()  # the default feature folder is not used at all
    for label in ("adaptive_testset_dev", "adaptive_testset_standardised_dev"):
        table = pd.read_csv(per_patient_table_path(cfg, label, "svm"))
        assert len(table) == 5 and "auc" in table.columns
        preds = pd.read_parquet(predictions_path(cfg, label, "svm"))
        assert set(preds["arm"]) == {label}
    # nothing was written under the plain arm names
    assert not per_patient_table_path(cfg, "raw", "svm").exists()


def test_only_the_requested_patients_need_features(project, tmp_path, write_cfg, load_script):
    cfg = project
    (adaptive_features_dir(cfg, "testset") / "chb05.parquet").unlink()  # features missing for chb05
    cfg_path = write_cfg(cfg, tmp_path / "cfg.yaml")
    script = load_script("05_run_baseline")
    patients = ["--patients", "chb01", "chb02", "chb03", "chb04"]
    assert script.main(["--config", str(cfg_path), "--setting", "testset", "--models", "svm",
                        "--arms", "raw", "--tuning", "fixed", "--tag", "four", *patients]) == 0
    assert len(pd.read_csv(per_patient_table_path(cfg, "adaptive_testset_four", "svm"))) == 4
    with pytest.raises(FileNotFoundError, match="04_extract_features.py --setting testset"):
        script.main(["--config", str(cfg_path), "--setting", "testset", "--models", "svm",
                     "--arms", "raw", "--tuning", "fixed"])  # all patients: chb05 is missing


def test_unknown_setting_is_an_error_not_a_crash(project, tmp_path, write_cfg, load_script):
    cfg_path = write_cfg(project, tmp_path / "cfg.yaml")
    assert load_script("05_run_baseline").main(["--config", str(cfg_path), "--setting", "nope"]) == 1
