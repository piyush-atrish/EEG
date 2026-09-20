"""End-to-end test of scripts/01_download_and_index.py on a fake CHB-MIT tree with real EDFs."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture()
def setup(fake_chbmit, tmp_path, write_cfg, load_script):
    cfg = fake_chbmit["cfg"]
    cfg["dataset"]["max_seizure_free_hours_per_patient"] = 0.02      # 72 s: cap binds on 60 s files
    cfg_path = write_cfg(cfg, tmp_path / "cfg.yaml")
    return cfg, cfg_path, load_script("01_download_and_index")


def _index(cfg):
    return pd.read_csv(Path(cfg["paths"]["interim"]) / "file_index.csv", keep_default_na=False)


def test_full_run_produces_contract_files(setup):
    cfg, cfg_path, script = setup
    assert script.main(["--config", str(cfg_path), "--skip-download"]) == 0
    idx = _index(cfg).set_index("file")
    ann = pd.read_csv(Path(cfg["paths"]["interim"]) / "annotations.csv")
    assert len(ann) == 3 and set(ann["patient"]) == {"chb01", "chb02"}
    assert idx.loc["chb02_02.edf", "exclude_reason"] == "missing_channel:P7-O1"
    assert idx.loc["chb03_01.edf", "exclude_reason"] == "no_seizure_patient"
    assert idx.loc["chb01_03.edf", "exclude_reason"] == "over_cap"
    for f in ("chb01_01.edf", "chb01_02.edf", "chb21_01.edf", "chb21_02.edf",
              "chb02_01.edf", "chb02_03.edf"):
        assert idx.loc[f, "include"]
    included = idx[idx["include"]]
    assert (included["n_samples"] == 15360).all()                     # exact, no off-by-one
    assert set(idx.loc[idx["case"] == "chb21", "patient"]) == {"chb01"}


def test_patient_subset_merges_instead_of_clobbering(setup):
    cfg, cfg_path, script = setup
    script.main(["--config", str(cfg_path), "--skip-download"])
    script.main(["--config", str(cfg_path), "--skip-download", "--patients", "chb02"])
    assert set(_index(cfg)["patient"]) == {"chb01", "chb02", "chb03"}   # others kept


def test_unknown_patient_is_a_hard_error(setup):
    cfg, cfg_path, script = setup
    assert script.main(["--config", str(cfg_path), "--skip-download", "--patients", "chb99"]) == 1


def test_missing_records_is_a_hard_error(setup):
    cfg, cfg_path, script = setup
    (Path(cfg["paths"]["raw"]) / "RECORDS").unlink()
    assert script.main(["--config", str(cfg_path), "--skip-download"]) == 1


def test_failed_download_is_excluded_and_reported(setup):
    """A selected file that is not on disk and cannot be downloaded must not stay included."""
    cfg, cfg_path, script = setup
    cfg["dataset"]["max_seizure_free_hours_per_patient"] = 2.0
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg))
    victim = Path(cfg["paths"]["raw"]) / "chb01" / "chb01_03.edf"
    victim.unlink()
    with patch.object(script, "download_metadata", return_value=[]), \
         patch.object(script, "download_edfs",
                      return_value={"ok": [], "failed": ["chb01/chb01_03.edf"]}):
        assert script.main(["--config", str(cfg_path)]) == 0
    idx = _index(cfg).set_index("file")
    assert not idx.loc["chb01_03.edf", "include"]
    assert idx.loc["chb01_03.edf", "exclude_reason"] == "download_failed"
    assert idx.loc["chb01_01.edf", "include"] and idx.loc["chb01_02.edf", "include"]


def test_patient_losing_all_seizure_files_returns_error(setup):
    cfg, cfg_path, script = setup
    # chb02_03 is chb02's only usable seizure file (chb02_02 lacks a channel)
    victim = Path(cfg["paths"]["raw"]) / "chb02" / "chb02_03.edf"
    victim.unlink()
    with patch.object(script, "download_metadata", return_value=[]), \
         patch.object(script, "download_edfs",
                      return_value={"ok": [], "failed": ["chb02/chb02_03.edf"]}):
        assert script.main(["--config", str(cfg_path)]) == 1
    idx = _index(cfg).set_index("file")
    assert not idx.loc["chb02_01.edf", "include"]
