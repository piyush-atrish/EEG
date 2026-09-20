"""Joint integration test (all three members extend it as their scripts land).

Today it checks the contracts produced by Member A (C1-C3) on the synthetic project and then
tries stages 03-06; stages that are still stubs (NotImplementedError) are skipped, not failed.
"""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration

STAGES = ["03_make_windows", "04_extract_features", "05_run_baseline", "06_make_report"]


def test_contracts_c1_c2_c3(synthetic_project):
    cfg = synthetic_project["cfg"]
    ann = pd.read_csv(synthetic_project["annotations_csv"])
    idx = pd.read_csv(synthetic_project["file_index_csv"])
    assert list(ann.columns) == ["patient", "case", "file", "seizure_idx", "seizure_start_s",
                                 "seizure_end_s"]
    assert set(idx["patient"]) == set(synthetic_project["patients"])
    for row in idx.itertuples():
        stem = row.file.removesuffix(".edf")
        arr = np.load(f"{cfg['paths']['preprocessed']}/{row.case}/{stem}.npy", mmap_mode="r")
        assert arr.shape == (18, row.n_samples) and arr.dtype == np.float32


@pytest.mark.parametrize("stage", STAGES)
def test_downstream_stages_when_implemented(stage, synthetic_project, tmp_path, write_cfg,
                                            load_script):
    cfg_path = write_cfg(synthetic_project["cfg"], tmp_path / "cfg.yaml")
    module = load_script(stage)
    try:
        rc = module.main(["--config", str(cfg_path)])
    except NotImplementedError:
        pytest.skip(f"{stage} is still a stub")
    assert rc in (0, None)
