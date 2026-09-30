"""Joint integration test (all three members extend it as their scripts land).

The stages run CHAINED inside one temporary project (03 needs C1-C3, 04 needs 03's windows,
05 needs 04's features, 06 needs 05's predictions). A stage that is still a stub
(``NotImplementedError``) ends the chain with a skip instead of a failure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


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


def test_full_chain_on_synthetic_project(synthetic_project, tmp_path, write_cfg, load_script):
    cfg = synthetic_project["cfg"]
    arg = ["--config", str(write_cfg(cfg, tmp_path / "cfg.yaml"))]

    # ---- Member B: contracts C4 and C5 ----
    assert load_script("03_make_windows").main(arg) in (0, None)
    windows = pd.concat(pd.read_parquet(p) for p in sorted(Path(cfg["paths"]["windows"]).glob("*.parquet")))
    assert list(windows.columns) == ["patient", "case", "file", "start_sample", "t_start_s", "label"]
    assert (windows["label"] == 1).sum() > 0 and set(windows["label"]) <= {0, 1}

    assert load_script("04_extract_features").main(arg + ["--n-jobs", "1"]) in (0, None)
    feats = pd.concat(pd.read_parquet(p) for p in sorted(Path(cfg["paths"]["features"]).glob("*.parquet")))
    cols = [c for c in feats.columns if c.startswith("f_")]
    assert len(cols) == 576 and np.isfinite(feats[cols].to_numpy()).all()
    assert len(feats) == len(windows) and set(feats["patient"]) == set(synthetic_project["patients"])

    # ---- Member C: contracts C6 and C7 ----
    assert load_script("05_run_baseline").main(
        arg + ["--tuning", "fixed", "--max-patients", "3"]
    ) in (0, None)

    arms = cfg["evaluation"]["arms"]
    models = cfg["evaluation"]["models"]
    baseline_patients = set(sorted(feats["patient"].unique())[:3])

    for arm in arms:
        for model in models:
            pred_path = Path(cfg["paths"]["predictions"]) / f"{arm}__{model}.parquet"
            assert pred_path.exists(), pred_path
            preds = pd.read_parquet(pred_path)
            assert list(preds.columns) == [
                "patient", "case", "file", "start_sample",
                "y_true", "y_score", "y_pred", "arm", "model",
            ]
            assert set(preds["patient"]) <= baseline_patients
            assert set(preds["y_true"]) <= {0, 1} and set(preds["y_pred"]) <= {0, 1}
            assert np.isfinite(preds["y_score"].to_numpy()).all()
            assert (preds["arm"] == arm).all() and (preds["model"] == model).all()

            table_path = Path(cfg["paths"]["tables"]) / f"per_patient_{arm}__{model}.csv"
            assert table_path.exists(), table_path
            per_patient = pd.read_csv(table_path)
            assert list(per_patient.columns) == [
                "patient", "n_windows", "n_ictal", "tp", "fp", "tn", "fn",
                "sensitivity", "specificity", "f1", "auc", "fa_per_hour",
            ]
            assert set(per_patient["patient"]) <= baseline_patients

    assert load_script("06_make_report").main(arg) in (0, None)

    summary_path = Path(cfg["paths"]["tables"]) / "summary_all.csv"
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert list(summary.columns) == ["arm", "model", "metric", "mean", "std", "median", "n_patients"]
    assert set(summary["arm"]) == set(arms) and set(summary["model"]) == set(models)

    figures = sorted(Path(cfg["paths"]["figures"]).glob("*.png"))
    assert len(figures) == len(models) * 3  # sensitivity, specificity, auc per model