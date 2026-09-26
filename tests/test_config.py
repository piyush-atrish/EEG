from pathlib import Path

import pytest
import yaml

from eegpipe.config import (
    channel_slug,
    find_repo_root,
    get_paths,
    load_config,
)


def test_default_config_is_the_frozen_contract(repo_cfg):
    """Every section Members B and C rely on is present (guards against a forked config)."""
    for section in ("project", "paths", "dataset", "preprocessing", "segmentation", "features",
                    "evaluation"):
        assert section in repo_cfg
    for key in ("raw", "interim", "preprocessed", "windows", "features", "predictions", "tables",
                "figures", "logs"):
        assert key in repo_cfg["paths"]
    assert "max_seizure_free_hours_per_patient" in repo_cfg["dataset"]
    assert len(repo_cfg["dataset"]["channels"]) == 18
    for key in ("arms", "models", "calibration", "train_sampling", "feature_selection",
                "tuning", "grids", "fixed_params"):
        assert key in repo_cfg["evaluation"], f"evaluation.{key} missing (Member C reads this)"
    assert set(repo_cfg["evaluation"]["arms"]) == {"raw", "subject_standardised"}
    assert set(repo_cfg["evaluation"]["models"]) == {"svm", "rf"}


def test_paths_resolved_against_repo_root(repo_cfg):
    root = find_repo_root()
    for value in repo_cfg["paths"].values():
        assert Path(value).is_absolute()
        assert Path(value).is_relative_to(root)


def test_absolute_override_paths_untouched(tmp_path):
    cfg = load_config(overrides={"paths": {"raw": str(tmp_path / "raw")}})
    assert cfg["paths"]["raw"] == str(tmp_path / "raw")


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_overrides_are_deep_merged():
    cfg = load_config(overrides={"preprocessing": {"fir": {"ripple_db": 40.0}}})
    fir = cfg["preprocessing"]["fir"]
    assert fir["ripple_db"] == 40.0
    assert fir["window"] == "kaiser" and fir["transition_hz"] == 1.0


def test_channel_validation():
    with pytest.raises(ValueError, match="exactly 18 unique channels"):
        load_config(overrides={"dataset": {"channels": ["FP1-F7"]}})
    dup = ["FP1-F7"] * 18
    with pytest.raises(ValueError, match="exactly 18 unique channels"):
        load_config(overrides={"dataset": {"channels": dup}})


def test_bad_segmentation_rejected():
    with pytest.raises(ValueError):
        load_config(overrides={"segmentation": {"overlap": 1.0}})


def test_channel_slug():
    assert channel_slug("FP1-F7") == "FP1_F7"
    assert channel_slug("fz-cz") == "FZ_CZ"


def test_get_paths_returns_path_objects(repo_cfg):
    paths = get_paths(repo_cfg)
    assert all(isinstance(p, Path) for p in paths.values())


def test_custom_config_file_is_used(tmp_path, repo_cfg):
    cfg = dict(repo_cfg)
    cfg["project"] = {**cfg["project"], "seed": 7}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert load_config(path)["project"]["seed"] == 7
