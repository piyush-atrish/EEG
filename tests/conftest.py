"""Shared fixtures (joint file: change only by a PR that all three members approve)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from eegpipe.config import load_config
from tests.fixtures.synth_signals import make_fake_chbmit_raw, make_synthetic_project, make_test_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_cfg() -> dict:
    """The frozen project config (paths resolved against the repo root)."""
    return load_config()


@pytest.fixture()
def tmp_cfg(tmp_path) -> dict:
    """Frozen config with every path redirected into a temporary directory."""
    return make_test_cfg(tmp_path)


@pytest.fixture()
def synthetic_project(tmp_path) -> dict:
    """Synthetic project in contract layout (C1, C2, C3) under ``tmp_path``."""
    return make_synthetic_project(tmp_path, seed=0)


@pytest.fixture()
def fake_chbmit(tmp_path) -> dict:
    """Miniature CHB-MIT raw tree with real EDF files under ``tmp_path``."""
    return make_fake_chbmit_raw(tmp_path, seed=0)


@pytest.fixture()
def write_cfg():
    """Return a function that writes a config dict to a YAML file and returns its path."""

    def _write(cfg: dict, path: Path) -> Path:
        path.write_text(yaml.safe_dump(cfg))
        return path

    return _write


@pytest.fixture(scope="session")
def load_script():
    """Import ``scripts/<name>.py`` as a module (their names start with digits)."""

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return _load


def pytest_collection_modifyitems(config, items):
    """Skip ``needs_data`` tests unless real CHB-MIT files are present."""
    raw = REPO_ROOT / "data" / "raw" / "chbmit" / "RECORDS"
    if raw.exists():
        return
    skip = pytest.mark.skip(reason="real CHB-MIT data not present")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)
