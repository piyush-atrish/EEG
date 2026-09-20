"""Configuration loading. ``configs/config.yaml`` is the single source of truth.

All relative entries under ``paths`` are resolved against the repository root, so scripts
behave identically no matter which directory they are launched from. Absolute paths (for
example temporary directories used by tests) are left untouched.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

N_CHANNELS = 18


def find_repo_root(start: str | Path | None = None) -> Path:
    """Locate the repository root (the directory holding ``pyproject.toml`` and ``configs/``).

    Resolution order: the ``EEGPIPE_ROOT`` environment variable, then a walk upwards from
    ``start`` (default: this file), then the current working directory.
    """
    env = os.environ.get("EEGPIPE_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(start).resolve() if start else Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "configs").is_dir():
            return parent
    return Path.cwd().resolve()


def _deep_merge(base: dict, updates: dict) -> dict:
    """Return a new dict: ``updates`` merged recursively into ``base`` (lists are replaced)."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _validate(cfg: dict) -> None:
    channels = cfg.get("dataset", {}).get("channels", [])
    if len(channels) != N_CHANNELS or len(set(channels)) != N_CHANNELS:
        raise ValueError(f"Config must contain exactly {N_CHANNELS} unique channels.")
    seg = cfg.get("segmentation", {})
    if not seg.get("window_s", 1) > 0:
        raise ValueError("segmentation.window_s must be positive.")
    if not 0 <= seg.get("overlap", 0) < 1:
        raise ValueError("segmentation.overlap must be in [0, 1).")
    if not cfg.get("dataset", {}).get("fs", 1) > 0:
        raise ValueError("dataset.fs must be positive.")
    if "paths" not in cfg:
        raise ValueError("Config is missing the 'paths' section.")


def _resolve(root: Path, value: str | Path) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else root / p)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the YAML config, apply ``overrides`` (deep merge), validate, resolve paths.

    Parameters
    ----------
    path : str or Path, optional
        Config file. Defaults to ``<repo root>/configs/config.yaml``.
    overrides : dict, optional
        Nested dict deep-merged over the file contents (tests use it to redirect ``paths``).
    """
    root = find_repo_root(path)
    cfg_path = Path(path) if path else root / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    _validate(cfg)
    cfg["paths"] = {key: _resolve(root, val) for key, val in cfg["paths"].items()}
    return cfg


def channel_slug(name: str) -> str:
    """Canonical channel token used in feature column names, e.g. ``FP1-F7`` -> ``FP1_F7``."""
    return name.upper().replace("-", "_")


def get_paths(cfg: dict) -> dict[str, Path]:
    """Return every configured path as a ``Path`` object."""
    return {key: Path(val) for key, val in cfg["paths"].items()}