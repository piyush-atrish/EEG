"""
PLACEHOLDER — owned by Member A (Step 1). Not part of Member B's deliverable.

This is a minimal, spec-conformant stand-in so Member B's code (segmentation,
features) can be developed and tested against the real `load_config` /
`channel_slug` signatures before A's actual Step 1 PR lands. It should be
deleted wholesale and replaced by A's implementation, not merged as-is.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load configs/config.yaml, deep-merge `overrides`, and validate the channel list.

    Parameters
    ----------
    path : str | Path | None
        Path to the YAML config. Defaults to `configs/config.yaml` at the repo root.
    overrides : dict | None
        Deep-merged on top of the loaded config. Tests use this to point paths
        at temp directories without touching the frozen file on disk.

    Returns
    -------
    dict
        The fully-resolved configuration.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    channels = cfg["dataset"]["channels"]
    if len(channels) != 18:
        raise ValueError(f"Expected 18 channels, got {len(channels)}")
    if len(set(channels)) != len(channels):
        raise ValueError("Channel list contains duplicates")

    return cfg


def _deep_merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def channel_slug(name: str) -> str:
    """Canonical column-name fragment for a channel, e.g. 'FP1-F7' -> 'FP1_F7'."""
    return name.upper().replace("-", "_")
