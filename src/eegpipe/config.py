import copy
from pathlib import Path

import yaml


def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
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

def load_config(config_path: str | Path | None = None, overrides: dict = None) -> dict:
    root = find_repo_root()
    if config_path is None:
        config_path = root / "configs" / "config.yaml"

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    if "dataset" in cfg and "channels" in cfg["dataset"]:
        channels = cfg["dataset"]["channels"]
        if len(set(channels)) != 18:
            raise ValueError(f"Config must specify exactly 18 unique channels, got {len(set(channels))}")

    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            cfg["paths"][key] = str((root / val).resolve())

    return validate_config(cfg)

def set_global_seed(seed: int = 42):
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

def channel_slug(ch_name: str) -> str:
    return ch_name.replace(" ", "_").replace("-", "_").upper()

def validate_config(cfg: dict) -> dict:
    if "segmentation" in cfg:
        overlap = cfg["segmentation"].get("overlap", 0)
        if not (0 <= float(overlap) < 1):
            raise ValueError("Overlap must be >= 0 and < 1")
        if float(cfg["segmentation"].get("window_s", 1)) <= 0:
            raise ValueError("Window size must be positive")
    return cfg


def get_paths(cfg: dict):
    from pathlib import Path
    return {key: Path(val) for key, val in cfg["paths"].items()}
