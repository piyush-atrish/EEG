import yaml
import collections.abc
from pathlib import Path

def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parent.parent.parent

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_config(config_path: str | Path | None = None, overrides: dict = None) -> dict:
    root = find_repo_root()
    if config_path is None:
        config_path = root / "configs" / "config.yaml"
        
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
        
    if overrides:
        cfg = deep_update(cfg, overrides)
        
    if "dataset" in cfg and "channels" in cfg["dataset"]:
        channels = cfg["dataset"]["channels"]
        if len(set(channels)) != 18:
            raise ValueError(f"Config must specify exactly 18 unique channels, got {len(set(channels))}")
            
    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            cfg["paths"][key] = str((root / val).resolve())
            
    return cfg

def set_global_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

def channel_slug(ch_name: str) -> str:
    return ch_name.replace(" ", "_").replace("-", "_").upper()