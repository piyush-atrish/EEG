import yaml
from pathlib import Path

def find_repo_root() -> Path:
    """Find the root directory of the repository."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str | Path | None = None, overrides: dict = None) -> dict:
    root = find_repo_root()
    if config_path is None:
        config_path = root / "config.yaml"
        
    cfg = {}
    # Gracefully handle missing config.yaml during testing
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
            
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
                
    # VALIDATION: Enforce the 18-channel requirement
    if "dataset" in cfg and "channels" in cfg["dataset"]:
        channels = cfg["dataset"]["channels"]
        if len(set(channels)) != 18:
            raise ValueError(f"Config must specify exactly 18 unique channels, got {len(set(channels))}")
                
    # Force all paths to be absolute, relative to the repository root
    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            cfg["paths"][key] = str((root / val).resolve())
            
    return cfg

def set_global_seed(seed: int = 42):
    """Lock all RNG engines for perfect reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

def channel_slug(ch_name: str) -> str:
    """Convert a channel name to a filesystem-safe slug."""
    return ch_name.replace(" ", "_").replace("-", "_")