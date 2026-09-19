import yaml
from pathlib import Path

def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        def merge_dicts(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = merge_dicts(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        cfg = merge_dicts(cfg, overrides)

    channels = cfg.get("dataset", {}).get("channels", [])
    if len(channels) != 18 or len(set(channels)) != 18:
        raise ValueError("Config must contain exactly 18 unique channels.")

    return cfg

def channel_slug(name: str) -> str:
    return name.upper().replace("-", "_")