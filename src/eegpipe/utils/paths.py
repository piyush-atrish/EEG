from pathlib import Path


def annotations_csv(cfg: dict) -> Path:
    return Path(cfg["paths"]["interim"]) / "annotations.csv"

def file_index_csv(cfg: dict) -> Path:
    return Path(cfg["paths"]["interim"]) / "file_index.csv"

def preprocessed_path(cfg: dict, case: str, file_stem: str) -> Path:
    return Path(cfg["paths"]["preprocessed"]) / case / f"{file_stem}.npy"

def windows_path(cfg: dict, case: str) -> Path:
    return Path(cfg["paths"]["windows"]) / f"{case}.parquet"

def features_path(cfg: dict, case: str) -> Path:
    return Path(cfg["paths"]["features"]) / f"{case}.parquet"

def predictions_path(cfg: dict, arm: str, model: str) -> Path:
    return Path(cfg["paths"]["predictions"]) / f"{arm}__{model}.parquet"

def ensure_parent(path: Path | str) -> Path:
    """Ensure the parent directory of the given path exists, and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
