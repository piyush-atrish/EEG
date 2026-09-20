from pathlib import Path

def ensure_parent(path: Path | str) -> Path:
    """Ensure the parent directory of the given path exists, and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def annotations_csv(cfg: dict) -> Path:
    return Path(cfg["paths"]["interim"]) / "annotations.csv"

def file_index_csv(cfg: dict) -> Path:
    return Path(cfg["paths"]["interim"]) / "file_index.csv"

def preprocessed_path(cfg: dict, case: str, file_stem: str) -> Path:
    return Path(cfg["paths"]["preprocessed"]) / case / f"{file_stem}.npy"

def preprocess_status_csv(cfg: dict) -> Path:
    return Path(cfg["paths"]["logs"]) / "preprocess_status.csv"