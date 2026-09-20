"""Preprocessing runner: EDF -> filtered float32 ``.npy`` cache (contract C3).

Robustness features:

* **atomic writes** (temp file + ``os.replace``): a killed process can never leave a truncated
  cache file that later runs mistake for a valid one;
* **integrity check of existing files** before skipping them (header shape/dtype must match);
* **strict validation** of every output: finite values, shape, and length equal to the
  ``n_samples`` recorded in the file index;
* **worker cap** so that parallel runs on long recordings cannot exhaust memory;
* **run-tagged status log** appended across runs (never overwritten).
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from eegpipe.io.loader import load_edf_channels
from eegpipe.preprocessing.filters import apply_filters
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import ensure_parent, preprocess_status_csv, preprocessed_path

logger = get_logger(__name__)

DEFAULT_MAX_WORKERS = 4
IMPLAUSIBLE_STD_UV = 5000.0


def resolve_n_jobs(n_jobs: int | None, cfg: dict) -> int:
    """Number of workers: never above ``project.max_workers`` (default 4) or the CPU count."""
    cap = int(cfg.get("project", {}).get("max_workers", DEFAULT_MAX_WORKERS))
    cpus = os.cpu_count() or 1
    if n_jobs is None or n_jobs <= 0:
        return max(1, min(cap, cpus))
    return max(1, min(int(n_jobs), cap, cpus))


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Write ``array`` to ``path`` atomically (via a temp file in the same directory)."""
    ensure_parent(path)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "wb") as handle:  # a file handle stops numpy appending ".npy"
            np.save(handle, array)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _existing_ok(path: Path, n_ch: int, expected_n: int) -> bool:
    """Is ``path`` a complete cache file? (memory-mapped header check, no data read)."""
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return False
    if arr.dtype != np.float32 or arr.ndim != 2 or arr.shape[0] != n_ch:
        return False
    return not (expected_n > 0 and arr.shape[1] != expected_n)


def validate_output(x: np.ndarray, n_ch: int, expected_n: int) -> list[str]:
    """Return a list of problems (empty if the array is a valid output)."""
    problems = []
    if x.ndim != 2 or x.shape[0] != n_ch:
        problems.append(f"bad shape {x.shape}, expected ({n_ch}, n_samples)")
        return problems
    if not np.isfinite(x).all():
        problems.append("output contains NaN or Inf")
    if expected_n > 0 and x.shape[1] != expected_n:
        hint = ""
        if abs(x.shape[1] - expected_n) == 1:
            hint = " (off by one: the file index is stale; re-run scripts/01_download_and_index.py)"
        problems.append(f"length mismatch: got {x.shape[1]}, index says {expected_n}{hint}")
    return problems


def preprocess_file(row, cfg: dict, overwrite: bool = False) -> dict:
    """Preprocess one file. ``row`` is a file-index row (Series or dict). Never raises."""
    case, file_name = row["case"], row["file"]
    file_stem = Path(file_name).stem
    out_path = preprocessed_path(cfg, case, file_stem)
    channels = cfg["dataset"]["channels"]
    fs_cfg = int(cfg["dataset"]["fs"])
    expected = row.get("n_samples", 0)
    expected_n = int(expected) if pd.notna(expected) else 0

    result = {
        "file": file_name,
        "case": case,
        "status": "success",
        "n_samples": 0,
        "eeg_seconds": 0.0,
        "elapsed_s": 0.0,
        "flat_channels": 0,
        "error": "",
    }
    started = time.perf_counter()
    try:
        if out_path.exists() and not overwrite:
            if _existing_ok(out_path, len(channels), expected_n):
                result["status"] = "skipped_existing"
                result["n_samples"] = expected_n
                return result
            logger.warning("%s exists but is incomplete or stale; recomputing", out_path.name)
            result["status"] = "recomputed_invalid_cache"

        edf_path = Path(cfg["paths"]["raw"]) / case / file_name
        x, fs = load_edf_channels(
            edf_path, channels, cfg["dataset"].get("channel_aliases", {}), fs_cfg
        )
        stds = np.std(x, axis=1)
        result["flat_channels"] = int(np.sum(stds < 1e-6))
        if result["flat_channels"]:
            logger.warning("%s has %d flat channel(s)", file_name, result["flat_channels"])
        if np.any(stds > IMPLAUSIBLE_STD_UV):
            logger.warning("%s has implausibly large amplitude (check units)", file_name)

        y = apply_filters(x, fs, cfg)
        problems = validate_output(y, len(channels), expected_n)
        if problems:
            raise ValueError("; ".join(problems))

        _atomic_save_npy(out_path, y)
        result["n_samples"] = int(y.shape[1])
        result["eeg_seconds"] = y.shape[1] / fs
        if result["status"] != "recomputed_invalid_cache":
            result["status"] = "success"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("Failed to preprocess %s: %s", file_name, result["error"])
    finally:
        result["elapsed_s"] = round(time.perf_counter() - started, 3)
    return result


def _append_status_log(res_df: pd.DataFrame, cfg: dict) -> Path:
    """Append this run to the cumulative status log (atomic rewrite)."""
    log_path = ensure_parent(preprocess_status_csv(cfg))
    combined = res_df
    if log_path.exists():
        try:
            combined = pd.concat([pd.read_csv(log_path), res_df], ignore_index=True)
        except Exception:
            backup = log_path.with_suffix(".csv.bak")
            os.replace(log_path, backup)
            logger.warning("Old status log unreadable; moved to %s", backup.name)
    tmp = log_path.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False)
    os.replace(tmp, log_path)
    return log_path


def preprocess_all(
    file_index: pd.DataFrame,
    cfg: dict,
    patients: list[str] | None = None,
    n_jobs: int | None = None,
    overwrite: bool = False,
    backend: str = "loky",
) -> pd.DataFrame:
    """Preprocess every included file (optionally only some patients) in parallel.

    Returns the per-file status frame of this run (also appended to the cumulative log).
    """
    df = file_index[file_index["include"].astype(bool)].copy()
    if patients:
        df = df[df["patient"].isin(patients)]
    workers = resolve_n_jobs(n_jobs, cfg)
    logger.info("Preprocessing %d files with %d worker(s)...", len(df), workers)

    rows = [row.to_dict() for _, row in df.iterrows()]
    if workers == 1 or len(rows) <= 1:
        results = [preprocess_file(r, cfg, overwrite) for r in tqdm(rows, unit="file")]
    else:
        stream = Parallel(n_jobs=workers, backend=backend, return_as="generator_unordered")(
            delayed(preprocess_file)(r, cfg, overwrite) for r in rows
        )
        results = list(tqdm(stream, total=len(rows), unit="file"))

    res_df = pd.DataFrame(results)
    if res_df.empty:
        res_df = pd.DataFrame(columns=["file", "case", "status"])
    else:
        res_df = res_df.sort_values("file").reset_index(drop=True)
    res_df.insert(0, "run_id", datetime.now().strftime("%Y%m%d-%H%M%S"))
    _append_status_log(res_df, cfg)

    failed = int((res_df["status"] == "failed").sum())
    logger.info("Preprocessing complete. Failed: %d/%d", failed, len(res_df))
    return res_df
