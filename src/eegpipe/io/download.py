"""Resumable, atomic downloads from PhysioNet.

Design points (each one fixes a failure mode of the first implementation):

* every file is downloaded into a private temporary directory and only moved into place when
  the transfer *and* an existence/size check have succeeded, so an interrupted download can
  never leave a truncated EDF that later runs would treat as complete;
* ``wfdb.dl_files`` returning normally is not trusted: the file must actually exist and be
  non-empty afterwards;
* plain HTTPS (with a timeout and a ``Content-Length`` check) is the fallback;
* failures are returned to the caller, never swallowed.
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import wfdb

from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_VERSION = "1.0.0"
DEFAULT_N_CASES = 24
TIMEOUT_S = 60
CHUNK = 1 << 20


def _base_url(cfg: dict) -> str:
    db = cfg["dataset"]["physionet_db"]
    version = cfg["dataset"].get("physionet_version", DEFAULT_VERSION)
    return f"https://physionet.org/files/{db}/{version}/"


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _fetch_https(url: str, dest: Path, timeout: float = TIMEOUT_S) -> None:
    """Stream ``url`` to ``dest``; raise if the transfer is incomplete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        expected = resp.headers.get("Content-Length")
        written = 0
        with open(dest, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                handle.write(chunk)
                written += len(chunk)
    if expected is not None and written != int(expected):
        raise OSError(f"incomplete download of {url}: {written} of {expected} bytes")


def fetch_file(
    cfg: dict,
    rel_path: str,
    retries: int = 3,
    backoff_s: float = 2.0,
    timeout: float = TIMEOUT_S,
) -> bool:
    """Ensure ``<raw>/<rel_path>`` exists, downloading it if necessary. Returns success."""
    raw_dir = Path(cfg["paths"]["raw"])
    target = raw_dir / rel_path
    if _nonempty(target):
        return True
    db = cfg["dataset"]["physionet_db"]
    for attempt in range(1, retries + 1):
        tmp_root = raw_dir / ".partial" / uuid.uuid4().hex
        staged = tmp_root / rel_path
        try:
            tmp_root.mkdir(parents=True, exist_ok=True)
            try:
                wfdb.dl_files(db, str(tmp_root), [rel_path], keep_subdirs=True, overwrite=True)
            except Exception as exc:
                logger.warning("wfdb failed for %s (%s); trying HTTPS", rel_path, exc)
            if not _nonempty(staged):  # wfdb failed or "succeeded" without writing the file
                _fetch_https(_base_url(cfg) + rel_path, staged, timeout)
            if not _nonempty(staged):
                raise OSError("download produced no data")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)
            return True
        except Exception as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, rel_path, exc)
            if attempt < retries:
                time.sleep(backoff_s * attempt)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
    logger.error("Giving up on %s after %d attempts", rel_path, retries)
    return False


def _fetch_many(cfg: dict, files: list[str], max_workers: int) -> dict[str, list[str]]:
    if not files:
        return {"ok": [], "failed": []}
    workers = max(1, min(max_workers, len(files)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda rel: fetch_file(cfg, rel), files))
    return {
        "ok": [f for f, ok in zip(files, results, strict=True) if ok],
        "failed": [f for f, ok in zip(files, results, strict=True) if not ok],
    }


def metadata_files(cfg: dict) -> list[str]:
    """``RECORDS``, ``RECORDS-WITH-SEIZURES`` and every case summary file."""
    n_cases = int(cfg["dataset"].get("n_cases", DEFAULT_N_CASES))
    files = ["RECORDS", "RECORDS-WITH-SEIZURES"]
    files += [f"chb{i:02d}/chb{i:02d}-summary.txt" for i in range(1, n_cases + 1)]
    return files


def download_metadata(cfg: dict, max_workers: int = 4) -> list[str]:
    """Download the small metadata files. Returns the list of files that failed."""
    logger.info("Downloading metadata files from PhysioNet...")
    result = _fetch_many(cfg, metadata_files(cfg), max_workers)
    if result["failed"]:
        logger.error("Metadata files that failed: %s", result["failed"])
    return result["failed"]


def download_edfs(cfg: dict, files: list[str], max_workers: int = 4) -> dict[str, list[str]]:
    """Download EDF files (``case/file.edf``). Returns ``{"ok": [...], "failed": [...]}``."""
    logger.info("Downloading %d EDF files...", len(files))
    result = _fetch_many(cfg, files, max_workers)
    logger.info("EDF download finished: %d ok, %d failed", len(result["ok"]), len(result["failed"]))
    return result