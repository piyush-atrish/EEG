"""Resumable, atomic downloads from PhysioNet.

Design points (each one fixes a failure mode of the first implementation):

* every file is downloaded into a private temporary directory and only moved into place when
  the transfer *and* an existence/size check have succeeded, so an interrupted download can
  never leave a truncated EDF that later runs would treat as complete;
* plain HTTPS (with a timeout and a ``Content-Length`` check) is the only transport. The
  ``wfdb.dl_files`` path was removed because it could hang on the PhysioNet server;
* progress is logged per file (``[i/N] OK <file>``) and, for large files, every
  ``PROGRESS_STEP`` bytes, so a long download is never silent;
* failures are returned to the caller, never swallowed.
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_VERSION = "1.0.0"
DEFAULT_N_CASES = 24
TIMEOUT_S = 60
CHUNK = 1 << 20  # 1 MiB
PROGRESS_STEP = 50 << 20  # log byte progress every 50 MiB


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
        header = resp.headers.get("Content-Length")
        expected = int(header) if header is not None else None
        written = 0
        with open(dest, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                handle.write(chunk)
                previous = written
                written += len(chunk)
                if written // PROGRESS_STEP > previous // PROGRESS_STEP:
                    if expected:
                        logger.info(
                            "  %s: %.0f / %.0f MB (%.0f%%)",
                            dest.name,
                            written / 1e6,
                            expected / 1e6,
                            100.0 * written / expected,
                        )
                    else:
                        logger.info("  %s: %.0f MB", dest.name, written / 1e6)
    if expected is not None and written != expected:
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
    url = _base_url(cfg) + rel_path
    for attempt in range(1, retries + 1):
        tmp_root = raw_dir / ".partial" / uuid.uuid4().hex
        staged = tmp_root / rel_path
        try:
            tmp_root.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s (attempt %d/%d)", rel_path, attempt, retries)
            _fetch_https(url, staged, timeout)
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
    """Download ``files`` in parallel, logging ``[i/N] OK|FAIL <file>`` as each one finishes."""
    if not files:
        return {"ok": [], "failed": []}
    workers = max(1, min(max_workers, len(files)))
    ok: list[str] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_file, cfg, rel): rel for rel in files}
        for done, future in enumerate(as_completed(futures), start=1):
            rel = futures[future]
            try:
                success = future.result()
            except Exception as exc:  # fetch_file should not raise; never lose a file silently
                logger.error("Unexpected error for %s: %s", rel, exc)
                success = False
            (ok if success else failed).append(rel)
            logger.info("[%d/%d] %s %s", done, len(files), "OK  " if success else "FAIL", rel)
    return {"ok": ok, "failed": failed}


def metadata_files(cfg: dict) -> list[str]:
    """``RECORDS``, ``RECORDS-WITH-SEIZURES`` and every case summary file."""
    n_cases = int(cfg["dataset"].get("n_cases", DEFAULT_N_CASES))
    files = ["RECORDS", "RECORDS-WITH-SEIZURES"]
    files += [f"chb{i:02d}/chb{i:02d}-summary.txt" for i in range(1, n_cases + 1)]
    return files


def download_metadata(cfg: dict, max_workers: int = 4) -> list[str]:
    """Download the small metadata files. Returns the list of files that failed."""
    files = metadata_files(cfg)
    logger.info("Downloading %d metadata files from PhysioNet...", len(files))
    result = _fetch_many(cfg, files, max_workers)
    logger.info(
        "Metadata download finished: %d ok, %d failed", len(result["ok"]), len(result["failed"])
    )
    if result["failed"]:
        logger.error("Metadata files that failed: %s", result["failed"])
    return result["failed"]


def download_edfs(cfg: dict, files: list[str], max_workers: int = 4) -> dict[str, list[str]]:
    """Download EDF files (``case/file.edf``). Returns ``{"ok": [...], "failed": [...]}``."""
    logger.info("Downloading %d EDF files...", len(files))
    result = _fetch_many(cfg, files, max_workers)
    logger.info("EDF download finished: %d ok, %d failed", len(result["ok"]), len(result["failed"]))
    return result
