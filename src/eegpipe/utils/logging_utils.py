"""Logging helpers (the project uses ``logging``, never ``print``)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: Path | str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    try:
        from eegpipe.config import find_repo_root
        if log_file is None:
            log_dir = find_repo_root() / "results" / "logs"
            log_file = log_dir / "pipeline.log"

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception:
        pass

    return logger
