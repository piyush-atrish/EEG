"""Logging helpers (the project uses ``logging``, never ``print``)."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Return a logger with console output and, optionally, a file handler.

    Handlers are only added once per logger, so calling this repeatedly is safe.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not any(getattr(h, "_eegpipe_console", False) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_FORMAT))
        console._eegpipe_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)
    if log_file is not None:
        log_file = Path(log_file)
        already = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
            for h in logger.handlers
        )
        if not already:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(fh)
    return logger