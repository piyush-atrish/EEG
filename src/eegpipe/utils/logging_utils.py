import logging
import sys
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler (Ensures logs are never lost if terminal closes)
    try:
        from eegpipe.config import find_repo_root
        log_dir = find_repo_root() / "results" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "pipeline.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception:
        pass # Fallback gracefully if used outside project structure

    return logger