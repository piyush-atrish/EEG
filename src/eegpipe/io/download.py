import time
from pathlib import Path
import wfdb
import requests
from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

def download_metadata(cfg: dict) -> None:
    """Download RECORDS and all summary text files from PhysioNet."""
    raw_dir = Path(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    db = cfg["dataset"]["physionet_db"]

    files_to_download = ["RECORDS", "RECORDS-WITH-SEIZURES"]
    for i in range(1, 25):  # chb01 to chb24
        case = f"chb{i:02d}"
        files_to_download.append(f"{case}/{case}-summary.txt")

    logger.info("Downloading metadata files from PhysioNet...")
    try:
        wfdb.dl_files(db, str(raw_dir), files_to_download, keep_subdirs=True, overwrite=False)
    except Exception as e:
        logger.warning(f"wfdb download failed ({e}). Falling back to HTTPS.")
        base_url = f"https://physionet.org/files/{db}/1.0.0/"
        for file_path in files_to_download:
            target = raw_dir / file_path
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                resp = requests.get(base_url + file_path)
                if resp.status_code == 200:
                    with open(target, 'wb') as f:
                        f.write(resp.content)
                else:
                    logger.error(f"Failed to fetch {file_path}")
    logger.info("Metadata download complete.")

def download_edfs(cfg: dict, files: list[str]) -> None:
    """Download specific EDF files with retry logic, skipping existing files."""
    raw_dir = Path(cfg["paths"]["raw"])
    db = cfg["dataset"]["physionet_db"]
    
    logger.info(f"Downloading {len(files)} EDF files...")
    for file_path in files:
        target = raw_dir / file_path
        if target.exists():
            continue
        
        retries = 3
        for attempt in range(retries):
            try:
                wfdb.dl_files(db, str(raw_dir), [file_path], keep_subdirs=True, overwrite=False)
                break
            except Exception as e:
                logger.warning(f"Failed to download {file_path} (Attempt {attempt+1}/{retries}): {e}")
                time.sleep(2)
    logger.info("EDF download phase complete.")