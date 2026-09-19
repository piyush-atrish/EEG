import pandas as pd
import numpy as np
from pathlib import Path
from eegpipe.io.loader import read_edf_header
from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

def build_file_index(cfg: dict, annotations: pd.DataFrame) -> pd.DataFrame:
    """Scan downloaded records to build the baseline index (Contract C2 structure)."""
    raw_dir = Path(cfg["paths"]["raw"])
    fs = cfg["dataset"]["fs"]
    
    records_path = raw_dir / "RECORDS"
    if not records_path.exists():
        return pd.DataFrame()
        
    with open(records_path, 'r') as f:
        edf_files = [line.strip() for line in f if line.strip().endswith('.edf')]
        
    rows = []
    for edf_rel in edf_files:
        case, file_name = edf_rel.split('/')
        patient = cfg["dataset"]["patient_map"].get(case, case)
        
        # Extract numeric suffix for ordering (e.g. chb17a_03.edf -> 3)
        stem = Path(file_name).stem
        suffix = ''.join(filter(str.isdigit, stem.split('_')[-1]))
        file_order = int(suffix) if suffix else 0
        
        n_seizures = len(annotations[annotations["file"] == file_name])
        
        # Default duration (1 hour) if not yet downloaded
        duration_s = 3600.0 
        edf_full_path = raw_dir / edf_rel
        if edf_full_path.exists():
            try:
                hdr = read_edf_header(edf_full_path)
                duration_s = hdr["duration_s"]
            except Exception:
                pass
        
        rows.append({
            "patient": patient, "case": case, "file": file_name,
            "file_order": file_order, "duration_s": duration_s,
            "n_samples": int(round(duration_s * fs)),
            "n_seizures": n_seizures,
            "role": "seizure" if n_seizures > 0 else "seizure_free",
            "include": False, "exclude_reason": ""
        })
        
    df = pd.DataFrame(rows)
    return df.sort_values(by=["patient", "case", "file_order"]).reset_index(drop=True)

def select_cohort(index: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply inclusion rules and data capping per patient."""
    df = index.copy()
    df["include"] = False
    df["exclude_reason"] = "over_cap"
    max_hours = cfg["dataset"]["max_seizure_free_hours_per_patient"]
    
    for patient, group in df.groupby("patient"):
        seizure_mask = group["role"] == "seizure"
        
        if not seizure_mask.any():
            df.loc[group.index, "exclude_reason"] = "no_seizure_patient"
            continue
            
        # 1. Include all seizure files
        df.loc[group[seizure_mask].index, "include"] = True
        df.loc[group[seizure_mask].index, "exclude_reason"] = ""
        
        included_sf_hours = 0.0
        
        # 2. First file of every case is included for calibration
        for case, case_group in group.groupby("case"):
            first_idx = case_group.index[0]
            if not df.loc[first_idx, "include"]:
                df.loc[first_idx, "include"] = True
                df.loc[first_idx, "exclude_reason"] = ""
                included_sf_hours += df.loc[first_idx, "duration_s"] / 3600.0
        
        # 3. Add random seizure-free files until cap is reached deterministically
        sf_candidates = group[(group["role"] == "seizure_free") & (~df.loc[group.index, "include"])]
        if not sf_candidates.empty and included_sf_hours < max_hours:
            rng = np.random.default_rng(cfg["project"]["seed"])
            shuffled_indices = rng.permutation(sf_candidates.index)
            
            for idx in shuffled_indices:
                if included_sf_hours >= max_hours:
                    break
                df.loc[idx, "include"] = True
                df.loc[idx, "exclude_reason"] = ""
                included_sf_hours += df.loc[idx, "duration_s"] / 3600.0
    return df