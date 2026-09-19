import numpy as np
import pandas as pd
from pathlib import Path
from eegpipe.utils.paths import annotations_csv, file_index_csv, preprocessed_path

def make_synthetic_summary_text(kind: str = "v1") -> str:
    if kind == "v1":
        return (
            "File Name: chb01_03.edf\n"
            "File Start Time: 13:43:04\n"
            "File End Time: 14:43:04\n"
            "Number of Seizures in File: 1\n"
            "Seizure Start Time: 2996 seconds\n"
            "Seizure End Time: 3036 seconds\n"
        )
    return (
        "File Name: chb04_05.edf\n"
        "Number of Seizures in File: 2\n"
        "Seizure 1 Start Time: 7804 seconds\n"
        "Seizure 1 End Time: 7853 seconds\n"
        "Seizure 2 Start Time: 9081 seconds\n"
        "Seizure 2 End Time: 9196 seconds\n"
    )

def make_synthetic_raw_array(fs: int = 256, seconds: float = 30.0, tones_hz=(10.0, 60.0),
                             n_ch: int = 18, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    signal = np.zeros_like(t)
    for tone in tones_hz:
        signal += np.sin(2 * np.pi * tone * t)
    noise = rng.normal(0, 0.1, size=(n_ch, len(t)))
    return (signal + noise).astype(np.float32)

def make_synthetic_project(root: Path, cfg: dict, n_patients: int = 4, files_per_patient: int = 3,
                           minutes_per_file: float = 3.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    fs = cfg["dataset"]["fs"]
    n_ch = len(cfg["dataset"]["channels"])
    n_samples = int(minutes_per_file * 60 * fs)
    
    annotations = []
    file_index = []
    
    # Force patient 0 to have two cases: chb01 and chb21
    cases_list = [("chb01", "chb01"), ("chb01", "chb21")] 
    for p in range(1, n_patients):
        case_str = f"chb{p+1:02d}"
        cases_list.append((case_str, case_str))
        
    for patient, case in cases_list:
        ar_coef = rng.uniform(0.7, 0.95)
        gain = rng.uniform(0.5, 2.0)
        
        preprocessed_dir = Path(cfg["paths"]["preprocessed"]) / case
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        for f_idx in range(files_per_patient):
            file_stem = f"{case}_{f_idx:02d}"
            file_name = f"{file_stem}.edf"
            
            # Baseline interictal noise (AR1-like for spectral color)
            noise = rng.normal(0, 1, size=(n_ch, n_samples))
            signal = np.zeros_like(noise)
            for i in range(1, n_samples):
                signal[:, i] = ar_coef * signal[:, i-1] + noise[:, i]
            signal *= gain
            
            n_seizures = 0
            if f_idx > 0: # First file is calibration
                n_seizures = rng.integers(0, 3)
                
            for s in range(n_seizures):
                start_s = rng.uniform(10, minutes_per_file * 60 - 30)
                end_s = start_s + rng.uniform(10, 20)
                
                # Inject ~3Hz rhythmic seizure
                t = np.arange(int((end_s - start_s) * fs)) / fs
                sz_rhythm = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 6 * t)
                sz_rhythm *= gain * rng.uniform(3, 5)
                
                start_idx = int(start_s * fs)
                end_idx = start_idx + len(t)
                signal[:, start_idx:end_idx] += sz_rhythm
                
                annotations.append({
                    "patient": patient, "case": case, "file": file_name,
                    "seizure_idx": s, "seizure_start_s": start_s, "seizure_end_s": end_s
                })
                
            file_index.append({
                "patient": patient, "case": case, "file": file_name,
                "file_order": f_idx, "duration_s": minutes_per_file * 60,
                "n_samples": n_samples, "n_seizures": n_seizures,
                "role": "seizure" if n_seizures > 0 else "seizure_free",
                "include": True, "exclude_reason": ""
            })
            
            np.save(preprocessed_path(cfg, case, file_stem), signal.astype(np.float32))

    ann_df = pd.DataFrame(annotations)
    idx_df = pd.DataFrame(file_index)
    
    Path(cfg["paths"]["interim"]).mkdir(parents=True, exist_ok=True)
    ann_df.to_csv(annotations_csv(cfg), index=False)
    idx_df.to_csv(file_index_csv(cfg), index=False)
    
    return {"root": root, "cfg": cfg}