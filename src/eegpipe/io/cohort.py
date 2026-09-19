import numpy as np
import pandas as pd
from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

def select_cohort(df: pd.DataFrame, cfg: dict, patients: list[str] | None = None) -> pd.DataFrame:
    """Apply data caps and filter to requested patients."""
    if patients:
        df = df[df["patient"].isin(patients)].copy()
    else:
        df = df.copy()

    df["include"] = False
    df["exclude_reason"] = "over_duration_cap"

    cap_seconds = cfg["project"]["max_hours_per_patient"] * 3600

    for patient in df['patient'].unique():
        mask = df['patient'] == patient
        pat_df = df[mask]

        # 1. Always include files with seizures
        seizure_mask = pat_df['role'] == 'seizure'
        df.loc[pat_df[seizure_mask].index, 'include'] = True
        df.loc[pat_df[seizure_mask].index, 'exclude_reason'] = ""

        current_s = pat_df.loc[seizure_mask, 'duration_s'].sum()

        # 2. Always include the very first file (calibration)
        if not pat_df.empty:
            cal_idx = pat_df.index[0]
            if not df.loc[cal_idx, 'include']:
                df.loc[cal_idx, 'include'] = True
                df.loc[cal_idx, 'exclude_reason'] = ""
                current_s += df.loc[cal_idx, 'duration_s']

        # 3. Evenly space remaining seizure-free files
        free_df = pat_df[(~pat_df.index.isin(pat_df[seizure_mask].index)) & (pat_df.index != cal_idx)]

        if not free_df.empty and current_s < cap_seconds:
            n_to_pick = min(len(free_df), int(np.ceil((cap_seconds - current_s) / 3600.0)))
            if n_to_pick > 0:
                indices = np.linspace(0, len(free_df) - 1, n_to_pick, dtype=int)
                selected_indices = free_df.iloc[indices].index

                for idx in selected_indices:
                    if current_s + df.loc[idx, 'duration_s'] <= cap_seconds + 1800: # allow slight overlap
                        df.loc[idx, 'include'] = True
                        df.loc[idx, 'exclude_reason'] = ""
                        current_s += df.loc[idx, 'duration_s']

    return df