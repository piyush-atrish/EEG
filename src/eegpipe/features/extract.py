import numpy as np
import pandas as pd

def feature_names(cfg: dict) -> list[str]:
    raise NotImplementedError

def extract_window_features(win: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    raise NotImplementedError

def extract_batch_features(batch: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    raise NotImplementedError

def extract_case_features(case: str, file_index: pd.DataFrame, window_table: pd.DataFrame,
                          cfg: dict, batch_size: int = 256) -> pd.DataFrame:
    raise NotImplementedError