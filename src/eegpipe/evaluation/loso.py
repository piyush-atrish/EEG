import pandas as pd


def run_loso(df: pd.DataFrame, feature_cols: list[str], model_name: str, cfg: dict, arm: str,
             patients: list[str] | None = None, shuffle_train_labels: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    raise NotImplementedError

def subsample_training(df: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    raise NotImplementedError

def tune_model(pipe, grid: dict, X, y, groups, cfg: dict, seed: int):
    raise NotImplementedError
