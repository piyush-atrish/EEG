"""Run the L8 label-shuffle sanity check against real feature data.

Usage:  python tools/check_label_shuffle.py [--config configs/config.yaml] [--model svm]
"""
from __future__ import annotations

import argparse

import pandas as pd

from eegpipe.config import load_config
from eegpipe.evaluation.leakage_checks import label_shuffle_check
from eegpipe.utils.paths import features_path, file_index_csv

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--model", choices=["svm", "rf"], default="svm")
args = parser.parse_args()

cfg = load_config(args.config)
file_index = pd.read_csv(file_index_csv(cfg))
included = file_index.loc[file_index["include"].astype(bool)]
cases = sorted(included["case"].unique())
df = pd.concat([pd.read_parquet(features_path(cfg, case)) for case in cases], ignore_index=True)
feature_cols = [c for c in df.columns if c.startswith("f_")]

n_patients = df["patient"].nunique()
n_test = min(3, n_patients)
auc = label_shuffle_check(df, feature_cols, args.model, cfg, n_test_patients=n_test, seed=0)

print(f"Patients available: {n_patients}")
print(f"Label-shuffle AUC ({args.model}, {n_test} held-out patient(s)): {auc:.3f}")
if 0.35 <= auc <= 0.65:
    print("OK: near chance, as expected when training labels are shuffled.")
else:
    print("WARNING: far from 0.5 -- investigate for a leak before trusting any real result.")
