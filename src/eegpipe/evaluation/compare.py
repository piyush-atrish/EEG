"""Paired comparison of arms (raw / subject_standardised / adaptive ...) across patients.

Reads the per-patient tables written by the LOSO baseline (one row per held-out patient) and,
for every model, compares each arm with a reference arm using the patient-level statistics in
:mod:`eegpipe.evaluation.stats`. Results are reported for all patients and separately for the
development patients and the evaluation patients (see docs/adaptive_design.md, section 6).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eegpipe.evaluation.stats import paired_bootstrap_ci, paired_wilcoxon

METRICS = ("auc", "sensitivity", "specificity", "fa_per_hour")
PREFIX = "per_patient_"


def load_per_patient_tables(table_dir: str | Path) -> dict[tuple[str, str], pd.DataFrame]:
    """Load ``per_patient_<arm>__<model>.csv`` files as ``{(arm, model): table indexed by patient}``."""
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    for path in sorted(Path(table_dir).glob(f"{PREFIX}*__*.csv")):
        arm, model = path.stem[len(PREFIX):].split("__", 1)
        tables[(arm, model)] = pd.read_csv(path).set_index("patient")
    return tables


def compare_arms(
    tables: dict[tuple[str, str], pd.DataFrame],
    reference: str,
    dev_patients: list[str],
    metrics: tuple[str, ...] = METRICS,
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per (model, arm, subset, metric); the difference is ``arm - reference``."""
    models = sorted({model for _, model in tables})
    if not any(arm == reference for arm, _ in tables):
        raise ValueError(f"reference arm '{reference}' not found among {sorted({a for a, _ in tables})}")
    rows = []
    for model in models:
        if (reference, model) not in tables:
            continue
        ref = tables[(reference, model)]
        for (arm, m), table in sorted(tables.items()):
            if m != model or arm == reference:
                continue
            common = ref.index.intersection(table.index)
            subsets = {
                "all": list(common),
                "dev": [p for p in common if p in set(dev_patients)],
                "eval": [p for p in common if p not in set(dev_patients)],
            }
            for subset, patients in subsets.items():
                if not patients:
                    continue
                for metric in metrics:
                    a = table.loc[patients, metric].to_numpy(dtype=float)
                    r = ref.loc[patients, metric].to_numpy(dtype=float)
                    test = paired_wilcoxon(a, r)
                    ci = paired_bootstrap_ci(a, r, n_boot=n_boot, seed=seed)
                    rows.append(
                        {
                            "model": model,
                            "arm": arm,
                            "reference": reference,
                            "subset": subset,
                            "metric": metric,
                            "n_patients": test["n_pairs"],
                            "mean_reference": float(r.mean()),
                            "mean_arm": float(a.mean()),
                            "mean_diff": test["mean_diff"],
                            "ci_low": ci["low"],
                            "ci_high": ci["high"],
                            "n_arm_greater": test["n_a_greater"],
                            "n_reference_greater": test["n_b_greater"],
                            "p_value": test["p_value"],
                            "min_possible_p": test["min_possible_p"],
                        }
                    )
    return pd.DataFrame(rows)