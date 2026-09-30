"""Classifier pipelines (owner: Member C, Step 11)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_model(name: str, cfg: dict, n_features: int, seed: int) -> tuple[Pipeline, dict]:
    """Return `(Pipeline, param_grid)` for `name` in {"svm", "rf"}.

    The pipeline is `StandardScaler -> SelectKBest(f_classif) -> classifier`. Both the scaler
    and the selector live inside the pipeline (not applied beforehand) so that
    `tune_model`/`run_loso` refit them on training data only, per fold (leakage rule L2).
    Class imbalance is handled by weighting (`class_weight`), not by resampling here --
    resampling of the *training* set is `evaluation.loso.subsample_training`'s job.

    Parameters
    ----------
    name : {"svm", "rf"}
    cfg : dict
        Reads `cfg["evaluation"]["feature_selection"]["k"]` and
        `cfg["evaluation"]["grids"][name]`.
    n_features : int
        Number of available feature columns; `k` is capped at this so `SelectKBest` never
        asks for more features than exist (relevant for small synthetic test datasets).
    seed : int
        `random_state` for the classifier (SVM's RBF fit is stochastic-adjacent via internal
        solvers' tie-breaking; RF's tree bootstrapping is directly randomised).

    Returns
    -------
    (pipeline, param_grid) : the untrained pipeline and a grid dict with keys prefixed
    `"clf__"` (as `GridSearchCV` expects for a nested pipeline step), taken from
    `cfg["evaluation"]["grids"][name]`.
    """
    if name not in ("svm", "rf"):
        raise ValueError(f"Unknown model name {name!r}; expected 'svm' or 'rf'")

    k = min(cfg["evaluation"]["feature_selection"]["k"], n_features)

    if name == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced", cache_size=1000, random_state=seed)
    else:
        clf = RandomForestClassifier(class_weight="balanced_subsample", random_state=seed)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", clf),
    ])
    grid = {f"clf__{param}": values for param, values in cfg["evaluation"]["grids"][name].items()}
    return pipe, grid