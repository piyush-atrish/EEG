"""SVM and Random Forest wrapped in leakage-safe sklearn pipelines (L2)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ("svm", "rf")


def make_model(name: str, cfg: dict, n_features: int, seed: int) -> tuple[Pipeline, dict]:
    """Build an unfitted model pipeline and its hyper-parameter grid.

    The scaler and the feature selector live *inside* the pipeline, so they are
    re-fitted on the training portion of every cross-validation fold and every
    LOSO fold (rule L2). Class imbalance is handled with class weights rather
    than by resampling the data seen by the classifier.

    Parameters
    ----------
    name : {"svm", "rf"}
        ``svm`` = RBF-kernel SVC, ``rf`` = Random Forest.
    cfg : dict
        Project config (``evaluation.feature_selection`` and
        ``evaluation.grids`` are read).
    n_features : int
        Number of input feature columns; ``k`` is clipped to this value.
    seed : int
        Random state for the classifier.

    Returns
    -------
    (Pipeline, dict)
        Pipeline with steps ``scale``, ``select``, ``clf`` and a parameter
        grid whose keys are prefixed with ``clf__``.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    ev = cfg["evaluation"]
    k = min(int(ev["feature_selection"]["k"]), int(n_features))

    if name == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced", cache_size=1000, random_state=seed)
    else:
        clf = RandomForestClassifier(class_weight="balanced_subsample", random_state=seed)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", clf),
    ])
    grid = {f"clf__{key}": list(values) for key, values in ev["grids"][name].items()}
    return pipe, grid
