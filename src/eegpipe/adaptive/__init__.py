"""Causal, label-free adaptive filtering stage (Milestone 2).

Use :func:`apply_adaptive` from the pipeline; the individual filter classes live in
``nlms``, ``rls`` and ``kalman_gain``. Design: docs/adaptive_design.md.
"""

from eegpipe.adaptive.factory import (
    METHODS,
    adaptive_dir,
    adaptive_features_dir,
    adaptive_root,
    apply_adaptive,
    build_filter,
    expand_pilot_grid,
    list_settings,
    params_from_config,
    params_tag,
    resolve_params,
    warm_start_from_config,
)

__all__ = [
    "METHODS",
    "adaptive_dir",
    "adaptive_features_dir",
    "adaptive_root",
    "apply_adaptive",
    "build_filter",
    "expand_pilot_grid",
    "list_settings",
    "params_from_config",
    "params_tag",
    "resolve_params",
    "warm_start_from_config",
]
