"""One entry point for all adaptive methods: ``apply_adaptive(x, method, params)``.

The rest of the pipeline (script 02b, the pilot, the final run) never touches NLMS/RLS/Kalman
classes directly. It names a method and a parameter dict, and gets back the filtered array plus a
state object that can be passed to the next file of the same case, so the filter keeps adapting
across consecutive recordings (docs/adaptive_design.md, section 3).

Parameter dicts are validated: an unknown key (a typo in the YAML) is an error, not ignored.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from eegpipe.adaptive.kalman_gain import KalmanGainTracker
from eegpipe.adaptive.nlms import NLMSFilter
from eegpipe.adaptive.rls import RLSFilter

METHODS = ("nlms", "rls", "kalman")
DEFAULT_WARM_START_S = 60.0
DEFAULT_METHOD = "nlms"

# Placeholders until the pilot (chunk 4.2) selects values on the development patients. Each passed
# the synthetic seizure-preservation screen (tools/screen_adaptive.py) on 2026-11-21.
DEFAULTS: dict[str, dict] = {
    "nlms": {"length": 16, "mu": 0.005, "eps": 1e-6},
    "rls": {"length": 8, "forgetting": 0.9999, "delta": 1.0, "eps": 1e-6, "cap_factor": 100.0},
    "kalman": {"block_s": 1.0, "q_over_r": 1e-4, "r": 1.0, "min_rms": 0.1, "max_rms": 1e4},
}
# parameters that identify a setting in directory names; others appear only when non-default
_IDENTIFYING = {"nlms": ("length", "mu"), "rls": ("length", "forgetting"), "kalman": ("q_over_r", "block_s")}
_SHORT = {"length": "L", "mu": "mu", "forgetting": "lam", "q_over_r": "qr", "block_s": "blk"}


def _check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown adaptive method '{method}'; choose from {METHODS}")


def resolve_params(method: str, params: dict | None = None) -> dict:
    """Defaults for ``method`` overlaid with ``params``; unknown keys raise ``ValueError``."""
    _check_method(method)
    merged = dict(DEFAULTS[method])
    for key, value in (params or {}).items():
        if key not in merged:
            raise ValueError(f"unknown parameter '{key}' for method '{method}' (allowed: {sorted(merged)})")
        merged[key] = value
    return merged


_BLOCK_KEYS = {"method", "warm_start_s", "pilot_grid", *METHODS}


def _validate_block(block: dict) -> None:
    """Reject typos anywhere in the ``adaptive:`` block, not just in the selected method."""
    unknown = sorted(set(block) - _BLOCK_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) in the adaptive config block: {unknown}")
    for name in METHODS:
        if block.get(name):
            resolve_params(name, block[name])


def params_from_config(cfg: dict, method: str | None = None) -> tuple[str, dict]:
    """``(method, resolved params)`` from the optional ``adaptive:`` block of the config."""
    block = cfg.get("adaptive", {}) or {}
    _validate_block(block)
    method = method or block.get("method", DEFAULT_METHOD)
    _check_method(method)
    return method, resolve_params(method, block.get(method))


def warm_start_from_config(cfg: dict) -> float:
    """Warm-start length in seconds (``adaptive.warm_start_s``, default 60)."""
    value = float((cfg.get("adaptive", {}) or {}).get("warm_start_s", DEFAULT_WARM_START_S))
    if value < 0:
        raise ValueError("adaptive.warm_start_s must be >= 0")
    return value


def expand_pilot_grid(cfg: dict, method: str) -> list[dict]:
    """All parameter dicts of the pilot grid for ``method`` (cartesian product, config order)."""
    _check_method(method)
    grid = ((cfg.get("adaptive", {}) or {}).get("pilot_grid", {}) or {}).get(method, {})
    if not grid:
        return []
    keys = list(grid)
    return [resolve_params(method, dict(zip(keys, values)))
            for values in itertools.product(*(grid[k] for k in keys))]


def params_tag(method: str, params: dict | None = None) -> str:
    """Short deterministic name of a setting, e.g. ``rls_L16_lam0.9995``."""
    resolved = resolve_params(method, params)
    parts = [method]
    for key in _IDENTIFYING[method]:
        parts.append(f"{_SHORT[key]}{resolved[key]:g}")
    for key in sorted(resolved):
        if key not in _IDENTIFYING[method] and resolved[key] != DEFAULTS[method][key]:
            parts.append(f"{key}{resolved[key]:g}")
    return "_".join(parts)


def adaptive_dir(
    cfg: dict, method: str, params: dict | None = None, warm_start_s: float | None = None
) -> Path:
    """Directory of the adaptive-filtered arrays for one setting, next to the preprocessed ones.

    A warm start other than the default (60 s) is part of the name (``..._ws600``), so outputs made
    with different warm starts can never be mixed up.
    """
    warm = warm_start_from_config(cfg) if warm_start_s is None else float(warm_start_s)
    tag = params_tag(method, params)
    if warm != DEFAULT_WARM_START_S:
        tag += f"_ws{warm:g}"
    return Path(cfg["paths"]["interim"]) / "preprocessed_adaptive" / tag


def build_filter(method: str, n_channels: int, fs: float = 256.0, params: dict | None = None):
    """A fresh filter object of the requested method."""
    p = resolve_params(method, params)
    if method == "nlms":
        return NLMSFilter(n_channels, **p)
    if method == "rls":
        return RLSFilter(n_channels, **p)
    return KalmanGainTracker(n_channels, fs=fs, **p)


def apply_adaptive(
    x: np.ndarray,
    method: str,
    params: dict | None = None,
    fs: float = 256.0,
    warm_start_s: float = DEFAULT_WARM_START_S,
    state: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Filter ``x`` (n_channels, n_samples) and return ``(y float32, state)``.

    ``state=None`` starts a fresh filter, warm-started on the first ``warm_start_s`` seconds of
    ``x``. Passing the ``state`` returned for the previous file of the same case continues the same
    filter instead (no warm start), which is how a case's files are chained.
    """
    resolved = resolve_params(method, params)
    data = np.asarray(x)
    if data.ndim != 2:
        raise ValueError("x must have shape (n_channels, n_samples)")
    filt = build_filter(method, data.shape[0], fs=fs, params=resolved)
    if state is None:
        if warm_start_s > 0 and data.shape[1] > 0:
            filt.warm_start(data[:, : max(1, int(round(warm_start_s * fs)))])
    else:
        if state.get("method") != method or state.get("params") != resolved:
            raise ValueError("state was produced by a different method or parameter set")
        filt.set_state(state["arrays"])
    y = filt.process(data)
    return y, {"method": method, "params": resolved, "warm_start_s": float(warm_start_s),
               "arrays": filt.get_state()}
