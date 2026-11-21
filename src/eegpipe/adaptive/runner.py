"""Apply one adaptive setting to every included recording, case by case (chunk 3.1).

Design (docs/adaptive_design.md, section 3):

* one **state stream per case**: files are processed in chronological order (the number in the
  file name) and the filter state is carried from each file to the next; states are never shared
  across cases, and therefore never across patients;
* the first file of a case is **warm-started** on its first ``adaptive.warm_start_s`` seconds;
* only *included* files exist in the index, so the state is carried across the gaps left by
  excluded files (a documented deviation from a device that would have run continuously);
* **resumable**: after each file the filter state is saved next to its output, so an interrupted
  run continues from the last finished file. Skipping a finished file requires its output AND a
  state file of the same method and parameters; as soon as one file of a case has to be
  recomputed, every later file of that case is recomputed too, because its input state changed;
* **fail-safe**: a failed file stops its case (later files are ``not_run``) instead of continuing
  with a broken chain; other cases are unaffected;
* outputs are written atomically and validated (finite, correct shape and length).
"""

from __future__ import annotations

import os
import pickle
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from eegpipe.adaptive.factory import (
    adaptive_dir,
    apply_adaptive,
    params_from_config,
    resolve_params,
    warm_start_from_config,
)
from eegpipe.preprocessing.pipeline import (
    _atomic_save_npy,
    _existing_ok,
    resolve_n_jobs,
    validate_output,
)
from eegpipe.utils.logging_utils import get_logger
from eegpipe.utils.paths import ensure_parent, preprocessed_path

logger = get_logger(__name__)


def status_csv(cfg: dict) -> Path:
    return Path(cfg["paths"]["logs"]) / "adaptive_status.csv"


def file_order(file_name: str) -> tuple[int, str]:
    """Chronological sort key: CHB-MIT numbers files in recording order (``chb01_03.edf``)."""
    match = re.search(r"_(\d+)", Path(file_name).stem)
    return (int(match.group(1)) if match else 10**9, file_name)


def resolve_run(cfg: dict, method: str | None = None, overrides: dict | None = None) -> tuple[str, dict]:
    """Method and fully resolved parameters: config block first, then command-line overrides."""
    method, params = params_from_config(cfg, method)
    return method, resolve_params(method, {**params, **(overrides or {})})


def _state_path(out_path: Path) -> Path:
    return out_path.with_suffix(".state.pkl")


def _save_state(path: Path, state: dict) -> None:
    ensure_parent(path)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_state(path: Path, method: str, params: dict, warm_start_s: float) -> dict | None:
    """The saved state, or None if missing, unreadable or made with other settings."""
    try:
        with open(path, "rb") as handle:
            state = pickle.load(handle)  # local file written by this module
    except Exception:
        return None
    if not isinstance(state, dict) or state.get("method") != method or state.get("params") != params:
        return None
    if state.get("warm_start_s") != warm_start_s:
        return None
    return state


def process_case(
    rows: list[dict],
    cfg: dict,
    method: str,
    params: dict,
    overwrite: bool = False,
    warm_start_s: float | None = None,
) -> list[dict]:
    """Process the included files of ONE case in chronological order. Never raises."""
    n_ch = len(cfg["dataset"]["channels"])
    fs = float(cfg["dataset"]["fs"])
    warm = warm_start_from_config(cfg) if warm_start_s is None else float(warm_start_s)
    out_dir = adaptive_dir(cfg, method, params, warm)
    results: list[dict] = []
    state: dict | None = None
    force = overwrite  # becomes True once a file is recomputed: the chain below it must follow
    computed_any = False
    failed = False

    for row in sorted(rows, key=lambda r: file_order(r["file"])):
        case, file_name = row["case"], row["file"]
        stem = Path(file_name).stem
        out_path = out_dir / case / f"{stem}.npy"
        expected = row.get("n_samples", 0)
        expected_n = int(expected) if pd.notna(expected) else 0
        res = {"patient": row.get("patient", ""), "case": case, "file": file_name, "status": "success",
               "n_samples": 0, "eeg_seconds": 0.0, "rms_in": np.nan, "rms_out": np.nan,
               "elapsed_s": 0.0, "error": ""}
        if failed:
            res["status"] = "not_run"
            res["error"] = "an earlier file of this case failed"
            results.append(res)
            continue
        started = time.perf_counter()
        try:
            existed_valid = out_path.exists() and _existing_ok(out_path, n_ch, expected_n)
            if existed_valid and not force:
                loaded = _load_state(_state_path(out_path), method, params, warm)
                if loaded is not None:
                    state = loaded
                    res.update(status="skipped_existing", n_samples=expected_n)
                    results.append(res)
                    continue
            x = np.load(preprocessed_path(cfg, case, stem))
            y, new_state = apply_adaptive(x, method, params, fs=fs, warm_start_s=warm, state=state)
            problems = validate_output(y, n_ch, expected_n)
            if problems:
                raise ValueError("; ".join(problems))
            _atomic_save_npy(out_path, y)
            _save_state(_state_path(out_path), new_state)
            state, force = new_state, True
            if existed_valid and computed_any and not overwrite:
                res["status"] = "recomputed_downstream"
            computed_any = True
            res.update(n_samples=int(y.shape[1]), eeg_seconds=y.shape[1] / fs,
                       rms_in=float(np.sqrt(np.mean(x.astype(np.float64) ** 2))),
                       rms_out=float(np.sqrt(np.mean(y.astype(np.float64) ** 2))))
        except Exception as exc:
            failed = True
            res["status"] = "failed"
            res["error"] = f"{type(exc).__name__}: {exc}"
            logger.error("Adaptive filtering failed for %s: %s", file_name, res["error"])
        finally:
            res["elapsed_s"] = round(time.perf_counter() - started, 3)
        results.append(res)
    return results


def _append_status_log(res_df: pd.DataFrame, cfg: dict) -> Path:
    log_path = ensure_parent(status_csv(cfg))
    combined = res_df
    if log_path.exists():
        try:
            combined = pd.concat([pd.read_csv(log_path), res_df], ignore_index=True)
        except Exception:
            os.replace(log_path, log_path.with_suffix(".csv.bak"))
    tmp = log_path.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False)
    os.replace(tmp, log_path)
    return log_path


def run_adaptive(
    file_index: pd.DataFrame,
    cfg: dict,
    method: str | None = None,
    overrides: dict | None = None,
    patients: list[str] | None = None,
    n_jobs: int | None = None,
    overwrite: bool = False,
    warm_start_s: float | None = None,
) -> pd.DataFrame:
    """Run one setting over all included files (optionally some patients); cases run in parallel.

    ``warm_start_s`` overrides ``adaptive.warm_start_s`` of the config.
    """
    method, params = resolve_run(cfg, method, overrides)
    warm = warm_start_from_config(cfg) if warm_start_s is None else float(warm_start_s)
    df = file_index[file_index["include"].astype(bool)].copy()
    if patients:
        df = df[df["patient"].isin(patients)]
    groups = {case: [r.to_dict() for _, r in g.iterrows()] for case, g in df.groupby("case", sort=True)}
    workers = max(1, min(resolve_n_jobs(n_jobs, cfg), len(groups))) if groups else 1
    tag = adaptive_dir(cfg, method, params, warm).name
    logger.info("Adaptive setting %s: %d file(s) in %d case(s), %d worker(s)",
                tag, len(df), len(groups), workers)

    if workers == 1:
        outcomes = [process_case(rows, cfg, method, params, overwrite, warm)
                    for rows in tqdm(groups.values(), unit="case")]
    else:
        stream = Parallel(n_jobs=workers, backend="loky", return_as="generator_unordered")(
            delayed(process_case)(rows, cfg, method, params, overwrite, warm) for rows in groups.values()
        )
        outcomes = list(tqdm(stream, total=len(groups), unit="case"))

    results = [row for case_rows in outcomes for row in case_rows]
    res_df = pd.DataFrame(results) if results else pd.DataFrame(columns=["case", "file", "status"])
    if not res_df.empty:
        res_df = res_df.sort_values(["case", "file"]).reset_index(drop=True)
    res_df.insert(0, "run_id", f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}")
    res_df.insert(1, "setting", tag)
    _append_status_log(res_df, cfg)
    return res_df
