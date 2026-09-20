# Member A: fix checklist (from the review of the agent output)

Legend: **VERIFIED** = covered by an automated test in this repo (92 tests, ruff clean);
**YOUR MACHINE** = needs a run on real data or Windows that only you can do.

## Contract and scaffold
- [x] `configs/config.yaml` restored verbatim from README section 6 (66 keys; test guards against forks) - VERIFIED
- [x] `config.py`: reads `configs/config.yaml`, raises on a missing file, deep-merges overrides,
      resolves paths against the repo root, validates channels/window/overlap - VERIFIED
- [x] `utils/paths.py`: all 9 helpers incl. `windows_path`, `features_path`, `predictions_path`, `preprocess_status_csv`
- [x] `utils/logging_utils.py`: idempotent handlers, optional `log_file`, no import-time file side effects
- [x] Stubs with the README file names and signatures for every B/C module and scripts 03-06 - VERIFIED (skipped in integration test)
- [x] Makefile with every README target (real tabs) - `make -n all` checked
- [x] `.gitignore` tracks nested `.gitkeep`, ignores data/edf/npy/parquet/logs - checked in a fresh repo
- [x] `requirements.txt` covers every import (scan: none missing); B and C blocks filled
- [x] CI (`ci.yml`): ruff + unit + integration; PR template; CODEOWNERS per folder; pre-commit (ruff, nbstripout)
- [ ] Replace `@handleA/@handleB/@handleC` in `.github/CODEOWNERS` with real GitHub usernames - YOUR MACHINE
- [ ] Pin exact versions in `requirements.txt` after `pip freeze` on your machine - YOUR MACHINE

## Data layer (Steps 2-3)
- [x] Annotation parser: both formats, declared-vs-parsed count check, invalid intervals skipped, deterministic order, midnight-safe durations - VERIFIED
- [x] Loader: name-based, microvolts, duplicate `T8-P8`, chunked low-memory read, bit-identical to old loader, exact `n_samples` (no off-by-one) - VERIFIED on real EDF bytes
- [x] `check_edf` reasons: `unreadable_edf`, `bad_fs`, `missing_channel` - VERIFIED
- [x] Cohort: cap counts seizure-free hours only, real durations, evenly spread extras, back-fill, no-seizure patients, `file_order` unique per patient - VERIFIED
- [x] Download: atomic (temp dir + replace), post-download existence check, HTTPS fallback with Content-Length check, timeout, retries, failed-files list - VERIFIED with mocks
- [x] Script 01: iterate-until-stable, merge (never clobbers other patients), exit codes - VERIFIED end to end
- [ ] Real download of PhysioNet files with the new `download.py` - YOUR MACHINE (sandbox has no access)

## DSP layer (Steps 4-5)
- [x] DC test now meaningful (relative, and fails for a no-op filter) - VERIFIED
- [x] Filter applied per channel (4.3x lower peak memory on a 4 h file), same numbers as before - VERIFIED
- [x] Plot moved out of the tests (`python -m eegpipe.preprocessing.filters`) - figure regenerated
- [x] Pipeline: atomic writes, existing-cache integrity check, strict validation (finite, shape, length vs index, stale-index hint), worker cap 4, run-tagged appended log, tqdm - VERIFIED
- [x] Script 02: exit code 1 on any failure, `--overwrite`, `--workers` - VERIFIED
- [ ] Re-run script 01 then 02 for chb01 chb02 on your machine (Windows, Python 3.13) - YOUR MACHINE
- [ ] Fill in `docs/data_notes.md` (volume, problem files, T8-P8 duplicate check) - YOUR MACHINE

## Tests
- [x] `tests/test_synth_signals.py`, `test_annotations.py`, `test_download.py`, `test_loader_cohort.py`,
      `test_index_script.py`, `test_filters.py`, `test_preprocess_pipeline.py`, `test_integration.py`, `test_config.py`
- [x] `tests/conftest.py` shared fixtures (joint file: needs all three approvals in the PR)
