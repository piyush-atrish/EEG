# eeg-seizure-loso

**Cross-Subject Generalisation of EEG Signals Using Adaptive Filtering for Epileptic Seizure Detection**
B.Tech Final Year Project, Department of Electronics and Communication Engineering, Delhi Technological University (Batch 2023-2027). Guide: Dr. Sonal Singh.

> **This README covers Milestone 1 only: the non-adapted baseline DSP pipeline** (pre-processing, windowing, feature extraction, and a Leave-One-Subject-Out baseline with SVM and Random Forest). Adaptive filtering (RLS/LMS) is Milestone 2 and is **out of scope here**; its folder is reserved and empty.

This file is written so that a team member can hand it, together with the project context documents (synopsis and literature report), to an AI coding agent, and the agent can start working immediately without further explanation. It is the single source of truth for structure, file ownership, configuration, and interface contracts.

---

## Table of contents

- [0. Quick start for AI agents](#0-quick-start-for-ai-agents)
- [1. Project context and scope](#1-project-context-and-scope)
- [2. Design lock](#2-design-lock)
- [3. Repository structure](#3-repository-structure)
- [4. Ownership and Git workflow](#4-ownership-and-git-workflow)
- [5. Environment and commands](#5-environment-and-commands)
- [6. Configuration](#6-configuration)
- [7. Interface contracts](#7-interface-contracts)
- [8. Work table of contents](#8-work-table-of-contents)
- [9. Member A: data and preprocessing](#9-member-a-data-and-preprocessing)
- [10. Member B: windows and features](#10-member-b-windows-and-features)
- [11. Member C: evaluation and baseline models](#11-member-c-evaluation-and-baseline-models)
- [12. Integration and definition of done](#12-integration-and-definition-of-done)
- [Appendix A: CHB-MIT dataset notes](#appendix-a-chb-mit-dataset-notes)

---

## 0. Quick start for AI agents

**For the human:** attach the project context documents plus this README, then paste the prompt below with your member letter filled in. Do this in your own clone of the repo.

```text
You are my coding agent on the repository `eeg-seizure-loso`.
I am Member <A | B | C>.
Attached: the project context documents (synopsis and literature report) and README.md.

1. Read README.md completely. Sections 3 (structure), 6 (config) and 7 (contracts) are binding.
2. Before writing any code, restate: (a) the files I own, (b) what I consume from others,
   (c) what I must produce, (d) my steps in order.
3. Work through my steps in order, starting at the first step not yet ticked in Section 8.
4. Edit ONLY files I own (Section 3 legend). If something outside my files must change,
   stop and tell me. Do not edit it yourself.
5. Implement exactly the function signatures and file formats in Section 7.
   Build and test against SYNTHETIC data first. Use real data only after synthetic tests pass.
6. Write the tests listed in each step, run them, and show me the output before claiming a step is done.
7. Briefly explain every non-trivial design decision so I can defend it in the viva.
```

**Rules the agent must follow (non-negotiable):**

1. **Ownership.** Only create or edit files tagged with your letter in [Section 3](#3-repository-structure). Files tagged `[J]` (joint) change only by a pull request approved by all three members.
2. **Contracts are frozen.** Column names, file formats, shapes, units, and function signatures in [Section 7](#7-interface-contracts) must not change. If you believe a change is needed, stop and raise it with the human. Do not silently deviate.
3. **No hard-coded values or paths.** Everything tunable comes from `configs/config.yaml` through `eegpipe.config.load_config`.
4. **Leakage rules L1-L8** in [Section 2](#2-design-lock) apply to all code that touches labels or subjects.
5. **Determinism.** Seed everything from `cfg["project"]["seed"]`.
6. **Never commit data**, caches, predictions, or logs. They are git-ignored.
7. **Code standard:** Python 3.11, type hints on public functions, NumPy-style docstrings, `logging` instead of `print`, no notebook code imported by library modules.
8. **Subpackage `__init__.py` files stay empty** (docstring only), so there are no cross-owner edits and no circular imports. Import from the module directly, for example `from eegpipe.preprocessing.filters import apply_filters`.
9. **Every step ends with tests passing** (`pytest tests/<your files>`) and a short entry in your section of the progress checklist in Section 8.

---

## 1. Project context and scope

### 1.1 The problem

Seizure classifiers trained per patient degrade on unseen patients (cross-subject generalisation gap). The literature mostly closes this gap with deep domain adaptation. This project tests a lightweight, DSP-first alternative: adaptive filtering (RLS/LMS) to normalise inter-subject variability before feature extraction, followed by interpretable classifiers (SVM, Random Forest), evaluated with Leave-One-Subject-Out (LOSO) on the CHB-MIT Scalp EEG Database.

### 1.2 What Milestone 1 delivers

A complete, leakage-free, **non-adapted** baseline that Milestone 2 will be compared against:

```text
raw EDF (CHB-MIT)
  |
  |-- 01 download + index ---------> annotations.csv, file_index.csv
  |
  |-- 02 preprocess (notch 60 Hz + 0.5-45 Hz Kaiser FIR band-pass)
  |         ---------------------> preprocessed/{case}/{file}.npy
  |
  |-- 03 windows (4 s, 50 % overlap, labels) -------> windows/{case}.parquet
  |
  |-- 04 features (32 per channel x 18 channels) ---> features/{case}.parquet
  |
  |-- 05 LOSO baseline
  |        arm "raw"                  (no normalisation)
  |        arm "subject_standardised" (per-patient standardisation from a calibration segment)
  |        models: SVM (RBF), Random Forest
  |         ---------------------> predictions + per-patient metrics
  |
  '-- 06 report ------------------> tables + figures
```

### 1.3 In scope and out of scope

| In scope (Milestone 1) | Out of scope (later milestones) |
|---|---|
| Download, indexing, annotation parsing | Adaptive filtering (RLS/LMS/Kalman) |
| Notch + Kaiser FIR band-pass pre-processing | Quantisation sweep, channel-reduction Pareto, multiplierless FIR |
| 4 s windowing and labelling | Pre-ictal prediction (this project is detection only) |
| Time, frequency, entropy and wavelet features | Deep learning models |
| SVM and Random Forest under LOSO | Event-based (onset-latency) scoring |
| Arm 0 (raw) and Arm 1 (per-patient standardisation) | Full MATLAB cross-validation of every design (one quick check only) |

---

## 2. Design lock

These decisions are fixed for Milestone 1. Changing one requires agreement from all three members.

| Topic | Decision |
|---|---|
| Dataset | CHB-MIT Scalp EEG Database (PhysioNet), fs = 256 Hz |
| Patient identity | `chb21` is the **same patient** as `chb01` (recorded 1.5 years later). Both map to patient `chb01`. LOSO groups by `patient`, never by `case` |
| Channels | 18 common bipolar channels in a fixed order (see `configs/config.yaml`), selected **by name**, never by index |
| Signal units | Microvolts (uV), float32 |
| Pre-processing | Notch at 60 Hz (US mains) then 0.5-45 Hz linear-phase Kaiser-window FIR band-pass, delay-compensated |
| Windows | 4 s (1024 samples), 50 % overlap (step 2 s) |
| Labels | Window fully inside an annotated seizure = 1 (ictal). Window with no overlap with any seizure = 0 (interictal). Window partially overlapping a seizure = **dropped** |
| Detection, not prediction | No pre-ictal class |
| Features | 32 per channel, 18 channels, 576 columns (catalog in `configs/config.yaml`) |
| Models | SVM (RBF kernel) and Random Forest, each inside an sklearn `Pipeline` (StandardScaler, SelectKBest, classifier) |
| Evaluation | Leave-One-Subject-Out by `patient`; held-out patient scored at natural class balance |
| Arms | `raw` (no normalisation) and `subject_standardised` (per-patient statistics from that patient's own early interictal calibration windows) |
| Calibration windows | First N interictal windows before the patient's first ictal window (N from config, default 10 min). **Excluded from scoring in both arms** so both arms are scored on identical windows |
| Metrics (window-level) | Sensitivity, specificity, F1, AUC, false alarms per hour, reported per patient plus mean, SD, median |
| Data volume control | All seizure files are kept. Seizure-free data is capped per patient (config) |

### Leakage rules (L1-L8)

| ID | Rule |
|---|---|
| L1 | Train and test patient sets are disjoint. chb01 and chb21 are one patient |
| L2 | Anything that is *fitted* (scaler, feature selection, hyper-parameters, resampling) is fitted on training patients only, inside an sklearn `Pipeline` |
| L3 | The held-out patient is never resampled or rebalanced. It is scored at its natural class distribution |
| L4 | Hyper-parameter tuning uses `GroupKFold` grouped by `patient`, inside the training patients only (nested) |
| L5 | Arm 1 statistics come only from that patient's own calibration windows (no labels used to compute them, no other patient involved) |
| L6 | Features are computed per window from that window only (no cross-window or cross-patient statistics) |
| L7 | Random seeds fixed from config |
| L8 | Sanity gate: a label-shuffle run must give AUC near 0.5. Near-perfect LOSO results are treated as a bug until proven otherwise |

---
## 3. Repository structure

**Legend:** `[A]` `[B]` `[C]` = owned exclusively by that member. `[J]` = joint, changed only by a PR approved by all three. `[gen]` = generated at run time, never committed. `[reserved]` = placeholder for a later milestone.

```text
eeg-seizure-loso/
|
|-- README.md                                   [J]  this file
|-- pyproject.toml                              [J]  package metadata (src layout), pytest markers, ruff config
|-- requirements.txt                            [J]  pinned deps, split into "# --- A ---", "# --- B ---", "# --- C ---" blocks
|-- Makefile                                    [J]  one target per stage (see Section 5)
|-- .gitignore                                  [J]  data/, results/predictions, results/logs, caches, checkpoints
|-- .pre-commit-config.yaml                     [J]  ruff + nbstripout (optional)
|
|-- .github/
|   |-- CODEOWNERS                              [J]  folder -> owner mapping
|   |-- pull_request_template.md                [J]
|   '-- workflows/
|       '-- ci.yml                              [J]  ruff + pytest (synthetic data only) on every PR
|
|-- configs/
|   '-- config.yaml                             [J]  FROZEN: fs, channels, filters, windows, features catalog, eval settings
|
|-- docs/
|   |-- design_lock.md                          [J]  copy of Section 2 (short, for the report)
|   |-- data_notes.md                           [A]  data quirks log: problem files, channel issues, exclusions
|   |-- filter_validation.md                    [A]  Step 4 results: response plots, attenuation numbers
|   |-- feature_catalog.md                      [B]  definition and formula of every feature
|   |-- baseline_report.md                      [C]  Step 12 results summary and known limitations
|   '-- figures/                                [A/B/C]  each member prefixes files with own letter (A_*.png ...)
|
|-- data/                                       [gen]  NEVER committed (only .gitkeep files)
|   |-- raw/chbmit/                                    EDF files, summary files, RECORDS lists
|   |-- interim/
|   |   |-- annotations.csv                            Contract C1 (A)
|   |   |-- file_index.csv                             Contract C2 (A)
|   |   '-- preprocessed/{case}/{file_stem}.npy        Contract C3 (A)
|   '-- processed/
|       |-- windows/{case}.parquet                     Contract C4 (B)
|       '-- features/{case}.parquet                    Contract C5 (B)
|
|-- src/eegpipe/
|   |-- __init__.py                             [A]  version string only
|   |-- config.py                               [A]  load_config(), channel_slug(), get_paths()
|   |
|   |-- utils/
|   |   |-- __init__.py                         [A]
|   |   |-- logging_utils.py                    [A]  get_logger(name), file+console logging setup
|   |   |-- seed.py                             [A]  set_global_seed(seed)
|   |   '-- paths.py                            [A]  path builders for every artifact in Section 7
|   |
|   |-- io/
|   |   |-- __init__.py                         [A]
|   |   |-- download.py                         [A]  fetch summaries/RECORDS, then only cohort EDFs
|   |   |-- annotations.py                      [A]  parse chbXX-summary.txt -> annotations.csv
|   |   |-- loader.py                           [A]  read EDF -> (18, n) float32 uV by channel name
|   |   '-- cohort.py                           [A]  build file_index.csv, apply cohort selection rules
|   |
|   |-- preprocessing/
|   |   |-- __init__.py                         [A]
|   |   |-- filters.py                          [A]  Kaiser FIR band-pass, notch, apply_filters(), freq_response()
|   |   '-- pipeline.py                         [A]  preprocess_file(), preprocess_all() (parallel, cached)
|   |
|   |-- segmentation/
|   |   |-- __init__.py                         [B]
|   |   '-- windows.py                          [B]  make_windows(), build_window_table()
|   |
|   |-- features/
|   |   |-- __init__.py                         [B]
|   |   |-- time_domain.py                      [B]  mean, std, skew, kurt, line length, Hjorth
|   |   |-- frequency_domain.py                 [B]  Welch band power (log and relative)
|   |   |-- nonlinear.py                        [B]  sample entropy, approximate entropy (numba)
|   |   |-- wavelet.py                          [B]  DWT db4 level 5: log-energy and std per sub-band
|   |   '-- extract.py                          [B]  feature_names(), extract_window_features(), extract_case_features()
|   |
|   |-- models/
|   |   |-- __init__.py                         [C]
|   |   |-- normalisation.py                    [C]  mark_calibration(), apply_arm()
|   |   '-- classifiers.py                      [C]  make_model() -> (Pipeline, param_grid)
|   |
|   |-- evaluation/
|   |   |-- __init__.py                         [C]
|   |   |-- metrics.py                          [C]  compute_metrics(), aggregate_metrics()
|   |   |-- leakage_checks.py                   [C]  assertions + label_shuffle_check()
|   |   |-- loso.py                             [C]  run_loso(), subsample_training(), tune_model()
|   |   '-- reporting.py                        [C]  save_tables(), plot_per_patient_bars(), summary tables
|   |
|   '-- adaptive/
|       '-- __init__.py                         [reserved]  Milestone 2 (RLS/NLMS/Kalman), leave empty
|
|-- scripts/                                    thin CLI wrappers; each has main(argv) and accepts --config and --patients
|   |-- 01_download_and_index.py                [A]
|   |-- 02_preprocess.py                        [A]
|   |-- 03_make_windows.py                      [B]
|   |-- 04_extract_features.py                  [B]
|   |-- 05_run_baseline.py                      [C]
|   '-- 06_make_report.py                       [C]
|
|-- tests/
|   |-- conftest.py                             [J]  shared fixtures; frozen after kickoff
|   |-- fixtures/
|   |   |-- __init__.py                         [A]
|   |   |-- synth_signals.py                    [A]  fake summary text, fake patients/signals/annotations/file_index
|   |   '-- synth_features.py                   [C]  fake feature table in Contract C5 format
|   |-- test_config.py                          [A]
|   |-- test_annotations.py                     [A]
|   |-- test_loader_cohort.py                   [A]
|   |-- test_filters.py                         [A]
|   |-- test_preprocess_pipeline.py             [A]
|   |-- test_windows.py                         [B]
|   |-- test_time_frequency_features.py         [B]
|   |-- test_nonlinear_wavelet_features.py      [B]
|   |-- test_extract.py                         [B]
|   |-- test_metrics.py                         [C]
|   |-- test_leakage_checks.py                  [C]
|   |-- test_models_normalisation.py            [C]
|   |-- test_loso.py                            [C]
|   '-- test_integration.py                     [J]  full 03 -> 05 chain on synthetic data (marker: integration)
|
|-- notebooks/                                  exploration only, outputs stripped, never imported by src/
|   |-- A_data_and_filter_exploration.ipynb     [A]
|   |-- B_feature_exploration.ipynb             [B]
|   '-- C_results_exploration.ipynb             [C]
|
'-- results/
    |-- tables/                                 [C]  committed: per_patient_{arm}__{model}.csv, summary_all.csv
    |-- figures/                                [C]  committed: baseline plots (PNG)
    |-- predictions/                            [gen] Contract C6, git-ignored
    '-- logs/                                   [gen] git-ignored
```

**Counts (excluding notebooks):** Member A owns 16 source/script files, 5 test files, 2 fixture files and 2 docs. Member B owns 10 source/script files, 4 test files and 1 doc. Member C owns 10 source/script files, 4 test files, 1 fixture file and 1 doc. Member C carries the largest module (`loso.py`); Member A has the longest wall-clock dependency (the download).

---

## 4. Ownership and Git workflow

### 4.1 Rules that prevent merge conflicts

1. **Edit only your own files.** Shared files (`[J]`) change only through a PR approved by all three.
2. **`requirements.txt` is pre-split** into three blocks at kickoff (`# --- A ---`, `# --- B ---`, `# --- C ---`). Add dependencies only inside your own block. Different regions merge without conflicts.
3. **`Makefile` is pre-split** the same way: every target already exists after kickoff. Nobody adds targets later.
4. **Subpackage `__init__.py` files are empty**, so no re-exports and no cross-owner edits.
5. **Each member has their own fixtures file**, so nobody edits another member's test data generator.

### 4.2 Branches, commits, pull requests

- `main` is protected. Nobody pushes to it directly. Use short-lived branches: `A/step-3-loader`, `B/step-7-features`, `C/step-10-loso`.
- Commit messages start with owner and step: `[A][step-4] add Kaiser FIR design and validation test`.
- Open a PR as soon as a step passes its tests. Another member reviews it (target: same day). Squash-merge. Rebase on `main` before merging.
- CI must pass (ruff plus pytest on synthetic data). Broken interfaces are caught here, not at integration.
- PR checklist (`.github/pull_request_template.md`): only own files touched, contracts unchanged, tests added and passing, no data committed, leakage rules respected.

### 4.3 CODEOWNERS (replace handles at kickoff)

```text
/README.md                          @handleA @handleB @handleC
/configs/                           @handleA @handleB @handleC
/pyproject.toml                     @handleA @handleB @handleC
/requirements.txt                   @handleA @handleB @handleC
/Makefile                           @handleA @handleB @handleC
/.github/                           @handleA @handleB @handleC
/tests/conftest.py                  @handleA @handleB @handleC
/tests/test_integration.py          @handleA @handleB @handleC

/src/eegpipe/config.py              @handleA
/src/eegpipe/utils/                 @handleA
/src/eegpipe/io/                    @handleA
/src/eegpipe/preprocessing/         @handleA
/scripts/01_*.py                    @handleA
/scripts/02_*.py                    @handleA
/tests/fixtures/synth_signals.py    @handleA
/tests/test_config.py               @handleA
/tests/test_annotations.py          @handleA
/tests/test_loader_cohort.py        @handleA
/tests/test_filters.py              @handleA
/tests/test_preprocess_pipeline.py  @handleA
/docs/data_notes.md                 @handleA
/docs/filter_validation.md          @handleA

/src/eegpipe/segmentation/          @handleB
/src/eegpipe/features/              @handleB
/scripts/03_*.py                    @handleB
/scripts/04_*.py                    @handleB
/tests/test_windows.py              @handleB
/tests/test_time_frequency_features.py    @handleB
/tests/test_nonlinear_wavelet_features.py @handleB
/tests/test_extract.py              @handleB
/docs/feature_catalog.md            @handleB

/src/eegpipe/models/                @handleC
/src/eegpipe/evaluation/            @handleC
/scripts/05_*.py                    @handleC
/scripts/06_*.py                    @handleC
/tests/fixtures/synth_features.py   @handleC
/tests/test_metrics.py              @handleC
/tests/test_leakage_checks.py       @handleC
/tests/test_models_normalisation.py @handleC
/tests/test_loso.py                 @handleC
/docs/baseline_report.md            @handleC
/results/tables/                    @handleC
/results/figures/                   @handleC
```

### 4.4 Sharing data between members

Data and caches are huge and git-ignored. Two mechanisms:

- **Everyone can regenerate** any artifact from scripts (`--patients chb01 chb02` limits the scope).
- **Shared drive folder** `eeg-seizure-loso-shared/` (Google Drive or similar) with two sub-folders: `preprocessed_slice/` (A uploads a zip of `data/interim/preprocessed/` plus both CSVs for 2-3 patients) and `features_slice/` (B uploads `data/processed/features/` for the same patients). A "slice" is the first real data other members test against.

---

## 5. Environment and commands

**Python 3.11.** Windows users: run `make` targets in WSL or Git Bash, or call the Python commands listed here directly.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                          # makes `import eegpipe` work everywhere
pytest -q                                                 # synthetic data only, must pass on a fresh clone
```

### Makefile targets (all pre-created at kickoff)

| Target | Runs | Owner |
|---|---|---|
| `make test` | `pytest -q -m "not integration and not needs_data"` | all |
| `make lint` | `ruff check src tests scripts` | all |
| `make download` | `python scripts/01_download_and_index.py` | A |
| `make preprocess` | `python scripts/02_preprocess.py` | A |
| `make windows` | `python scripts/03_make_windows.py` | B |
| `make features` | `python scripts/04_extract_features.py` | B |
| `make baseline` | `python scripts/05_run_baseline.py` | C |
| `make report` | `python scripts/06_make_report.py` | C |
| `make all` | download, preprocess, windows, features, baseline, report in order | joint |
| `make integration` | `pytest -q -m integration` (full chain on synthetic data) | joint |

Every script accepts `--config PATH` (default `configs/config.yaml`) and, where relevant, `--patients chb01 chb02 ...` to process a subset. Tests create a temporary config that points every path at a temp directory; that is how the integration test runs without touching real data.

**Pytest markers** (declared in `pyproject.toml`): `slow`, `integration`, `needs_data` (skipped in CI).

---
## 6. Configuration

`configs/config.yaml` is created at kickoff (Step 1) exactly as below and is then **frozen**. Access it only through `eegpipe.config.load_config()`.

```yaml
project:
  seed: 42
  n_jobs: -1                      # joblib workers (parallelism happens at case/file level only)

paths:                            # relative to repo root; tests override these with temp dirs
  raw: data/raw/chbmit
  interim: data/interim
  preprocessed: data/interim/preprocessed
  windows: data/processed/windows
  features: data/processed/features
  predictions: results/predictions
  tables: results/tables
  figures: results/figures
  logs: results/logs

dataset:
  name: chb-mit
  physionet_db: chbmit
  fs: 256
  patient_map: {chb21: chb01}     # same patient, recorded 1.5 years later
  channels:                       # 18 common bipolar channels, FIXED ORDER
    - FP1-F7
    - F7-T7
    - T7-P7
    - P7-O1
    - FP1-F3
    - F3-C3
    - C3-P3
    - P3-O1
    - FP2-F4
    - F4-C4
    - C4-P4
    - P4-O2
    - FP2-F8
    - F8-T8
    - T8-P8
    - P8-O2
    - FZ-CZ
    - CZ-PZ
  channel_aliases:                # raw EDF name -> canonical name (extend after inspecting real files)
    T8-P8-0: T8-P8
  max_seizure_free_hours_per_patient: 2.0   # includes each case's first (calibration) file

preprocessing:
  notch_hz: 60.0
  notch_q: 30.0
  bandpass_hz: [0.5, 45.0]
  fir:
    window: kaiser
    ripple_db: 60.0               # stop-band attenuation used with scipy.signal.kaiserord
    transition_hz: 1.0            # numtaps and beta are derived; expect roughly 900-1000 taps

segmentation:
  window_s: 4.0
  overlap: 0.5                    # step = window_s * (1 - overlap) = 2 s
  drop_boundary_windows: true

features:
  welch: {nperseg_s: 1.0, noverlap_frac: 0.5}
  bands: {delta: [0.5, 4.0], theta: [4.0, 8.0], alpha: [8.0, 13.0], beta: [13.0, 30.0], gamma: [30.0, 45.0]}
  entropy: {enabled: true, m: 2, r_factor: 0.2, decimate: 2}   # computed on the window decimated by 2 (128 Hz)
  wavelet: {name: db4, level: 5}
  dtype: float32
  catalog:                        # per-channel features, FIXED ORDER (32 per channel)
    time: [mean, std, skew, kurt, line_length, hjorth_activity, hjorth_mobility, hjorth_complexity]
    frequency: [logpow_delta, logpow_theta, logpow_alpha, logpow_beta, logpow_gamma,
                rel_delta, rel_theta, rel_alpha, rel_beta, rel_gamma]
    nonlinear: [sampen, apen]
    wavelet: [dwt_logE_d1, dwt_logE_d2, dwt_logE_d3, dwt_logE_d4, dwt_logE_d5, dwt_logE_a5,
              dwt_std_d1, dwt_std_d2, dwt_std_d3, dwt_std_d4, dwt_std_d5, dwt_std_a5]

evaluation:
  arms: [raw, subject_standardised]
  models: [svm, rf]
  calibration: {minutes: 10, min_windows: 60}        # Arm 1 (and calibration exclusion in both arms)
  train_sampling: {interictal_to_ictal_ratio: 5, max_train_windows: 30000}
  feature_selection: {method: f_classif, k: 100}     # SelectKBest inside the Pipeline
  tuning:
    mode: nested                  # nested | fixed  (fixed = fallback when compute is short)
    inner_splits: 3               # GroupKFold by patient inside the training patients
    scoring: roc_auc
  grids:
    svm: {C: [1, 10], gamma: [scale]}
    rf: {n_estimators: [200], max_depth: [null, 20], min_samples_leaf: [1, 5]}
  fixed_params:                   # used when tuning.mode == fixed
    svm: {C: 1, gamma: scale}
    rf: {n_estimators: 200, max_depth: null, min_samples_leaf: 1}
```

**Derived facts everyone can rely on:** 18 channels x 32 features = 576 feature columns (fewer if `entropy.enabled` is false, so downstream code must infer feature columns as those starting with `f_`, not hard-code 576). Step = 2 s = 512 samples. Window = 1024 samples.

`eegpipe.config.channel_slug(name)` returns `name.upper().replace("-", "_")` (for example `FP1-F7` becomes `FP1_F7`). It is the single definition used to build feature column names.

---

## 7. Interface contracts

All artifacts below are frozen. Producer and consumer test against these formats using synthetic data before real data exists.

### C1. `data/interim/annotations.csv` (producer A, consumer B)

One row per seizure. Files without seizures do not appear.

| Column | Type | Meaning |
|---|---|---|
| `patient` | str | after patient_map, for example `chb01` (also for chb21 rows) |
| `case` | str | original case id, for example `chb21` |
| `file` | str | for example `chb21_03.edf` |
| `seizure_idx` | int | 0-based index within the file |
| `seizure_start_s` | float | seconds from file start |
| `seizure_end_s` | float | seconds from file start |

### C2. `data/interim/file_index.csv` (producer A, consumers B and C)

One row per **scanned** EDF file.

| Column | Type | Meaning |
|---|---|---|
| `patient`, `case`, `file` | str | as in C1 |
| `file_order` | int | chronological rank within `patient` (sorted by case, then numeric file suffix) |
| `duration_s` | float | from the EDF header |
| `n_samples` | int | `round(duration_s * fs)` |
| `n_seizures` | int | rows for this file in C1 |
| `role` | str | `seizure` (n_seizures at least 1) or `seizure_free` |
| `include` | bool | selected into the cohort and loadable |
| `exclude_reason` | str | empty if included, otherwise for example `missing_channel:P7-O1`, `over_cap`, `no_seizure_patient` |

### C3. `data/interim/preprocessed/{case}/{file_stem}.npy` (producer A, consumer B)

- NumPy array, **float32, shape (18, n_samples)**, units uV, channel order = `dataset.channels`.
- Already notch- and band-pass-filtered and delay-compensated. Same length as the original recording.
- `file_stem` is the file name without `.edf`.

### C4. `data/processed/windows/{case}.parquet` (producer B, consumer B)

Sorted by (`file`, `start_sample`). Windows lie fully inside the file.

| Column | Type | Meaning |
|---|---|---|
| `patient`, `case`, `file` | str | |
| `start_sample` | int | first sample of the window |
| `t_start_s` | float | `start_sample / fs` |
| `label` | int8 | 0 interictal, 1 ictal. Boundary-straddling windows are not present |

### C5. `data/processed/features/{case}.parquet` (producer B, consumer C)

The C4 columns plus one float32 column per feature and channel, named `f_{feature}_{channel_slug}` (for example `f_hjorth_mobility_FP1_F7`, `f_logpow_alpha_CZ_PZ`). Column order: features in catalog order (outer loop), channels in config order (inner loop). **No NaN or inf** values are allowed in a delivered file.

Feature definitions (all computed per channel from one filtered window `x`, in uV, at fs = 256 Hz):

| Feature | Definition |
|---|---|
| `mean`, `std`, `skew`, `kurt` | sample mean, standard deviation (ddof=0), skewness, excess kurtosis |
| `line_length` | `sum(abs(diff(x)))` |
| `hjorth_activity` | `var(x)` |
| `hjorth_mobility` | `sqrt(var(dx) / var(x))` with `dx = diff(x)` |
| `hjorth_complexity` | `mobility(dx) / mobility(x)` |
| `logpow_{band}` | `log10(band power + 1e-12)`; Welch PSD (segment 1 s, 50 % overlap, Hann window) integrated over the band |
| `rel_{band}` | band power divided by total power in 0.5-45 Hz, in [0, 1] |
| `sampen`, `apen` | sample entropy and approximate entropy, m = 2, r = 0.2 x std of the (decimated) window |
| `dwt_logE_{d1..d5,a5}` | `log10(sum(coeff^2) + 1e-12)` per sub-band of a 5-level db4 DWT |
| `dwt_std_{d1..d5,a5}` | standard deviation of the coefficients in that sub-band |

### C6. `results/predictions/{arm}__{model}.parquet` (producer C)

Scored windows only (calibration windows excluded).

| Column | Type | Meaning |
|---|---|---|
| `patient`, `case`, `file` | str | |
| `start_sample` | int | |
| `y_true` | int8 | 0 or 1 |
| `y_score` | float | RF: probability of class 1. SVM: `decision_function` value |
| `y_pred` | int8 | the model's own decision rule (`predict`) |
| `arm`, `model` | str | for example `subject_standardised`, `svm` |

### C7. Metrics tables (producer C)

- `results/tables/per_patient_{arm}__{model}.csv`: `patient, n_windows, n_ictal, tp, fp, tn, fn, sensitivity, specificity, f1, auc, fa_per_hour`.
- `results/tables/summary_all.csv`: `arm, model, metric, mean, std, median, n_patients`.
- `fa_per_hour = fp / (n_windows * step_s / 3600)`. It is a window-based approximation computed on the included data only (seizure-free data is capped), so it is not a clinical false-alarm rate. State this in the report.

### Function signatures that cross member boundaries

```python
# A -> everyone
eegpipe.config.load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict
eegpipe.config.channel_slug(name: str) -> str
eegpipe.utils.paths  # helper functions returning Path objects for every artifact above

# A -> B
# (files C1, C2, C3 only; B never imports A's IO code)

# B -> C
eegpipe.features.extract.feature_names(cfg: dict) -> list[str]     # full column names, catalog order
# (file C5 only; C never imports B's feature code except in test_integration)

# Each feature group, batched: x has shape (..., n_samples); leading dims are free
# (single window: (n_ch, n_samples); batch: (n_win, n_ch, n_samples)).
# Returns {feature_name: ndarray with shape x.shape[:-1]}.
time_domain_features(x: np.ndarray) -> dict[str, np.ndarray]
frequency_features(x: np.ndarray, fs: float, cfg: dict) -> dict[str, np.ndarray]
nonlinear_features(x: np.ndarray, fs: float, cfg: dict) -> dict[str, np.ndarray]
wavelet_features(x: np.ndarray, cfg: dict) -> dict[str, np.ndarray]
```

---
## 8. Work table of contents

Tick the box in your own row when the step's tests pass and its PR is merged. Hour estimates are working hours; each track is about 11 hours, so the three tracks finish together.

### 8.1 Member A: data and preprocessing

| Done | Step | Files (create) | Depends on | Est. |
|---|---|---|---|---|
| [ ] | [Step 1: Repo bootstrap config and contracts](#step-1-repo-bootstrap-config-and-contracts) (joint kickoff, A executes) | whole scaffold, `config.py`, `utils/*`, `tests/fixtures/synth_signals.py` | nobody | 1.5 h |
| [ ] | [Step 2: Download and annotation parsing](#step-2-download-and-annotation-parsing) | `io/download.py`, `io/annotations.py`, `scripts/01_*.py`, `tests/test_annotations.py` | Step 1 | 2 h + download wait |
| [ ] | [Step 3: Loader channel selection and cohort](#step-3-loader-channel-selection-and-cohort) | `io/loader.py`, `io/cohort.py`, `tests/test_loader_cohort.py`, `docs/data_notes.md` | Step 2 (summaries) | 2 h |
| [ ] | [Step 4: Filter design and validation](#step-4-filter-design-and-validation) | `preprocessing/filters.py`, `tests/test_filters.py`, `docs/filter_validation.md` | Step 1 only | 3 h |
| [ ] | [Step 5: Preprocessing runner](#step-5-preprocessing-runner) | `preprocessing/pipeline.py`, `scripts/02_*.py`, `tests/test_preprocess_pipeline.py` | Steps 3 and 4 | 2.5 h |

### 8.2 Member B: windows and features

| Done | Step | Files (create) | Depends on | Est. |
|---|---|---|---|---|
| [ ] | [Step 6: Windowing and labelling](#step-6-windowing-and-labelling) | `segmentation/windows.py`, `scripts/03_*.py`, `tests/test_windows.py` | contracts C1-C3 (synthetic fixture) | 2.5 h |
| [ ] | [Step 7: Time and frequency features](#step-7-time-and-frequency-features) | `features/time_domain.py`, `features/frequency_domain.py`, `tests/test_time_frequency_features.py` | config only | 3 h |
| [ ] | [Step 8: Nonlinear and wavelet features](#step-8-nonlinear-and-wavelet-features) | `features/nonlinear.py`, `features/wavelet.py`, `tests/test_nonlinear_wavelet_features.py` | config only | 3.5 h |
| [ ] | [Step 9: Feature runner](#step-9-feature-runner) | `features/extract.py`, `scripts/04_*.py`, `tests/test_extract.py`, `docs/feature_catalog.md` | Steps 6, 7, 8; preprocessed slice from A | 2.5 h |

### 8.3 Member C: evaluation and baseline models

| Done | Step | Files (create) | Depends on | Est. |
|---|---|---|---|---|
| [ ] | [Step 10: LOSO harness metrics and leakage guards](#step-10-loso-harness-metrics-and-leakage-guards) | `evaluation/metrics.py`, `evaluation/leakage_checks.py`, `evaluation/loso.py` (loop only), `tests/fixtures/synth_features.py`, `tests/test_metrics.py`, `tests/test_leakage_checks.py` | contract C5 (own synthetic data) | 4 h |
| [ ] | [Step 11: Classifiers and nested tuning](#step-11-classifiers-and-nested-tuning) | `models/classifiers.py`, `evaluation/loso.py` (sampling and tuning), `tests/test_loso.py` | Step 10 | 3.5 h |
| [ ] | [Step 12: Arm 1 reporting and integration](#step-12-arm-1-reporting-and-integration) | `models/normalisation.py`, `evaluation/reporting.py`, `scripts/05_*.py`, `scripts/06_*.py`, `tests/test_models_normalisation.py`, `docs/baseline_report.md`; joint: `tests/test_integration.py` | Step 11; features slice from B | 3.5 h |

### 8.4 Sync points

| When (approx. working hour) | What happens | Who |
|---|---|---|
| 0 to 1.5 | Kickoff: A pushes the scaffold (Step 1). B and C meanwhile read the README, install the environment, and prepare locally | A pushes; B and C review the PR |
| about 8 to 10 | **First real slice.** A uploads preprocessed data for 2-3 patients (for example chb01, chb02) to the shared drive. B and C run their code on real data and report contract mismatches | A to B, C |
| about 18 to 20 | **Features ready.** B uploads the features slice. C runs the real baseline | B to C |
| final 3 h | **Integration.** `make all`, leakage review, README and report notes ([Section 12](#12-integration-and-definition-of-done)) | all |

### 8.5 Dependency picture

```text
A: Step 1 -> Step 2 -> Step 3 -----------\
   |                                      -> Step 5 -> (preprocessed .npy) ----> B: Step 9 (real run)
   '-> Step 4 -------------------------/
B: Step 6, Step 7, Step 8 (parallel, synthetic data) -> Step 9 -> (features .parquet) -> C: Step 12 (real run)
C: Step 10 -> Step 11 -> Step 12 (synthetic features until B delivers)
```

---

## 9. Member A: data and preprocessing

**You own:** everything under `src/eegpipe/{config.py, utils, io, preprocessing}`, `scripts/01_*`, `scripts/02_*`, `tests/fixtures/{__init__.py, synth_signals.py}`, your five test files, `docs/data_notes.md`, `docs/filter_validation.md`, `notebooks/A_*`.
**You produce:** contracts C1, C2, C3 and the shared scaffold.
**You consume:** nothing (you are upstream of everyone).

### Step 1: Repo bootstrap config and contracts

**Goal:** everyone can clone, install, and run `pytest` on synthetic data within the first 45-90 minutes, so B and C can work in parallel immediately.

**Deliverables (all in one PR, pushed as fast as possible):**

1. The complete folder tree from [Section 3](#3-repository-structure), including all empty `__init__.py` files and `.gitkeep` in `data/`, `results/`.
2. `configs/config.yaml` copied **exactly** from [Section 6](#6-configuration).
3. `pyproject.toml` (src layout, package `eegpipe`, Python at least 3.10, pytest markers `slow`, `integration`, `needs_data`, ruff settings), `requirements.txt` with the three blocks, `Makefile` with every target from Section 5, `.gitignore`, `CODEOWNERS`, PR template, `ci.yml` (Python 3.11, install requirements, ruff, `pytest -q -m "not integration and not needs_data"`).
4. `src/eegpipe/config.py` (fully working):
   - `load_config(path=None, overrides=None) -> dict`: reads YAML, applies `overrides` as a deep merge (tests use this to point paths at temp dirs), validates that the 18 channels exist and are unique.
   - `channel_slug(name) -> str`.
5. `src/eegpipe/utils/{logging_utils.py, seed.py, paths.py}`: `get_logger(name)`, `set_global_seed(seed)`, and path helpers such as `annotations_csv(cfg)`, `file_index_csv(cfg)`, `preprocessed_path(cfg, case, file_stem)`, `windows_path(cfg, case)`, `features_path(cfg, case)`, `predictions_path(cfg, arm, model)`.
6. **Stubs** with the exact signatures from Section 7 for every module owned by B and C, and for `tests/conftest.py`. Stubs raise `NotImplementedError`. This lets B and C import and test against agreed signatures from minute one.
7. `tests/fixtures/synth_signals.py` (see below). A first working version is needed by **hour 2**.
8. `tests/test_config.py`.

**`synth_signals.py` specification:**

```python
def make_synthetic_summary_text(kind: str = "v1") -> str
    # kind "v1": "Seizure Start Time: 2996 seconds" (single seizure per file)
    # kind "v2": "Seizure 1 Start Time: 1467 seconds" (numbered, multiple seizures)
    # Include File Name / File Start Time / File End Time / Number of Seizures in File lines.

def make_synthetic_raw_array(fs: int = 256, seconds: float = 30.0, tones_hz=(10.0, 60.0),
                             n_ch: int = 18, seed: int = 0) -> np.ndarray
    # (n_ch, n_samples) float32: sum of sinusoids at tones_hz plus small Gaussian noise. For filter tests.

def make_synthetic_project(root: Path, cfg: dict, n_patients: int = 4, files_per_patient: int = 3,
                           minutes_per_file: float = 3.0, seed: int = 0) -> dict
    # Writes into `root` in exact contract layout: annotations.csv (C1), file_index.csv (C2),
    # preprocessed/{case}/{stem}.npy (C3, already "preprocessed", skips raw EDF).
    # Requirements:
    #   - patient-specific spectral colour (AR(1) coefficient and gain vary by patient) to mimic inter-subject variability
    #   - 1-2 seizures per patient, 10-20 s, modelled as ~3 Hz spike-wave-like rhythm with harmonics, amplitude x3-x5
    #   - one patient consists of TWO cases (like chb01 and chb21) to test patient grouping
    #   - each patient's first file is seizure-free (calibration)
    # Returns dict of created paths.
```

**Done when:** a fresh clone does `pip install -r requirements.txt && pip install -e . && pytest -q` successfully; CI is green; B and C confirm they can import their stubs.

**Understand (viva):** why LOSO groups by `patient`; why contracts are frozen; what `overrides` in `load_config` is used for.

### Step 2: Download and annotation parsing

**Goal:** get the data and produce C1 (`annotations.csv`). **Start the download immediately** because it is the slowest job in the whole project.

**Files:** `io/download.py`, `io/annotations.py`, `scripts/01_download_and_index.py`, `tests/test_annotations.py`.

**Specification:**

- `download.py`
  - `download_metadata(cfg) -> None`: fetch `RECORDS`, `RECORDS-WITH-SEIZURES`, and every `chbXX-summary.txt` from PhysioNet database `chbmit` (small files). Use `wfdb.dl_files(db, dl_dir, files, keep_subdirs=True, overwrite=False)`; fall back to direct HTTPS from the PhysioNet `chbmit/1.0.0/` file tree if wfdb fails.
  - `download_edfs(cfg, files: list[str]) -> None`: fetch only the listed EDF files. Skip files already present (resumable). Log failures and retry a few times.
- `annotations.py`
  - `parse_summary_file(path: Path) -> list[dict]`: returns one dict per file with `file`, `start_time`, `end_time`, `seizures: list[(start_s, end_s)]`. Must handle **both formats** (`Seizure Start Time: N seconds` and `Seizure k Start Time: N seconds`) and files with zero, one, or many seizures. Use regexes; ignore the "Channels in EDF Files" block.
  - `build_annotations(raw_dir: Path, patient_map: dict) -> pd.DataFrame`: concatenates all summaries into C1. Applies `patient_map` (`chb21` rows get `patient = chb01`, `case` stays `chb21`).
- `scripts/01_download_and_index.py` flow: (1) download metadata, (2) build and save `annotations.csv`, (3) *(after Step 3 exists)* select cohort from metadata, (4) download only cohort EDFs, (5) build `file_index.csv`. Until Step 3 lands, the script stops after (2).

**Tests (`test_annotations.py`):** parse both synthetic summary formats; zero-seizure file; multi-seizure file; `chb21` rows map to patient `chb01`; row counts match hand-counted seizures; seizure end greater than start.

**Done when:** `annotations.csv` exists for the real data and its per-patient seizure counts match a manual count from two or three summary files. Record the total seizure count in `docs/data_notes.md`.

**Understand (viva):** why files are downloaded selectively; the two summary formats.

### Step 3: Loader channel selection and cohort

**Goal:** load any included EDF to a `(18, n_samples)` float32 array in uV, and decide which files are in the cohort.

**Files:** `io/loader.py`, `io/cohort.py`, `tests/test_loader_cohort.py`, `docs/data_notes.md`.

**Specification:**

- `loader.py`
  - `class ChannelMissingError(Exception)`.
  - `load_edf_channels(edf_path: Path, channels: list[str], aliases: dict, fs_expected: int = 256) -> tuple[np.ndarray, float]`: read with `mne.io.read_raw_edf(preload=True, verbose=False)`, resolve aliases (for example `T8-P8-0` to `T8-P8`), pick channels **by name in the configured order**, convert volts to **uV**, return `float32` array and the sampling rate. Raise `ChannelMissingError` naming the missing channel. Placeholder channels named `-` and duplicated names must not break loading. Fail loudly if fs differs from 256.
  - `read_edf_header(edf_path) -> dict`: duration and channel names without preloading (used for indexing).
- `cohort.py`
  - `build_file_index(cfg, annotations: pd.DataFrame) -> pd.DataFrame`: scan all EDFs listed in `RECORDS`, fill C2 (`file_order` from the numeric suffix, robust to lettered series such as `chb17a_*`; durations from headers where downloaded, otherwise from summary start/end times with midnight wrap).
  - `select_cohort(index: pd.DataFrame, cfg: dict) -> pd.DataFrame`: sets `include` and `exclude_reason` by these rules, in order: (1) every `seizure` file is included; (2) for every case, its first file is included as the calibration file even if seizure-free; (3) additional `seizure_free` files are added deterministically (seeded, evenly spaced by `file_order`) until `max_seizure_free_hours_per_patient` is reached, counting hours already included; (4) files missing any of the 18 channels get `include = False` with `missing_channel:<name>` and are logged in `docs/data_notes.md`; (5) patients with no seizure at all are excluded (`no_seizure_patient`).
- Update step (5) of `scripts/01_download_and_index.py` to run cohort selection, download only included EDFs, and save `file_index.csv`.

**Tests (`test_loader_cohort.py`):** cohort rules on a synthetic index (cap respected; first file of each case present; chb21 grouped with chb01; deterministic across runs with the same seed); loader test on a tiny synthetic EDF, skipped if EDF writing (`edfio`) is unavailable; alias resolution; missing channel raises `ChannelMissingError`.

**Done when:** every included real file loads to shape `(18, n_samples)` with sensible amplitude (roughly tens of uV, not volts) and all problem files are logged. Record expected data volume (hours and GB) in `docs/data_notes.md`. As a rough guide, a preprocessed hour is about 66 MB, so several hundred hours mean tens of GB of cache; check disk space.

**Understand (viva):** why channels are selected by name; why chb01 and chb21 are one patient; what the cohort cap trades away.

### Step 4: Filter design and validation

**Goal:** the general filter of the baseline: 60 Hz notch plus 0.5-45 Hz linear-phase Kaiser FIR band-pass, validated. This step needs no real data and can run in parallel with Steps 2-3.

**Files:** `preprocessing/filters.py`, `tests/test_filters.py`, `docs/filter_validation.md`.

**Specification:**

```python
def design_bandpass_fir(fs: float, low: float, high: float, ripple_db: float, transition_hz: float) -> np.ndarray
    # numtaps, beta = scipy.signal.kaiserord(ripple_db, transition_hz / (fs / 2)); force numtaps odd;
    # h = scipy.signal.firwin(numtaps, [low, high], window=("kaiser", beta), pass_zero=False, fs=fs)

def design_notch(fs: float, f0: float, q: float) -> tuple[np.ndarray, np.ndarray]
    # scipy.signal.iirnotch(f0, q, fs)

def apply_filters(x: np.ndarray, fs: float, cfg: dict) -> np.ndarray
    # x: (n_ch, n_samples) float32 -> same shape, float32.
    # 1) notch, zero-phase (filtfilt along axis=-1)
    # 2) FIR band-pass, delay-compensated: reflect-pad by numtaps//2, scipy.signal.oaconvolve(..., mode="same"), trim
    # Output length must equal input length. No time shift relative to the input.

def freq_response(h: np.ndarray, fs: float, n: int = 8192) -> tuple[np.ndarray, np.ndarray]
    # (frequencies in Hz, magnitude in dB)
```

Notes: with a 0.5 Hz lower edge the filter is long (roughly 900-1000 taps, about 2 s group delay). This is fine offline; note it in the report as a wearable-deployment cost. The band-pass already attenuates 60 Hz, so the notch is redundant but retained to match the synopsis.

**Tests (`test_filters.py`):** on `make_synthetic_raw_array` (10 Hz and 60 Hz tones): 60 Hz attenuated by at least 40 dB (measure with Welch), 10 Hz amplitude preserved within 0.5 dB; DC removed; output shape and dtype preserved; zero delay (cross-correlation peak at lag 0); frequency response passband ripple small and stop-band above 46 Hz at or below about -60 dB.

**Validation artifact:** plot magnitude and phase response and save `docs/figures/A_fir_response.png`; summarise numbers in `docs/filter_validation.md`. If MATLAB is available, run one `freqz` cross-check of the same coefficients and note the agreement; skip otherwise.

**Optional if time remains:** compare with a 4th-order Butterworth band-pass in the same document (literature disagrees on FIR vs IIR).

**Done when:** all filter tests pass and the response figure exists.

**Understand (viva):** why FIR and linear phase for EEG; what the Kaiser beta and ripple control; why reflect-padding at the edges.

### Step 5: Preprocessing runner

**Goal:** turn every included EDF into a cached C3 array, in parallel, resumable.

**Files:** `preprocessing/pipeline.py`, `scripts/02_preprocess.py`, `tests/test_preprocess_pipeline.py`.

**Specification:**

- `preprocess_file(row: pd.Series, cfg: dict, overwrite: bool = False) -> dict`: loads the EDF (Step 3 loader), applies `apply_filters`, saves `.npy` at `preprocessed_path(cfg, case, file_stem)`, returns `{file, status, n_samples, seconds}`. Skips existing output unless `overwrite`. Catches and reports errors instead of crashing the batch.
- `preprocess_all(file_index: pd.DataFrame, cfg: dict, patients: list[str] | None = None, n_jobs: int | None = None, overwrite: bool = False) -> pd.DataFrame`: joblib parallel over included files (one process per file; no nested parallelism), progress bar, writes `results/logs/preprocess_status.csv`.
- `scripts/02_preprocess.py --patients chb01 chb02` supports the slice used at the first sync point.
- Sanity check per file: no NaN/inf, shape `(18, n_samples)`, output length equals `n_samples` in the index, amplitude not degenerate (flat channels are counted and logged).

**Tests (`test_preprocess_pipeline.py`):** run on the synthetic project (feed synthetic "raw" arrays through a monkeypatched loader), check shape, dtype, resumability (second run skips), error isolation (one bad file does not stop the batch).

**Done when:** `--patients chb01 chb02` finishes; A uploads the slice (`preprocessed_slice/`) for B and C. Then run the full cohort in the background (`make preprocess`) while helping B and C.

**Understand (viva):** the cost of caching versus recomputing; how parallelism is organised; what the status log tells you.

---
## 10. Member B: windows and features

**You own:** `src/eegpipe/segmentation/*`, `src/eegpipe/features/*`, `scripts/03_*`, `scripts/04_*`, your four test files, `docs/feature_catalog.md`, `notebooks/B_*`.
**You consume:** contracts C1, C2, C3 (annotations, file index, preprocessed arrays), plus `load_config` and `channel_slug` from A's scaffold. Until A's real slice exists, use `tests/fixtures/synth_signals.py::make_synthetic_project`.
**You produce:** contracts C4 (windows) and C5 (features), and `feature_names(cfg)`.

### Step 6: Windowing and labelling

**Goal:** convert each recording into labelled 4 s windows.

**Files:** `segmentation/windows.py`, `scripts/03_make_windows.py`, `tests/test_windows.py`.

**Specification:**

```python
def seizure_intervals_for_file(annotations: pd.DataFrame, file: str) -> list[tuple[float, float]]
    # [(start_s, end_s), ...] for that file, from C1.

def make_windows(n_samples: int, fs: float, window_s: float, overlap: float,
                 seizure_intervals: list[tuple[float, float]],
                 drop_boundary: bool = True) -> pd.DataFrame
    # Returns columns: start_sample (int), t_start_s (float), label (int8).
    # W = round(window_s * fs); step = round(W * (1 - overlap)).
    # Number of windows = floor((n_samples - W) / step) + 1 (last incomplete window is dropped).
    # Seizure [on, off) in samples: on_s = round(on * fs), off_s = round(off * fs).
    # A window [s, s + W): fully inside a seizure (s >= on_s and s + W <= off_s) -> label 1;
    # no overlap with any seizure -> label 0; partial overlap -> DROPPED (when drop_boundary is True).

def build_window_table(file_index: pd.DataFrame, annotations: pd.DataFrame, cfg: dict,
                       patients: list[str] | None = None) -> pd.DataFrame
    # For every included file: read n_samples from the preprocessed .npy header (np.load(mmap_mode="r")),
    # call make_windows, add patient/case/file columns. Sorted by (file, start_sample).
```

- `scripts/03_make_windows.py` writes one parquet per case (`data/processed/windows/{case}.parquet`), supports `--patients`, and prints a summary (windows per patient, ictal fraction). Expect the ictal fraction to be small (well under a few percent).
- Warn (log) when a seizure is shorter than `window_s` (no ictal window can exist) and when a patient ends up with zero ictal windows.

**Tests (`test_windows.py`) with hand-computed expectations:** window count formula on several lengths; a single seizure with known onset/offset gives the expected number of ictal windows; straddling windows are absent; two seizures in one file; seizure at the very start and at the very end of a file; a window that spans the gap between two seizures is dropped; `drop_boundary=False` behaviour; output dtypes and sorting.

**Done when:** window counts and labels on the synthetic project equal hand calculations, and the script runs on the real slice from A.

**Understand (viva):** why partial windows are dropped; how overlap affects the number of ictal windows and the correlation between neighbouring samples.

### Step 7: Time and frequency features

**Goal:** the first two feature groups, batched and vectorised.

**Files:** `features/time_domain.py`, `features/frequency_domain.py`, `tests/test_time_frequency_features.py`.

**Specification (definitions in Section 7, C5):**

```python
def time_domain_features(x: np.ndarray) -> dict[str, np.ndarray]
    # x: (..., n_samples). Returns keys: mean, std, skew, kurt, line_length,
    # hjorth_activity, hjorth_mobility, hjorth_complexity; each shaped x.shape[:-1].
    # Use axis=-1 operations; guard divisions (var == 0 -> feature = 0.0, not NaN).

def frequency_features(x: np.ndarray, fs: float, cfg: dict) -> dict[str, np.ndarray]
    # scipy.signal.welch(x, fs, nperseg=round(nperseg_s * fs), noverlap=round(nperseg * noverlap_frac), axis=-1)
    # For each band in cfg["features"]["bands"]: absolute power = integral of PSD over the band (trapezoid).
    # Keys: logpow_{band} = log10(power + 1e-12); rel_{band} = power / total power over 0.5-45 Hz.
```

- Everything float32 on output. Do not loop over windows in Python; the same function must accept one window `(n_ch, n_samples)` and a batch `(n_win, n_ch, n_samples)`.

**Tests:** 10 Hz sine gives `logpow_alpha` dominant and `rel_alpha` close to 1; 2 Hz sine gives delta dominance; white noise gives roughly flat spectrum (relative powers proportional to band widths); constant signal gives zero Hjorth activity without NaN; line length of a known ramp; Hjorth values of a sine match the analytic result; batch call equals per-window loop.

**Done when:** tests pass and shapes are correct for both single-window and batched input.

**Understand (viva):** what Hjorth mobility and complexity measure; why log band power is used; what Welch averaging does.

### Step 8: Nonlinear and wavelet features

**Goal:** the expensive and the time-frequency features.

**Files:** `features/nonlinear.py`, `features/wavelet.py`, `tests/test_nonlinear_wavelet_features.py`.

**Specification:**

```python
def sample_entropy(x: np.ndarray, m: int, r: float) -> float      # numba @njit(cache=True), 1-D input
def approximate_entropy(x: np.ndarray, m: int, r: float) -> float  # numba @njit(cache=True), 1-D input

def nonlinear_features(x: np.ndarray, fs: float, cfg: dict) -> dict[str, np.ndarray]
    # Flatten leading dims, decimate each window by cfg["features"]["entropy"]["decimate"]
    # (scipy.signal.decimate or plain slicing after the band-limiting already done, document the choice),
    # r = r_factor * std(window), call the numba functions in a loop over flattened rows,
    # reshape back. Keys: sampen, apen.

def wavelet_features(x: np.ndarray, cfg: dict) -> dict[str, np.ndarray]
    # coeffs = pywt.wavedec(x, wavelet, level=5, axis=-1)  -> [a5, d5, d4, d3, d2, d1]
    # Keys: dwt_logE_{d1..d5,a5} = log10(sum(c**2) + 1e-12); dwt_std_{d1..d5,a5} = std(c).
```

- Sample entropy is O(N^2) per window per channel; decimating by 2 (512 samples) reduces cost roughly fourfold. Use `@njit(cache=True)` only. **Do not use numba parallelism**: parallelism is at case level in Step 9 (avoid nested parallelism).
- Sample entropy is undefined when no template matches: return `np.nan` internally; Step 9 handles replacement. Constant windows (std = 0) also yield `nan`.
- At fs = 256 Hz the 5-level db4 sub-bands are approximately: d1 64-128 Hz, d2 32-64, d3 16-32, d4 8-16, d5 4-8, a5 0-4 Hz. d1 is mostly empty because of the 45 Hz band-pass; keep it anyway for a fixed catalog.
- If `cfg["features"]["entropy"]["enabled"]` is false, `nonlinear_features` returns an empty dict (this is the first fallback in Section 12).

**Tests:** white noise has higher sample entropy than a sine wave; a constant signal returns `nan` without crashing; numba result equals a slow pure-Python reference on a short random signal (tolerance 1e-6); DWT log-energy peaks in the expected sub-band for tones at 6 Hz (d5), 12 Hz (d4), 24 Hz (d3); batched shapes; time a batch of 100 windows x 18 channels and log the per-window cost (informational, not asserted).

**Done when:** tests pass and you can state the measured entropy cost per window, which decides the full-run time in Step 9.

**Understand (viva):** why entropy drops during seizures (more regular, synchronised activity); what `m` and `r` mean; why the DWT sub-bands map to clinical rhythms.

### Step 9: Feature runner

**Goal:** produce the C5 feature parquet for every case and give C the real feature slice.

**Files:** `features/extract.py`, `scripts/04_extract_features.py`, `tests/test_extract.py`, `docs/feature_catalog.md`.

**Specification:**

```python
def feature_names(cfg: dict) -> list[str]
    # ["f_{feature}_{channel_slug}", ...]: catalog order outer loop, channel order inner loop.
    # If entropy is disabled, nonlinear features are omitted.

def extract_window_features(win: np.ndarray, fs: float, cfg: dict) -> np.ndarray
    # win: (n_ch, n_samples) -> 1-D float32 vector ordered exactly as feature_names(cfg).

def extract_batch_features(batch: np.ndarray, fs: float, cfg: dict) -> np.ndarray
    # batch: (n_win, n_ch, n_samples) -> (n_win, n_features) float32, same order. Calls the four group functions once each.

def extract_case_features(case: str, file_index: pd.DataFrame, window_table: pd.DataFrame,
                          cfg: dict, batch_size: int = 256) -> pd.DataFrame
    # For each file of the case: np.load (mmap), slice windows by start_sample in batches, extract, concatenate.
    # Returns C4 key columns + feature columns (Contract C5).
```

- Non-finite handling: convert any non-finite value to `0.0`, count replacements per feature, log them, and **warn if more than 0.1 % of values were replaced**.
- `scripts/04_extract_features.py` runs cases in parallel with joblib (one process per case), supports `--patients`, writes `data/processed/features/{case}.parquet` (pyarrow, float32), and logs time per case.
- `docs/feature_catalog.md`: one table with every feature, formula, unit, and expected behaviour during seizure versus interictal (fill in after inspecting real data).

**Tests (`test_extract.py`):** `feature_names` length equals 18 x 32 = 576 and is unique; column order matches the vector order; batch equals per-window; the output on the synthetic project has no NaN/inf and includes key columns; ictal windows have visibly larger line length or lower entropy than interictal on synthetic seizures (sanity).

**Done when:** features exist for the real slice (2-3 patients) and B has uploaded them to `features_slice/`; then run the full cohort in the background and monitor time. Report the measured runtime to the team.

**Understand (viva):** why features are batched and per-window only (leakage rule L6); how non-finite values are handled and why that is acceptable at the 0.1 % level.

---
## 11. Member C: evaluation and baseline models

**You own:** `src/eegpipe/{models, evaluation}/*`, `scripts/05_*`, `scripts/06_*`, `tests/fixtures/synth_features.py`, your four test files, `docs/baseline_report.md`, `results/tables/`, `results/figures/`, `notebooks/C_*`. You also lead the joint `tests/test_integration.py`.
**You consume:** contract C5 (feature parquet files). Until B's real slice exists, use your own `synth_features.py`.
**You produce:** contracts C6 and C7, and the final baseline numbers.
**You are the guardian of the leakage rules L1-L8.** Every check in Step 10 protects the credibility of the whole project.

### Step 10: LOSO harness metrics and leakage guards

**Goal:** a correct LOSO loop, correct metrics, and automated leakage tests, all verified on synthetic data before real features exist.

**Files:** `evaluation/metrics.py`, `evaluation/leakage_checks.py`, `evaluation/loso.py` (loop only in this step), `tests/fixtures/synth_features.py`, `tests/test_metrics.py`, `tests/test_leakage_checks.py`.

**`synth_features.py` specification:**

```python
def make_synthetic_features(cfg: dict, n_patients: int = 8, windows_per_patient: int = 600,
                            ictal_fraction: float = 0.05, effect: float = 1.0,
                            patient_shift: float = 1.0, n_informative: int = 20,
                            seed: int = 0) -> pd.DataFrame
    # Returns a Contract C5 DataFrame: keys (patient, case, file, start_sample, t_start_s, label)
    # plus float32 columns f_{feature}_{channel_slug} built from cfg["features"]["catalog"] and channels
    # (build the names locally; do NOT import B's code).
    # - every patient gets its own additive offset and scale per feature (inter-subject variability, size = patient_shift)
    # - ictal windows shift `n_informative` features by `effect`
    # - the first 350 windows (chronological) of every patient are interictal, so calibration is possible
    # - one patient has two cases (like chb01 and chb21) to test grouping
    # - effect=0 -> no signal (AUC about 0.5); patient_shift=0 -> no subject variability
```

**Specification:**

- `metrics.py`
  - `compute_metrics(y_true, y_score, y_pred, step_s: float) -> dict`: returns `n_windows, n_ictal, tp, fp, tn, fn, sensitivity, specificity, f1, auc, fa_per_hour` (`fa_per_hour = fp / (n_windows * step_s / 3600)`). If a class is absent, set undefined metrics to `nan` (never crash).
  - `aggregate_metrics(per_patient: pd.DataFrame) -> pd.DataFrame`: mean, std, median, `n_patients` per metric (ignore NaN, report how many were ignored).
- `leakage_checks.py`
  - `assert_patient_disjoint(train_patients, test_patients)` (L1).
  - `assert_patient_grouping(df, patient_map)`: no value in `patient` equals a key of `patient_map` (for example `chb21`), and each `case` maps to exactly one `patient` (L1).
  - `assert_finite_features(df, feature_cols)`.
  - `assert_no_calibration_scored(pred_df, calibration_keys)`.
  - `label_shuffle_check(df, feature_cols, model_name, cfg, n_test_patients: int = 3, seed: int = 0) -> float`: shuffles **training labels only**, runs LOSO on a few held-out patients, returns mean AUC (L8). Expect roughly 0.5 (test asserts within 0.35-0.65).
- `loso.py` (loop; tuning and sampling are added in Step 11)
  - `run_loso(df, feature_cols, model_name, cfg, arm, patients=None, shuffle_train_labels=False) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(predictions C6, per_patient C7)`.
  - Loop: for each held-out `patient`: `train = df[df.patient != held]`, `test = df[(df.patient == held) & ~df.is_calibration]` (treat a missing `is_calibration` column as all False); call `assert_patient_disjoint`; fit; score the test patient at its **natural** class balance (L3); collect predictions and metrics. In this step use a fixed default model so the loop can be tested end to end.
  - Feature columns are always inferred as `[c for c in df.columns if c.startswith("f_")]`.

**Tests:** `test_metrics.py` with hand-computed confusion matrices, perfect classifier, constant classifier, and single-class test patient (NaN, no crash). `test_leakage_checks.py`: disjointness assertion fires when patients overlap; grouping assertion fires on `chb21`; label-shuffle AUC within 0.35-0.65 on synthetic data; loop returns one row per held-out patient; held-out patient's class ratio in the test set equals its natural ratio.

**Done when:** `run_loso` works end to end on synthetic features with the default model and all guards pass.

**Understand (viva):** why per-patient metrics matter more than pooled accuracy; why accuracy is misleading at this class ratio; what the label-shuffle test proves.

### Step 11: Classifiers and nested tuning

**Goal:** SVM and Random Forest with leakage-free preprocessing, class-imbalance handling, and nested tuning.

**Files:** `models/classifiers.py`, `evaluation/loso.py` (sampling and tuning added), `tests/test_loso.py`.

**Specification:**

```python
def make_model(name: str, cfg: dict, n_features: int, seed: int) -> tuple[Pipeline, dict]
    # name in {"svm", "rf"}.
    # Pipeline([("scale", StandardScaler()),
    #           ("select", SelectKBest(f_classif, k=min(k, n_features))),
    #           ("clf", SVC(kernel="rbf", class_weight="balanced", cache_size=1000, random_state=seed)   # svm
    #                   or RandomForestClassifier(class_weight="balanced_subsample", random_state=seed))]) # rf
    # Returns (pipeline, param_grid) with grid keys prefixed "clf__" from cfg["evaluation"]["grids"][name].

def subsample_training(df: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame
    # Applies ONLY to training data (L3). Keep all ictal windows; sample interictal at
    # interictal_to_ictal_ratio, drawn proportionally per patient. If the total would exceed
    # max_train_windows, scale both classes down keeping the ratio. Every patient keeps at least one ictal window.

def tune_model(pipe: Pipeline, grid: dict, X, y, groups, cfg: dict, seed: int)
    # tuning.mode == "nested": GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=min(inner_splits, n_groups)),
    #                          scoring=cfg scoring, refit=True), groups=patient ids of the training windows (L4).
    # tuning.mode == "fixed": set cfg["evaluation"]["fixed_params"][name] on the pipeline and fit once.
    # Returns the fitted estimator. The scaler and selector live inside the pipeline so they are refit per fold (L2).
```

- y_score: Random Forest uses `predict_proba(X)[:, 1]`; SVM uses `decision_function(X)`. `y_pred` is `predict(X)`. Do not tune a threshold on test data.
- Runtime warning: RBF SVM cost grows steeply with training size. This is why `max_train_windows` (30000) and `k` (100) exist. Log the time per fold. If nested tuning is too slow, use `tuning.mode: fixed` (Section 12 fallback ladder).

**Tests (`test_loso.py`):** `subsample_training` respects the ratio and cap, only touches training data, keeps at least one ictal window per patient; with `effect=0` LOSO AUC is about 0.5; with `effect` large and `patient_shift=0` LOSO AUC is high (above 0.9); with large `patient_shift` the raw arm degrades (used again in Step 12); SVM and RF both run end to end; nested tuning uses only training patients (assert the held-out patient never appears in the tuning groups); results are identical across two runs with the same seed.

**Done when:** both models run in both tuning modes on synthetic data and the tests pass.

**Understand (viva):** why the scaler and selector are inside the pipeline; why hyper-parameters are tuned by patient groups; what class weighting does compared with resampling.

### Step 12: Arm 1 reporting and integration

**Goal:** the two experimental arms, the results tables and figures, and the joint integration on real data.

**Files:** `models/normalisation.py`, `evaluation/reporting.py`, `scripts/05_run_baseline.py`, `scripts/06_make_report.py`, `tests/test_models_normalisation.py`, `docs/baseline_report.md`; joint `tests/test_integration.py`.

**Specification:**

```python
def mark_calibration(df: pd.DataFrame, cfg: dict) -> pd.DataFrame
    # Adds bool column `is_calibration`. Per patient: sort by (case, file, start_sample); find the first
    # ictal window; among the interictal windows BEFORE it, mark the first n = ceil(minutes * 60 / step_s)
    # as calibration (step_s = window_s * (1 - overlap); default n = 300). If fewer than min_windows are
    # available, fall back to the first n interictal windows overall, and log a warning.

def standardise_per_patient(df: pd.DataFrame, feature_cols: list[str], eps: float = 1e-8) -> pd.DataFrame
    # For each patient: mean and std computed from THAT patient's calibration windows only; apply
    # (x - mean) / max(std, eps) to ALL windows of that patient. No other patient is involved (L5).

def apply_arm(df: pd.DataFrame, arm: str, cfg: dict) -> pd.DataFrame
    # arm "raw": mark_calibration only (features unchanged).
    # arm "subject_standardised": mark_calibration then standardise_per_patient.
    # Both arms return the same rows and the same is_calibration mask, so both are scored on identical windows.
```

- Calibration windows stay in training data for the *other* patients' folds (they are legitimate labelled interictal samples) but are **never scored** for the held-out patient, in either arm.
- `reporting.py`: `save_predictions(pred_df, arm, model, cfg)` (C6), `save_tables(per_patient_df, arm, model, cfg)` (C7), `make_summary_table(cfg) -> DataFrame` (writes `summary_all.csv`), `plot_per_patient_bars(per_patient_by_arm, metric, out_path)` (sensitivity, specificity, AUC per patient for raw versus standardised).
- `scripts/05_run_baseline.py` flags: `--arms raw subject_standardised`, `--models svm rf`, `--patients ...` (subset), `--max-patients N`, `--tuning nested|fixed`, `--synthetic` (use `synth_features` instead of parquet files). Loads all feature parquets, checks `assert_patient_grouping` and `assert_finite_features`, runs each arm x model, saves C6 and C7.
- `scripts/06_make_report.py` builds `summary_all.csv` and the figures from saved results.
- `docs/baseline_report.md`: results table (mean, SD, median per arm and model), per-patient plots, observations about between-patient variation, and an honest **limitations** list (seizure-free data capped so false alarms per hour are not clinical rates; window-level scoring not event-level; few seizures in some patients so metrics are noisy; calibration windows assumed seizure-free, which is checked here only via labels, not in deployment).

**Joint integration test (`tests/test_integration.py`, marker `integration`):** create a temp config with `load_config(overrides=...)`, run `make_synthetic_project`, then scripts 03, 04, 05, 06 through their `main(argv)` functions, and assert that every contract file exists with the right columns, no NaN/inf in features, and metrics files are produced. A, B and C review it together.

**Tests (`test_models_normalisation.py`):** calibration mask marks exactly the first 300 interictal windows before the first ictal window; identical masks across arms; per-patient standardised calibration windows have mean about 0 and std about 1; another patient's data never changes a patient's statistics; on synthetic data with strong `patient_shift`, mean AUC of `subject_standardised` exceeds `raw` by a clear margin; fallback path logs a warning.

**Done when:** `make all` runs from raw data to the baseline table on the real slice, then on the full cohort; the report draft exists.

**Understand (viva):** why calibration windows are excluded from scoring in both arms; what arm 1 does and does not fix; why we compare against arm 1 and not only arm 0 (a simple per-patient standardisation baseline is what adaptive filtering must beat).

---

## 12. Integration and definition of done

### 12.1 Integration procedure (final ~3 hours, all three)

1. Merge everything to `main`; CI green; `pytest -q` and `make integration` pass on a fresh clone.
2. Run `make all` on the real slice, then on the full cohort (background).
3. **Leakage review, together, with the code open:**
   - grep for `train_test_split`, `fit`, `fit_transform`, and `StandardScaler` usage: fitting only inside `Pipeline` on training data (L2);
   - confirm `patient` (not `case`) is used for all grouping and that no `chb21` appears in a `patient` column (L1);
   - confirm the held-out patient is never resampled (L3) and tuning uses patient groups only (L4);
   - confirm arm 1 statistics use only the patient's own calibration windows (L5);
   - run `label_shuffle_check` on real features: AUC near 0.5 (L8);
   - if any LOSO result is near-perfect (for example AUC above 0.98 for most patients), treat it as a bug and investigate before reporting.
4. Record deviations from this README (for example a smaller cohort, entropy disabled) in `docs/data_notes.md` and `docs/baseline_report.md`.
5. Tag the commit: `git tag milestone-1 && git push --tags`.

### 12.2 Definition of done

Milestone 1 is complete when **all** of the following hold:

- [ ] Fresh clone: `pip install -r requirements.txt && pip install -e . && pytest -q` passes.
- [ ] `make integration` passes on synthetic data (full chain 03 to 06).
- [ ] Real data: `annotations.csv`, `file_index.csv`, preprocessed arrays, window tables, and feature tables exist for the cohort (or a documented subset).
- [ ] Filter validation figure and `docs/filter_validation.md` exist.
- [ ] `results/tables/per_patient_*.csv` and `summary_all.csv` exist for both arms and both models.
- [ ] The label-shuffle sanity check gives AUC near 0.5 on real features.
- [ ] Every member can explain their own modules and the leakage rules (Section "Understand" in each step).
- [ ] `docs/baseline_report.md` lists results and limitations. Commit tagged `milestone-1`.

### 12.3 Fallback ladder (use in this order if time or compute runs short)

1. Set `features.entropy.enabled: false` (drops sample and approximate entropy; the 32 features become 30 per channel). Re-add later if time remains.
2. Lower `max_seizure_free_hours_per_patient` (for example 0.5) and re-run windows and features.
3. Set `evaluation.tuning.mode: fixed` (no nested grid search).
4. Reduce `train_sampling.max_train_windows` (for example 15000) and `feature_selection.k` (for example 50).
5. Run the baseline on a subset of patients (`--patients` or `--max-patients`) and state this clearly in the report.

Never skip: the LOSO split by patient, training-only fitting, natural-class-balance scoring of the held-out patient, calibration-window exclusion, and the leakage tests.

### 12.4 Hand-off to Milestone 2

`src/eegpipe/adaptive/` is reserved. The adaptive filter will read C3-format arrays and write C3-format arrays to a different directory. Scripts 03 and 04 read the preprocessed directory from `cfg["paths"]["preprocessed"]`, so Milestone 2 can switch inputs through config overrides without editing Milestone 1 code. Milestone 2 compares three arms on identical windows and folds: `raw`, `subject_standardised`, and `adaptive`.

---

## Appendix A: CHB-MIT dataset notes

Verify every item below against the real files during Steps 2 and 3. Log surprises in `docs/data_notes.md`.

- **Layout:** `chbXX/chbXX_YY.edf` files, roughly one hour each (some longer), plus one `chbXX-summary.txt` per case, plus `RECORDS` and `RECORDS-WITH-SEIZURES` at the database root. Case ids run `chb01` to `chb24`. Some cases have lettered sub-series in file names (for example `chb17a_*`), so parse the numeric suffix robustly.
- **Patients versus cases:** `chb21` is the same patient as `chb01`, recorded 1.5 years later. Group them.
- **Sampling and montage:** 256 Hz, bipolar 10-20 montage, typically 23 channels. Most files share the 18 channels listed in the config. Some files have different ordering, placeholder channels named `-`, or a duplicated `T8-P8` (may appear as `T8-P8-0` and `T8-P8-1`). Select by name; add aliases to `channel_aliases` after inspection.
- **Summary file format** (two variants; support both):

  ```text
  File Name: chb01_03.edf
  File Start Time: 13:43:04
  File End Time: 14:43:04
  Number of Seizures in File: 1
  Seizure Start Time: 2996 seconds
  Seizure End Time: 3036 seconds
  ```

  ```text
  File Name: chb04_05.edf
  Number of Seizures in File: 2
  Seizure 1 Start Time: 7804 seconds
  Seizure 1 End Time: 7853 seconds
  Seizure 2 Start Time: 9081 seconds
  Seizure 2 End Time: 9196 seconds
  ```

- **Class imbalance:** ictal windows are a small fraction of all windows. Several patients have only a handful of seizures, so their per-patient metrics are noisy. Report medians as well as means.
- **Volume:** seizure files alone run to hundreds of hours of 18-channel 256 Hz data. A preprocessed hour is about 66 MB (float32). Plan disk space and use `--patients` for development.
- **Line frequency:** the recordings were made in the US, so the notch is at 60 Hz.

---

*End of README. If anything here is ambiguous or inconsistent, stop and ask the human before deviating.*