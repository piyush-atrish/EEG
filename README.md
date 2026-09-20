# eeg-seizure-loso

## Cross-Subject Generalisation of EEG Signals Using Adaptive Filtering for Epileptic Seizure Detection

**B.Tech Final Year Project — Department of Electronics and Communication Engineering, Delhi Technological University**  
**Batch:** 2023–2027  
**Guide:** Dr. Sonal Singh

---

## Project Overview

This project develops a DSP-first pipeline for **cross-subject EEG seizure detection** using the **CHB-MIT Scalp EEG Database**.

The current implementation establishes the complete **Milestone 1 non-adapted baseline**, covering EEG preprocessing, windowing, feature extraction, patient-level evaluation, baseline machine-learning models, and result generation.

The baseline is designed to provide a controlled reference for later adaptive-filtering methods.

---

# 1. Key Features

### EEG Data Processing

- CHB-MIT Scalp EEG Database
- Sampling frequency: **256 Hz**
- **18 common bipolar EEG channels**, selected by channel name
- Signal representation in **microvolts (uV)** using `float32`
- Seizure annotation parsing and patient/case indexing
- Correct patient grouping, including `chb21` and `chb01` as the same patient

### Signal Preprocessing

Each EEG recording is processed using:

- **60 Hz notch filtering** for power-line interference
- **0.5–45 Hz Kaiser-window FIR band-pass filtering**
- Linear-phase filtering with delay compensation
- Cached preprocessed signals for downstream processing

### EEG Windowing and Labelling

- Window length: **4 seconds**
- Window size: **1024 samples**
- Overlap: **50%**
- Step: **2 seconds / 512 samples**

Label definition:

- **Ictal (1):** window completely inside an annotated seizure
- **Interictal (0):** window does not overlap a seizure
- **Discarded:** window partially overlaps a seizure boundary

### Feature Extraction

The system extracts **32 features per channel** across 18 channels, producing up to **576 feature columns**.

**Time-domain**
- Mean
- Standard deviation
- Skewness
- Kurtosis
- Line length
- Hjorth activity
- Hjorth mobility
- Hjorth complexity

**Frequency-domain**
- Delta, theta, alpha, beta and gamma power
- Relative power for each frequency band
- Welch PSD-based spectral features

**Nonlinear**
- Sample entropy
- Approximate entropy

**Wavelet**
- 5-level **db4 discrete wavelet transform**
- Log-energy of detail and approximation sub-bands
- Standard deviation of wavelet coefficients

### Baseline Classification

Two baseline classifiers are implemented:

- **Support Vector Machine (RBF kernel)**
- **Random Forest**

The modelling pipeline includes feature scaling and feature selection before classification.

### Cross-Subject Evaluation

The system uses **Leave-One-Subject-Out (LOSO)** evaluation:

- One patient is held out for testing.
- Remaining patients form the training set.
- The process is repeated for each patient.
- Patient identity, rather than individual recording identity, defines the train/test split.

### Experimental Comparison

Two experimental arms are evaluated:

| Arm | Description |
|---|---|
| `raw` | Features are used without additional subject-level normalisation. |
| `subject_standardised` | Features are standardised using the patient's own early interictal calibration segment. |

Calibration windows are excluded from scoring in both arms to keep the comparison on identical evaluation windows.

### Evaluation Outputs

The system produces:

- Sensitivity
- Specificity
- F1-score
- AUC
- False alarms per hour
- Per-patient metrics
- Mean, standard deviation and median summaries
- Prediction files
- Comparison tables
- Result figures

---

# 2. System Flow

```text
                   CHB-MIT EEG DATABASE
                            |
                            v
                +------------------------+
                | Data Download & Index  |
                | Seizure Annotations    |
                +-----------+------------+
                            |
                            v
                +------------------------+
                | Channel Selection      |
                | Patient/Case Grouping   |
                | Cohort Construction    |
                +-----------+------------+
                            |
                            v
                +------------------------+
                | EEG Preprocessing      |
                | 60 Hz Notch             |
                | 0.5–45 Hz FIR           |
                +-----------+------------+
                            |
                            v
                +------------------------+
                | Windowing & Labelling  |
                | 4 s / 50% overlap      |
                +-----------+------------+
                            |
                            v
                +------------------------+
                | Feature Extraction     |
                | Time + Frequency       |
                | Nonlinear + Wavelet    |
                +-----------+------------+
                            |
                            v
                +------------------------+
                | Experimental Arms      |
                |                        |
                | Raw                    |
                | Subject Standardised  |
                +-----------+------------+
                            |
                            v
                +------------------------+
                | LOSO Classification    |
                |                        |
                | SVM + Random Forest    |
                +-----------+------------+
                            |
                            v
                +------------------------+
                | Evaluation & Reporting |
                | Metrics + Predictions  |
                | Tables + Figures       |
                +------------------------+
```

### Evaluation Flow

```text
                    ALL PATIENTS
                         |
            +------------+------------+
            |                         |
            v                         v
      TRAINING PATIENTS        HELD-OUT PATIENT
            |                         |
            | model fitting            | final scoring
            | feature selection       | natural class balance
            | training-only           |
            | preprocessing           |
            +------------+------------+
                         |
                         v
                 PATIENT-LEVEL RESULTS
```

The evaluation maintains patient-level separation so that recordings from the same subject do not appear in both training and testing sets.

---

# 3. Team Distribution

The project is divided into three complementary modules.

| Member | Module | Main Responsibilities | Primary Outputs |
|---|---|---|---|
| **User A** | Data & Preprocessing | Dataset acquisition, annotation parsing, channel selection, cohort construction, filtering and preprocessing | `annotations.csv`, `file_index.csv`, preprocessed EEG arrays |
| **User B** | Windowing & Feature Engineering | EEG segmentation, seizure labelling, time/frequency features, nonlinear features and wavelet features | Window tables and feature tables |
| **User C** | Evaluation & Baseline Models | LOSO evaluation, leakage checks, SVM, Random Forest, subject standardisation, metrics and reporting | Predictions, metric tables and figures |

### User A — Data & Preprocessing

Handles the complete upstream signal-processing pipeline:

```text
Raw CHB-MIT EEG
      ↓
Annotations + File Index
      ↓
Channel Selection
      ↓
Cohort Construction
      ↓
Notch + FIR Filtering
      ↓
Preprocessed EEG
```

### User B — Windowing & Features

Converts preprocessed EEG into machine-learning inputs:

```text
Preprocessed EEG
      ↓
4-second Windows
      ↓
Ictal / Interictal Labels
      ↓
Time Features
      ↓
Frequency Features
      ↓
Nonlinear Features
      ↓
Wavelet Features
      ↓
Feature Matrix
```

### User C — Evaluation & Models

Performs model training, cross-subject evaluation and result generation:

```text
Feature Matrix
      ↓
Raw / Subject-Standardised Arms
      ↓
SVM / Random Forest
      ↓
LOSO Evaluation
      ↓
Per-Patient Metrics
      ↓
Summary Tables + Figures
```

---

# 4. Repository Structure

```text
eeg-seizure-loso/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
│
├── .github/
│   ├── CODEOWNERS
│   ├── pull_request_template.md
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   └── config.yaml
│
├── docs/
│   ├── design_lock.md
│   ├── data_notes.md
│   ├── filter_validation.md
│   ├── feature_catalog.md
│   ├── baseline_report.md
│   └── figures/
│
├── data/
│   ├── raw/
│   │   └── chbmit/
│   │
│   ├── interim/
│   │   ├── annotations.csv
│   │   ├── file_index.csv
│   │   └── preprocessed/
│   │       └── {case}/
│   │           └── {file_stem}.npy
│   │
│   └── processed/
│       ├── windows/
│       │   └── {case}.parquet
│       └── features/
│           └── {case}.parquet
│
├── src/
│   └── eegpipe/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── logging_utils.py
│       │   ├── seed.py
│       │   └── paths.py
│       │
│       ├── io/
│       │   ├── __init__.py
│       │   ├── download.py
│       │   ├── annotations.py
│       │   ├── loader.py
│       │   └── cohort.py
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── filters.py
│       │   └── pipeline.py
│       │
│       ├── segmentation/
│       │   ├── __init__.py
│       │   └── windows.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── time_domain.py
│       │   ├── frequency_domain.py
│       │   ├── nonlinear.py
│       │   ├── wavelet.py
│       │   └── extract.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── normalisation.py
│       │   └── classifiers.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── leakage_checks.py
│       │   ├── loso.py
│       │   └── reporting.py
│       │
│       └── adaptive/
│           └── __init__.py
│
├── scripts/
│   ├── 01_download_and_index.py
│   ├── 02_preprocess.py
│   ├── 03_make_windows.py
│   ├── 04_extract_features.py
│   ├── 05_run_baseline.py
│   └── 06_make_report.py
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── __init__.py
│   │   ├── synth_signals.py
│   │   └── synth_features.py
│   ├── test_config.py
│   ├── test_annotations.py
│   ├── test_loader_cohort.py
│   ├── test_filters.py
│   ├── test_preprocess_pipeline.py
│   ├── test_windows.py
│   ├── test_time_frequency_features.py
│   ├── test_nonlinear_wavelet_features.py
│   ├── test_extract.py
│   ├── test_metrics.py
│   ├── test_leakage_checks.py
│   ├── test_models_normalisation.py
│   ├── test_loso.py
│   └── test_integration.py
│
├── notebooks/
│   ├── A_data_and_filter_exploration.ipynb
│   ├── B_feature_exploration.ipynb
│   └── C_results_exploration.ipynb
│
└── results/
    ├── tables/
    │   ├── per_patient_{arm}__{model}.csv
    │   └── summary_all.csv
    ├── figures/
    ├── predictions/
    └── logs/
```

### Module Mapping

```text
src/eegpipe/io/             → User A
src/eegpipe/preprocessing/  → User A

src/eegpipe/segmentation/   → User B
src/eegpipe/features/       → User B

src/eegpipe/models/         → User C
src/eegpipe/evaluation/     → User C
```

---

## Core Data Flow

```text
annotations.csv
       +
file_index.csv
       +
preprocessed EEG
       │
       ▼
window tables
       │
       ▼
feature tables
       │
       ▼
raw / subject-standardised data
       │
       ▼
SVM / Random Forest
       │
       ▼
LOSO predictions
       │
       ▼
patient-level metrics
       │
       ▼
final tables and figures
```
