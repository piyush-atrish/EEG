# Adaptive filtering stage: design note (chunk 2.1)

Status: DRAFT for review. Nothing below is implemented yet. Once agreed, the decisions
here are copied into `docs/decisions.md` and frozen before any tuning starts.

## 1. Goal and non-goals

**Goal.** Insert a causal, label-free adaptive stage between preprocessing (script 02) and
windowing/features (scripts 03-04) that reduces *between-patient* differences in the EEG while
keeping the *within-patient* seizure / non-seizure contrast, so that one shared classifier
generalises better across patients.

**Non-goals.** No deep models, no use of labels or of any other patient's data inside the
filter, no change to windows, labels, folds or features. All three arms (raw,
subject-standardised, adaptive) must be scored on **identical windows and folds**.

## 2. What the filter sees and predicts

All methods run per channel (18 channels, independent states), sample by sample, in time order.

| Method | Input u(n) | Desired d(n) | Output y(n) | Role |
|---|---|---|---|---|
| **NLMS predictor** | past L samples of the same channel | x(n) | prediction error e(n) = x(n) - w'u(n) | adaptive whitening; cheap, hardware-friendly |
| **RLS predictor** | same | x(n) | same | same idea, faster convergence |
| **Kalman gain tracker** | block log-amplitude of the channel | - | x(n) * exp(-s_hat), s_hat = tracked log-amplitude | adaptive gain control (time-varying cousin of arm 1) |

**Why a temporal predictor.** With slow adaptation the filter learns each patient's
*background* spectrum (its 1/f slope, dominant rhythms, noise floor). The error e(n) is what
the background model cannot predict. Patient-specific background shape is removed, and a
seizure, being a departure from the background, stays visible. With *fast* adaptation the
filter would learn the seizure rhythm itself and cancel it, so adaptation speed is the key
hyperparameter (section 4).

**Why not the other two options named in the report.**
- *Population reference* d(n): there is no meaningful sample-by-sample reference signal shared
  across patients, so this is not defined at signal level.
- *Neighbouring-channel prediction* (spatial): seizure activity becomes more coherent across
  channels, so the spatial residual would shrink exactly when a seizure starts. Kept as an
  optional later ablation, not part of the primary comparison.

**Gain tracker details.** Observations are block log-RMS values z_k (1 s blocks). Random-walk
state s_k = s_(k-1) + w, w ~ N(0, q); z_k = s_k + v, v ~ N(0, r). The gain applied to block k
uses the estimate from blocks up to k-1 only (one-block lag), so a seizure block never
normalises itself. The ratio q/r sets the adaptation speed.

## 3. State handling (decides leakage and comparability)

- One **independent state stream per case** (chb01 and chb21 are the same patient but recorded
  long apart, so they are separate streams). States are never shared across patients.
- Within a case, files are processed in chronological order and the state is **carried across
  files** (a deployed device would not reset every hour).
- **Warm start:** the filter is run once over the first 60 s of the first file to converge,
  then restarted from sample 0 with those weights. It uses no labels and only the first
  minute, which lies inside the 10-minute calibration period that arm 1 already excludes from
  scoring.
- The filter is causal after the warm start and uses no labels, so it cannot leak by construction.
  This is still covered by a unit test (output at time n unchanged if the future samples are
  altered).

## 4. Hyperparameters and pilot grid (dev patients only)

| Method | Parameter | Pilot values | Time scale |
|---|---|---|---|
| NLMS | filter length L | 8, 16, 32 | |
| NLMS | step size mu | 0.001, 0.005, 0.02 | converges in roughly L/mu samples |
| RLS | filter length L | 8, 16 | |
| RLS | forgetting factor lambda | 0.999, 0.9995, 0.9999 | memory 1/(1-lambda): 4 s, 8 s, 39 s |
| Kalman | q/r | 1e-4, 1e-3, 1e-2 | |

Rules to protect the seizure: input power below a small epsilon skips the update (flat
channels must not blow up the RLS covariance), the RLS covariance is symmetrised and capped,
and the NLMS update is normalised by input power. Adaptation gating during suspected
seizures is **not** in version 1; it is an ablation if the plain filters cancel seizures.

**Compute.** About 1.3 billion sample updates for the full cohort. Everything is implemented
with Numba (already a dependency), channels in parallel, cases in parallel. Estimated cost:
NLMS minutes, RLS (L=16) tens of minutes per full pass. These are estimates, to be measured
in chunk 2.3.

## 5. How we decide, before running a classifier

Two cheap diagnostics on feature tables from the adaptive signals, interictal windows only for
the first one:

- **M1, subject invariance:** for each feature, the fraction of variance explained by patient
  identity (eta-squared, between-patient SS / total SS). Lower is better. Reported for raw,
  subject-standardised and adaptive features.
- **M2, contrast kept:** within-patient seizure vs non-seizure separation (per-feature AUC,
  averaged over patients). It must not drop materially versus raw.
- **Synthetic seizure-preservation test (chunk 2.6):** a spike-wave burst injected into
  synthetic background must keep at least half of its power contrast at the chosen settings.

A setting only proceeds to a classifier run if M1 improves and M2 and the synthetic test hold.

## 6. Data split (frozen rule)

- **Development patients:** chb01-chb05 (already seen in Milestone 1). All hyperparameters,
  and the post-processing settings, are chosen here.
- **Evaluation patients:** chb06 onwards. The frozen settings are applied once. These are the
  primary reported results. Results on the development patients are shown separately and
  labelled optimistic.

## 7. Outputs and integration

- New array directory `data/interim/preprocessed_adaptive/<method>/` with the same shape and
  dtype as the preprocessed arrays (18 x N, float32), written by `scripts/02b_adaptive.py`,
  resumable per case.
- Windows and feature scripts get an input-directory option; nothing else changes.
- Arms for the comparison: `raw`, `subject_standardised`, `adaptive`, and optionally
  `adaptive_standardised` (feature z-score applied on top) to show whether the two effects add.

## 8. Honest expectations

The gain tracker is close to a time-varying version of arm 1, so a modest gain over it is the
realistic expectation. The predictor-based filters could help more or could hurt (cancelled
rhythms). A negative or mixed result is still a valid, reportable finding and the write-up is
planned that way.

## 9. Decisions needed from you

1. Approve the **temporal predictor (whitening)** formulation for NLMS and RLS as the primary
   design, with the spatial variant only as a later ablation.
2. Approve **per-case state carried across files** with the 60 s warm start.
3. Approve the **dev/evaluation split** in section 6.
