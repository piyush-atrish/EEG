# Decision log

Short, dated record of design choices that the final report must state explicitly.

## 2026-11-14 · Hyperparameter tuning mode: nested (patient-grouped)

- **Decision:** keep `evaluation.tuning.mode: nested` for every reported result.
- **Meaning:** inside each LOSO fold, hyperparameters are chosen by `GridSearchCV` with
  `GroupKFold` over the *training* patients only (3 inner splits, scoring ROC-AUC), so the
  held-out patient never influences model selection (leakage rule L4).
- **Why not `fixed`:** Milestone 1 fold times were about 2–3 s (SVM) and 24–34 s (RF), so
  nested tuning stays affordable on the full cohort. `fixed` remains only a fast fallback
  for debugging, never for reported numbers.
- **Report:** state "nested, patient-grouped, 3 inner folds" in the methodology and check
  that the Milestone 1 numbers were produced with this mode before reusing them.

## 2026-11-15 · Adaptive stage design approved

- **Decision:** NLMS and RLS run as per-channel temporal predictors (output = prediction
  error); a Kalman log-amplitude gain tracker is the third method. State is kept per case
  and carried across files in time order, with a 60 s warm start. Settings are chosen on
  chb01-chb05 only and frozen before scoring chb06 onwards.
- **Details:** see docs/adaptive_design.md.

## 2026-11-17 · Treatment of chb06

- **Decision:** chb06 stays in every reported result. The primary comparison is also
  reported without chb06 as a sensitivity analysis.
- **Evidence:** within-patient direction agreement 0.376 vs 0.85-0.98 for others; its seizures
  barely move the strongest consensus features. This is a missing shared signature, not a
  baseline shift, so it is reported as a failure case of any cross-subject model.
- **Rule:** no patient is excluded or added after seeing adaptive-filter results.
- Kalman q/r pilot grid extended to 1e-5, 3e-5, 1e-4, 1e-3, 1e-2 after a synthetic burst test showed only q/r ≤ 1e-4 keeps at least half of a 30 s burst’s power contrast. (That criterion was later replaced; see 2026-11-21.)

## 2026-11-21 · Seizure-preservation criterion changed (design note section 5)

- **Change:** the synthetic screen passes a setting if it keeps at least 85% of the burst's
  feature separability (window features, 15/30/60 s bursts, spike-wave and rhythmic, 2x
  amplitude), instead of 50% of broadband power contrast.
- **Why:** on synthetic bursts, whitening filters (NLMS/RLS) lose broadband power contrast
  by design while the burst stays separable in features, so power contrast wrongly penalised
  them. Feature separability is closer to what the classifier uses. Power contrast is still
  reported as a diagnostic.
- **Scope:** the screen removes clearly destructive settings only; the real-data pilot on
  chb01-chb05 selects among the survivors.

## 2026-11-22 · Warm start and pilot staging

- Warm start stays 60 s: a 600 s warm start reduced the NLMS first-file output/input RMS
  ratio only from 0.088 to 0.076 (chb01) and 0.104 to 0.085 (chb21), and changed nothing later.
- NLMS (mu 0.005) shows a first-hour convergence transient (ratio 0.09-0.10 in the first
  file, 0.02-0.05 later); mu 0.02 reduces ratios 25-40% but keeps the pattern. RLS L8 lam
  0.9999 shows no first-file transient (ratio 0.008). Recorded as a risk for NLMS.
- Pilot runs in two stages on chb01-chb05: (A) cheap window features, M1/M2 for all screen
  survivors, no classifier; (B) full features + LOSO for the top 2-3 settings only.

## 2026-11-22 · Pilot stage A result and stage B candidates

- Stage A (chb01-05, cheap features): kalman q/r 3e-5 and 1e-4 pass the gate (M1 improved, M2 >=
  0.9 of raw). NLMS (L16 mu 0.02; L8 mu 0.005) and RLS (L8 lam 0.9999) fail the M2 floor (0.82,
  0.79, 0.75) but cut patient variance most (M1 ratio 0.61, 0.57, 0.40) and raise direction
  agreement (0.86, 0.89, 0.97). Whitening filters lose shape features (relative band powers,
  spectral entropy, Hjorth), not amplitude features.
- Stage B runs the two passing Kalman settings plus rls_L8_lam0.9999 as a documented waiver of
  the M2 floor (decided before any classifier result). NLMS is left out of stage B to save time.
- The M2 floor is a screening heuristic, not a statistical test. The cheap-feature diagnostics
  cannot rank an adaptive arm against per-patient standardisation; only LOSO results can.

## 2026-11-23 · Adaptive setting frozen; evaluation plan fixed before scoring chb06-chb10

- Dev results (chb01-05, nested LOSO, window AUC): kalman q/r 3e-5 SVM 0.977 / RF 0.989 vs
  standardised 0.942 / 0.942 vs raw 0.773 / 0.891; RLS L8 0.837 / 0.870. Milestone 1 numbers
  were reproduced exactly (nested tuning confirmed).
- FROZEN: kalman, block 1 s, q/r 3e-5, r 1, warm start 60 s, state per case. No further tuning.
- Primary evaluation: window AUC of adaptive_kalman_qr3e-05_blk1 vs subject_standardised on
  chb06-chb10 (LOSO over all 10 patients, nested tuning), RF and SVM reported separately.
- Secondary: raw as reference; event-level sensitivity/FA per hour/latency; the
  same comparison without chb06; the paired test on all 10 patients.
- NLMS L8 mu 0.005 will be run for completeness (all three methods reported) but cannot
  change the frozen choice.
- Controls before trusting the result: label-shuffle check on the Kalman features;
  event-level report.

## 2026-11-23 · Primary evaluation result (chb06-chb10), frozen kalman q/r 3e-5

- Primary endpoint (window AUC, Kalman vs subject_standardised, chb06-10): RF 0.867 vs 0.860
  (+0.007, 95% CI -0.033 to 0.070, Kalman better on 2/5); SVM 0.809 vs 0.763 (+0.047,
  CI -0.111 to 0.204, 2/5). No evidence of a difference.
- Without chb06 (4 patients): RF 0.939 vs 0.962 (-0.023, CI -0.036 to -0.006, Kalman better
  on 1/4); SVM 0.878 vs 0.884.
- The development advantage (chb01-05) did not replicate: expected selection effect.
- Event level (59 seizures): Kalman RF 48 detected / 20.9 FA/h; standardised RF 46 / 23.9;
  raw RF 49 / 36.1; Kalman SVM 53 / 57.6; standardised SVM 48 / 39.9; raw SVM 53 / 80.8.
- Report as: Kalman beats raw on dev patients and lowers false alarms vs raw, but does not
  beat per-patient standardisation on unseen patients. NLMS/RLS gave no gain. chb06 remains a
  failure case for every arm.
- chb01-chb10 are now all seen. Any further design change must be judged on patients not yet
  used (chb11 onwards).

## 2026-11-25 · NLMS and RLS 10-patient results (completeness arms)

- Nested LOSO on chb01-chb10, mean window AUC, SVM / RF: NLMS L8 mu 0.005 0.831 / 0.847
  (with standardisation 0.834 / 0.883); RLS L8 lam 0.9999 0.852 / 0.862 (with standardisation
  0.768 / 0.870). Both are below subject-standardised (0.832 / 0.910) for RF and none is better
  than the frozen Kalman arm (0.885 / 0.926).
- These arms were run for completeness and do not change the frozen choice (2026-11-23).
- Event-level metrics for the RLS and NLMS arms are not yet computed.

## 2026-11-26 · Fresh test cohort (chb11-chb24): protocol fixed before any result exists

- **Why:** chb01-chb10 are all "seen" (2026-11-23), so they cannot settle Kalman vs per-patient
  standardisation. chb11-chb24 have not been downloaded, indexed, filtered or scored. The date of
  this entry precedes all of that.
- **Cohort size (chosen to limit compute):** 8 new patients. With an exact two-sided Wilcoxon,
  the smallest reachable p is 0.0156 at n=7 and 0.0078 at n=8, so n<7 could never reach the
  pre-set 0.025. Eight leaves room for one mechanical failure.
- **Selection rule (metadata only, fixed now, no outcome used):** rank chb11-chb24 (chb21 is
  chb01, excluded) by number of seizure-bearing files in RECORDS-WITH-SEIZURES, fewest first,
  ties by patient number. Selected: chb11, chb17, chb19, chb22, chb23 (3 files each), chb16,
  chb18, chb20 (6 each). If a selected patient fails the existing cohort rules in
  `configs/config.yaml` (18 channels readable by name, usable seizure file, 10 min calibration
  before the first seizure), it is replaced by the next in this order: chb14, chb13, chb24,
  chb12, chb15. The reason is logged. No patient is added or dropped after any model output
  exists for it.
- **Limitation to state in the report:** the subset favours patients with few seizures, so it
  is a smaller and less varied test than the full cohort; per-patient AUCs are noisier. The
  remaining patients (chb12-15, chb24) are listed as untested.
- **Frozen, unchanged:** Kalman q/r 3e-5, block 1 s, r 1, warm start 60 s, state per case;
  preprocessing, windows, 576 features, classifier grids, 5:1 sampling, 2 h seizure-free cap,
  10 min calibration. No retuning on the new patients.
- **Arms:** raw, subject_standardised, adaptive_kalman_qr3e-05_blk1 (primary comparison).
  Exploratory only: Kalman + standardised. RLS and NLMS are not rerun (no gain on chb01-10).
- **Training/evaluation:** nested, patient-grouped LOSO over all patients (chb01-chb10 plus the
  8 new ones). The 10-patient LOSO already reported stays in the report as the earlier analysis;
  the all-patient LOSO is the final one, and any difference between the two is stated.
- **Primary test:** window-level AUC, Kalman minus standardised, on the new patients only,
  RF and SVM separately; exact Wilcoxon across patients plus bootstrap CI resampling patients.
  Two primary tests, so the pre-set significance level is 0.025 each. Anything else is secondary.
- **Secondary:** raw as reference; event level (59 seizures before, more now); the same
  comparison without chb06 and pooled over all patients; per-patient pattern (chb04-type
  gains vs chb07/08/10-type losses); label-shuffle check on the new Kalman features.
- **Order of work:** (1) event post-processing settings chosen on chb01-chb05 and frozen
  (min_consecutive, merge gap, refractory); (2) download, index, preprocess, filter, features
  for chb11+; (3) LOSO; (4) analysis. No step is repeated because of how a result looks.
- **Change control:** a bug that affects chb11+ is fixed and all arms are rerun together, with
  a dated entry saying what changed and why.
- **How the outcome will be reported:** whichever way it goes. No difference on the new
  patients is a valid result and will be stated as such.

## 2026-11-26 · Event post-processing frozen (chosen on chb01-chb05 only)

- **Rule, fixed before the sweep was run:** one common setting for all arms and both models;
  minimise mean event-level false alarms per hour (pooled over raw, subject_standardised and
  Kalman q/r 3e-5, RF and SVM) subject to keeping at least 85% of the seizures the unsmoothed
  setting detects; ties go to the simpler setting (smaller min_consecutive, refractory, gap).
- **Data:** held-out predictions of chb01-chb05 from the 10-patient nested LOSO (not the
  `*_dev5` runs, which train on only 4 patients and are less like the final regime). 30 seizures
  per arm/model cell, 180 pooled.
- **Grid:** min_consecutive {1,2,3,4,6} x merge gap {default, 8, 30 s} x refractory {0, 60, 300 s}.
- **FROZEN:** min_consecutive 3, merge gap 8 s, refractory 0 s. Pooled: 150 of 180 detected
  (unsmoothed 166), mean FA/h 10.5 (unsmoothed 37.4), median latency 10.5 s (unsmoothed 4 s).
  In every arm/model cell FA/h fell 3-4x and detections fell by 1-4 of 30.
- **Cost, to be stated in the report:** latency rises from about 4 s to about 10 s.
- Applied unchanged to chb06-chb10 and to the chb11+ cohort. The unsmoothed (1, default, 0)
  numbers stay in the report as the raw event-level result.

## 2026-11-26 · Event-level results with frozen post-processing (secondary result, user-run)

- Setting: min_consecutive 3, merge gap 8 s, refractory 0 (see entry above). Source:
  `results/tables/event_per_patient_pp_frozen.csv`, run by the user on 2026-11-26.
- **All 10 patients (59 seizures)**, detected / mean FA per hour / median latency:
  Kalman RF 43 / 4.9 / 12 s, SVM 46 / 13.8 / 11 s; standardised RF 45 / 7.8 / 11 s, SVM 41 / 11.1 /
  10 s; raw RF 43 / 12.6 / 14 s, SVM 47 / 26.3 / 10 s; RLS RF 32 / 9.3, SVM 43 / 17.4;
  NLMS RF 30 / 6.0 / 25 s, SVM 45 / 21.6.
- **Evaluation patients chb06-chb10 (29 seizures):** Kalman RF 16 / 5.5, SVM 19 / 17.1;
  standardised RF 18 / 3.4, SVM 15 / 5.6; raw RF 23 / 20.7, SVM 24 / 37.7; RLS RF 17 / 12.9,
  SVM 24 / 24.7; NLMS RF 18 / 8.6, SVM 22 / 27.0.
- **Reading (descriptive, no test run):** Kalman lowers false alarms about 4x against raw but does
  not beat per-patient standardisation on the evaluation patients (RF: fewer detections and more
  false alarms; SVM: more detections at about 3x the false alarms). The chb01-05 false-alarm
  advantage of Kalman over standardisation (RF 4.2 vs 12.2 per hour) did not replicate on chb06-10,
  consistent with the selection effect already noted on 2026-11-23.
- The earlier unsmoothed event table (2026-11-23) stays in the report as the unprocessed result.
- Event-level conclusions remain provisional until the chb11+ cohort is scored.

## 2026-11-26 · Config clean-up: one adaptive block, defaults match the frozen primary

- `configs/config.yaml` held four concatenated `adaptive:` blocks (duplicate keys; the YAML loader
  silently kept the last). The effective Kalman default was q/r 1e-4, not the frozen 3e-5, so a run
  of `02b_adaptive.py --method kalman` without an explicit `--param` would have produced a
  different setting.
- **Change:** a single `adaptive:` block with `method: kalman` and `kalman.q_over_r: 3e-5` (block 1 s,
  r 1, warm start 60 s). NLMS and RLS defaults are unchanged. No design decision changes: this only
  aligns the file with the 2026-11-23 freeze. Existing result files carry the full setting in their
  names, so none depends on these defaults.
- The full unit suite (457 tests) passes with the new file.