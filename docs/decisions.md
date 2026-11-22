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
- Kalman q/r pilot grid extended to 1e-5, 3e-5, 1e-4, 1e-3, 1e-2 after a synthetic burst test showed only q/r ≤ 1e-4 keeps at least half of a 30 s burst’s power contrast.s

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