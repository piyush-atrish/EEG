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