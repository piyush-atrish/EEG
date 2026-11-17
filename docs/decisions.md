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