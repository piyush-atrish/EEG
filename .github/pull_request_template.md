## What this PR does
<!-- One or two sentences. Format the title as: [A][step-N] short description -->

## Checklist
- [ ] I edited **only files I own** (see CODEOWNERS); shared files have all three approvals
- [ ] Contracts and `configs/config.yaml` are unchanged (or the change is agreed by all three)
- [ ] Tests added or updated, and `pytest -q` passes locally
- [ ] `ruff check src tests scripts` passes
- [ ] No data, caches, predictions or logs are committed
- [ ] Leakage rules L1-L8 respected (if this touches labels, subjects or fitting)

## How I tested it
<!-- Paste the pytest summary line and anything else a reviewer should see -->
