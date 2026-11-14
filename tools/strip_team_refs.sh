#!/usr/bin/env bash
# One-off cleanup for solo work: removes "Member A/B/C" and "User A/B/C" ownership wording
# from docstrings, comments, README and docs. Run once from the repo root in Git Bash, then
# review with:  git diff --ignore-cr-at-eol
set -euo pipefail

FILES=$(grep -rlE "Member [ABC]|member [abc]|and reused by B and C" src scripts tests docs \
        --include="*.py" --include="*.md" || true)
if [ -n "$FILES" ]; then
sed -i -E \
 -e 's/^# Data notes \(Member A\)/# Data notes/' \
 -e 's/^# Filter validation \(Member A, (Step [0-9]+)\)/# Filter validation (\1)/' \
 -e 's/^# Member A: fix checklist/# Fix checklist/' \
 -e 's/\(owner: Member [ABC], (Steps? [0-9-]+)\)/(\1)/' \
 -e 's/\(Member [ABC], (Steps? [0-9-]+)\)/(\1)/' \
 -e 's/^Member [ABC], (Step [0-9]+)/\1/' \
 -e 's/# ---- Member B: contracts/# ---- Features: contracts/' \
 -e 's/# ---- Member C: contracts/# ---- Evaluation: contracts/' \
 -e 's/\(Member C reads this\)/(the evaluation code reads this)/' \
 -e 's/ \(Member [ABC]\)//' \
 -e "s/ \(and reused by B and C\)//" \
 -e "s/\(not Member A's tests\/fixtures\/synth_signals\.py,/(not tests\/fixtures\/synth_signals.py)/" \
 -e "s/^which is A's file to own\) built directly/built directly/" \
 -e "s/for Member A's modules/for the data modules/" \
 -e "s/Member A's/the data module's/g" \
 -e "s/Member B's/the feature module's/g" \
 -e "s/Member C's/the evaluation module's/g" \
 $FILES
fi

# README: team section -> neutral pipeline-module wording
sed -i -E \
 -e 's/^# 3\. Team Distribution/# 3. Pipeline Modules/' \
 -e 's/^The project is divided into three complementary modules\./The pipeline is organised into three modules./' \
 -e 's/^\| Member \| Module \|/| Stage | Module |/' \
 -e 's/^\| \*\*User A\*\* \|/| **1** |/' \
 -e 's/^\| \*\*User B\*\* \|/| **2** |/' \
 -e 's/^\| \*\*User C\*\* \|/| **3** |/' \
 -e 's/^### User A — /### Module 1 — /' \
 -e 's/^### User B — /### Module 2 — /' \
 -e 's/^### User C — /### Module 3 — /' \
 -e 's/→ User A/→ Module 1/' \
 -e 's/→ User B/→ Module 2/' \
 -e 's/→ User C/→ Module 3/' \
 -e '/CODEOWNERS/d' \
 README.md

# docs: drop the obsolete CODEOWNERS items from the fix checklist
sed -i -E \
 -e '/@handleA\/@handleB\/@handleC/d' \
 -e 's/; CODEOWNERS per folder//' \
 docs/A_fix_checklist.md

echo "Done. Remaining team wording (should be empty):"
grep -rnE "Member [ABC]|User [ABC]|handle[ABC]" README.md src scripts tests docs \
     --include="*.py" --include="*.md" || true
