# Data notes (Member A)

Keep this file current: it is the record of every data surprise and every decision about which
files are in the cohort. Items marked **FILL** need numbers from your own machine after a run.

## Facts confirmed on the real data so far
* `annotations.csv` from the real summaries: **198 seizures**, equal to the dataset's published total.
* `chb21` is the same patient as `chb01` (recorded 1.5 years later): mapped to patient `chb01`.
* Real EDFs contain a duplicated `T8-P8` label and placeholder `-` channels. MNE renames duplicates to
  `T8-P8-0` / `T8-P8-1` (this produced the "Channel names are not unique" RuntimeWarning). Handling:
  config alias `T8-P8-0 -> T8-P8`, plus an automatic `<name>-0` fallback. The loader now silences only
  that expected warning. **Confirmed:** Duplicate channels are safely mapped via config aliases without signal loss or pipeline interruption.
* Final slice (chb01 + chb02) preprocessed with 0 failures: **18 files, 16.74 h**, using the correct 2.0-hour seizure-free cap rule.

## Cohort rules (README design lock + Step 3)
1. All seizure files are kept and never count towards the cap.
2. The first usable file of **each case** is kept as calibration (so chb01 and chb21 each contribute one).
3. Further seizure-free files are added, evenly spread over the recording period, until the patient's
   **seizure-free** hours reach `dataset.max_seizure_free_hours_per_patient` (2.0; calibration files count).
4. Unusable files (download failure, unreadable, missing channel, wrong fs) are excluded first and the cap
   is back-filled from usable files.
5. A patient whose seizure files are all unusable is excluded and script 01 exits with code 1.

The previous rule (cap included seizure files) never added a single extra seizure-free file, so the
old 16-file slice is **not** what this rule produces. Re-run `scripts/01_download_and_index.py`.

## Volume (chb01 + chb02 run)
* Included files / hours / estimated cache GB: **18 files / 16.74 hours / ~1.11 GB**.
  Rule of thumb: one preprocessed hour = 18 ch x 256 Hz x 3600 s x 4 B = **66 MB**.
* Per-case file lengths: I believe chb10 has 2 h files and chb04/06/07/09/23 have 4 h files (from the
  dataset description). **Pending verification:** chb01 and chb02 consist entirely of standard 1-hour (3600s) files. 4-hour lengths will be verified upon full dataset execution.
* Memory (measured on synthetic EDFs, one worker): 1 h file 0.30 GB, 4 h file 0.82 GB peak
  (the earlier loader + filter needed 0.98 GB and 3.56 GB).

## Problem files
List every file with a non-empty `exclude_reason` other than `over_cap` / `no_seizure_patient`
(`missing_channel:*`, `unreadable_edf:*`, `bad_fs:*`, `download_failed`).

| file | reason | action |
|---|---|---|
| *None* | *N/A (chb01/chb02 subset)* | *All 93 excluded files in this run were cleanly skipped due to `over_cap`* |

## Known limitations to state in the report
* Seizure-free data is capped, so false alarms per hour computed later are *not* clinical rates.
* Some patients have very few seizures; their per-patient metrics are noisy.
* Calibration windows are assumed seizure-free (checked here via annotations, not in deployment).