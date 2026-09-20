import re

path = 'scripts/01_download_and_index.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix 1: Ensure it exits if there are NO files included for the requested patients
for i, line in enumerate(lines):
    if "logger.error(\"No files included for %s\", args.patients)" in line:
        if i+1 < len(lines) and "return 1" not in lines[i+1]:
            lines.insert(i+1, "            return 1\n")
        break

# Fix 2: Add the hard constraint to the absolute end of the main() function
out = []
for line in lines:
    if line.strip() == "return 0":
        # Intercept the final successful return
        out.append("""    # Enforce Member B's new constraint: patients must have usable seizure files
    if 'cohort' in locals() and 'include' in cohort.columns:
        for patient, group in cohort[cohort['include']].groupby('patient'):
            if (group['role'] == 'seizure').sum() == 0:
                logger.error(f"Patient {patient} lost all usable seizure files")
                return 1
    return 0\n""")
    else:
        out.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(out)
