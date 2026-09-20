path = 'scripts/01_download_and_index.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(len(lines)-1, -1, -1):
    if lines[i].startswith('    return 0'):
        lines[i] = (
            "    # If a patient originally had seizures but all were excluded, fail the pipeline\n"
            "    for p, g in cohort.groupby('patient'):\n"
            "        total_seizures = (g['role'] == 'seizure').sum()\n"
            "        included_seizures = (g['include'] & (g['role'] == 'seizure')).sum()\n"
            "        if total_seizures > 0 and included_seizures == 0:\n"
            "            return 1\n"
            "    return 0\n"
        )
        break

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
