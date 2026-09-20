path = 'scripts/01_download_and_index.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the very last 'return 0' in the file (the end of main) and safely replace it
for i in range(len(lines)-1, -1, -1):
    if lines[i].strip() == 'return 0':
        lines[i] = """        # If a patient originally had seizures but all were excluded, fail the pipeline
        for p, g in cohort.groupby('patient'):
            total_seizures = (g['role'] == 'seizure').sum()
            included_seizures = (g['include'] & (g['role'] == 'seizure')).sum()
            if total_seizures > 0 and included_seizures == 0:
                return 1
        return 0\n"""
        break

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
