import glob, os

# 1. Strip Member B's accidentally committed Git conflict markers
for path in glob.glob('scripts/*.py') + glob.glob('src/eegpipe/**/*.py', recursive=True):
    if not os.path.isfile(path): continue
    with open(path, 'r', encoding='utf-8') as f: lines = f.readlines()
    if not any(l.startswith('<<<<<<<') for l in lines): continue
    
    out = []
    keeping = True
    for line in lines:
        if line.startswith('<<<<<<<'): keeping = True; continue
        if line.startswith('======='): keeping = False; continue  # Drop the old stubs!
        if line.startswith('>>>>>>>'): keeping = True; continue
        if keeping: out.append(line)
            
    with open(path, 'w', encoding='utf-8') as f: f.writelines(out)

# 2. Fix the exit code bug in 01_download_and_index.py found by the new tests
path = 'scripts/01_download_and_index.py'
if os.path.isfile(path):
    with open(path, 'r', encoding='utf-8') as f: lines = f.readlines()
    for i, line in enumerate(lines):
        if 'seizure' in line.lower() and ('no usable' in line.lower() or 'all' in line.lower()):
            if i+1 < len(lines):
                lines[i+1] = lines[i+1].replace('continue', 'return 1').replace('return 0', 'return 1')
    with open(path, 'w', encoding='utf-8') as f: f.writelines(lines)
