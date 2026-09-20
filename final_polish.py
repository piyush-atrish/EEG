import yaml
import subprocess
from pathlib import Path
import re

# 1. Recover Member B's config parameters from Git history
b_yaml = subprocess.check_output(['git', 'show', 'origin/userb:configs/config.yaml']).decode('utf-8')
b_cfg = yaml.safe_load(b_yaml)
with open('configs/config.yaml', 'r') as f: 
    a_cfg = yaml.safe_load(f)

if 'features' in b_cfg: a_cfg['features'] = b_cfg['features']
if 'segmentation' in b_cfg: a_cfg['segmentation'] = b_cfg['segmentation']

with open('configs/config.yaml', 'w') as f: 
    yaml.dump(a_cfg, f, sort_keys=False)

# 2. Recover Member B's path helpers
with open('src/eegpipe/utils/paths.py', 'r') as f: 
    paths_code = f.read()

missing_paths = """
def windows_path(cfg: dict, case: str) -> Path:
    return Path(cfg["paths"]["windows"]) / f"{case}.parquet"

def features_path(cfg: dict, case: str) -> Path:
    return Path(cfg["paths"]["features"]) / f"{case}.parquet"

def predictions_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["predictions"]) / "predictions.csv"
"""
if "windows_path" not in paths_code:
    with open('src/eegpipe/utils/paths.py', 'a') as f: 
        f.write(missing_paths)

# 3. Enforce the hard exit code requirement in script 01
with open('scripts/01_download_and_index.py', 'r') as f: 
    content = f.read()

# Replace any 'continue' that happens after a 'no usable seizure' log
content = re.sub(
    r'(logger\.(?:error|warning)\([^)]*seizure[^)]*\)\n\s+)continue', 
    r'\1return 1', 
    content, 
    flags=re.IGNORECASE
)

# Safety net: intercept right before it finishes
safety_net = """
    try:
        if 'file_index' in locals() and 'include' in file_index.columns:
            for p, g in file_index[file_index['include']].groupby('patient'):
                if (g['role'] == 'seizure').sum() == 0:
                    return 1
    except Exception:
        pass
    return 0"""
content = content.replace("    return 0", safety_net)

with open('scripts/01_download_and_index.py', 'w') as f: 
    f.write(content)
