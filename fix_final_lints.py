# 1. Fix missing imports and function name in config.py
with open("src/eegpipe/config.py", "r") as f:
    c = f.read()
c = c.replace("import collections.abc", "import copy")
c = c.replace("deep_update(", "_deep_merge(")
with open("src/eegpipe/config.py", "w") as f:
    f.write(c)

# 2. Fix missing sys import in logging_utils.py
with open("src/eegpipe/utils/logging_utils.py", "r") as f:
    c = f.read()
if "import sys" not in c:
    c = c.replace("import logging", "import logging\nimport sys")
with open("src/eegpipe/utils/logging_utils.py", "w") as f:
    f.write(c)

# 3. Strip the duplicate functions in paths.py
with open("src/eegpipe/utils/paths.py", "r") as f:
    lines = f.readlines()
out, seen = [], set()
skip = False
for line in lines:
    if line.startswith("def "):
        func = line.split("(")[0]
        if func in seen: 
            skip = True
        else:
            seen.add(func)
            skip = False
    if not skip: 
        out.append(line)
with open("src/eegpipe/utils/paths.py", "w") as f:
    f.writelines(out)

# 4. Inject the missing imports and target channel list into the test file
with open("tests/test_loader_cohort.py", "r") as f:
    c = f.read()
missing_imports = """import numpy as np
import pytest
from eegpipe.io.loader import load_edf_channels, ChannelMissingError, read_edf_header, check_edf, resolve_channel_names
from tests.fixtures.synth_signals import write_synthetic_edf

CH = ["FP1-F7", "F7-T7", "T8-P8"]
"""
if "load_edf_channels" not in c[:500]:
    c = missing_imports + c.replace("import pytest", "").replace("import numpy as np", "")
with open("tests/test_loader_cohort.py", "w") as f:
    f.write(c)
