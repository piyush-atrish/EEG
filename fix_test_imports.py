with open("tests/test_loader_cohort.py", "r", encoding="utf-8") as f:
    content = f.read()

missing_code = """from eegpipe.io.loader import read_edf_header, check_edf, resolve_channel_names
from tests.fixtures.synth_signals import write_synthetic_edf
CH = ["FP1-F7", "F7-T7", "T8-P8"]
"""

with open("tests/test_loader_cohort.py", "w", encoding="utf-8") as f:
    f.write(missing_code + content)
