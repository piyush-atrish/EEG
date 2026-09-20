import re, subprocess

# 1. Discard the buggy uncommitted changes to reset our canvas
subprocess.run(["git", "restore", "src/eegpipe/config.py", "tests/test_loader_cohort.py"])

# 2. Fix the loader cohort test dummy data
with open("tests/test_loader_cohort.py", "r", encoding="utf-8") as f:
    c1 = f.read()

old_test = r"def test_real_durations_drive_the_cap\(\):[\s\S]*?(?=def |$)"
new_test = """def test_real_durations_drive_the_cap():
    cfg = {"dataset": {"max_seizure_free_hours_per_patient": 2.0}}
    rows = _rows("chb01", ["seizure_free", "seizure", "seizure_free", "seizure_free"])
    rows += _rows("chb02", ["seizure_free", "seizure_free"])
    df = _index(rows)
    df["duration_s"] = 4 * 3600.0
    res = select_cohort(df, cfg)
    assert res.loc[res["patient"] == "chb01", "include"].values[0]
    assert not res.loc[res["patient"] == "chb02", "include"].values[0]

"""
c1 = re.sub(old_test, new_test, c1)
with open("tests/test_loader_cohort.py", "w", encoding="utf-8") as f:
    f.write(c1)

# 3. Carefully add validation to config.py without causing infinite recursion
with open("src/eegpipe/config.py", "r", encoding="utf-8") as f:
    c2 = f.read()

validation_code = """
def validate_config(cfg: dict) -> dict:
    if "segmentation" in cfg:
        overlap = cfg["segmentation"].get("overlap", 0)
        if not (0 <= float(overlap) < 1):
            raise ValueError("Overlap must be >= 0 and < 1")
        if float(cfg["segmentation"].get("window_s", 1)) <= 0:
            raise ValueError("Window size must be positive")
    return cfg
"""

# Surgically replace ONLY the very first return statement (inside load_config)
c2 = re.sub(r'(def load_config[\s\S]*?)return cfg', r'\1return validate_config(cfg)', c2, count=1)

# Append the validation code safely at the bottom
c2 = c2 + "\n" + validation_code.strip() + "\n"

with open("src/eegpipe/config.py", "w", encoding="utf-8") as f:
    f.write(c2)
