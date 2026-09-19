import pytest
import pandas as pd
from pathlib import Path
from eegpipe.io.annotations import parse_summary_file, build_annotations

def test_parse_summary_formats(tmp_path):
    summary_text = """
File Name: chb01_01.edf
File Start Time: 11:42:54
File End Time: 12:42:54
Number of Seizures in File: 0

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb04_05.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 2
Seizure 1 Start Time: 7804 seconds
Seizure 1 End Time: 7853 seconds
Seizure 2 Start Time: 9081 seconds
Seizure 2 End Time: 9196 seconds
"""
    test_file = tmp_path / "test-summary.txt"
    test_file.write_text(summary_text)
    
    results = parse_summary_file(test_file)
    
    # Zero seizures should be handled but yield empty seizure lists
    assert len(results) == 3
    assert len(results[0]["seizures"]) == 0
    
    # Format v1 (single seizure)
    assert len(results[1]["seizures"]) == 1
    assert results[1]["seizures"][0] == (2996.0, 3036.0)
    
    # Format v2 (numbered multiple seizures)
    assert len(results[2]["seizures"]) == 2
    assert results[2]["seizures"][0] == (7804.0, 7853.0)
    assert results[2]["seizures"][1] == (9081.0, 9196.0)

def test_build_annotations_patient_mapping_and_logic(tmp_path):
    # Setup dummy directory structure resembling CHB-MIT
    chb21_dir = tmp_path / "chb21"
    chb21_dir.mkdir()
    summary_file = chb21_dir / "chb21-summary.txt"
    summary_file.write_text("""
File Name: chb21_03.edf
Number of Seizures in File: 1
Seizure Start Time: 100 seconds
Seizure End Time: 200 seconds
""")
    
    patient_map = {"chb21": "chb01"}
    df = build_annotations(tmp_path, patient_map)
    
    # Validate mapping: case remains chb21, but patient becomes chb01
    assert len(df) == 1
    assert df.iloc[0]["patient"] == "chb01"
    assert df.iloc[0]["case"] == "chb21"
    assert df.iloc[0]["file"] == "chb21_03.edf"
    
    # Validate logical constraints
    assert df.iloc[0]["seizure_end_s"] > df.iloc[0]["seizure_start_s"]