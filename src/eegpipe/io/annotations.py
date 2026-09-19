import re
from pathlib import Path

import pandas as pd

from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

def parse_summary_file(path: Path) -> list[dict]:
    """Parse a single chbXX-summary.txt file into a list of file dictionaries."""
    with open(path, 'r') as f:
        text = f.read()

    results = []
    # Split text by "File Name: " to isolate each file's block
    blocks = text.split("File Name: ")

    for block in blocks[1:]:
        lines = block.strip().split('\n')
        file_name = lines[0].strip()

        start_time_match = re.search(r"File Start Time:\s*(.+)", block)
        end_time_match = re.search(r"File End Time:\s*(.+)", block)
        num_seizures_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)

        if not num_seizures_match:
            continue

        num_seizures = int(num_seizures_match.group(1))
        seizures = []

        if num_seizures > 0:
            # Matches both "Seizure Start Time:" and "Seizure 1 Start Time:"
            starts = re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds", block)
            ends = re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds", block)

            for s_start, s_end in zip(starts, ends):
                seizures.append((float(s_start), float(s_end)))

        results.append({
            "file": file_name,
            "start_time": start_time_match.group(1).strip() if start_time_match else "",
            "end_time": end_time_match.group(1).strip() if end_time_match else "",
            "seizures": seizures
        })

    return results

def build_annotations(raw_dir: Path, patient_map: dict) -> pd.DataFrame:
    """Concatenate all parsed summaries into Contract C1 (annotations.csv)."""
    summary_files = list(raw_dir.rglob("*-summary.txt"))
    rows = []

    for summary_path in summary_files:
        case = summary_path.stem.split("-")[0]  # e.g., 'chb01'
        patient = patient_map.get(case, case)   # map chb21 to chb01 if needed

        file_data = parse_summary_file(summary_path)
        for fd in file_data:
            for idx, (s_start, s_end) in enumerate(fd["seizures"]):
                rows.append({
                    "patient": patient,
                    "case": case,
                    "file": fd["file"],
                    "seizure_idx": idx,
                    "seizure_start_s": s_start,
                    "seizure_end_s": s_end
                })

    return pd.DataFrame(rows)
