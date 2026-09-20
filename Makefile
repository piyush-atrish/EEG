# Milestone 1 pipeline. On Windows run this in WSL/Git Bash, or run the python commands directly.
PYTHON ?= python

.PHONY: setup test lint download preprocess windows features baseline report all integration filter-plot

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q -m "not integration and not needs_data"

lint:
	ruff check src tests scripts

download:            # member A
	$(PYTHON) scripts/01_download_and_index.py

preprocess:          # member A
	$(PYTHON) scripts/02_preprocess.py

windows:             # member B
	$(PYTHON) scripts/03_make_windows.py

features:            # member B
	$(PYTHON) scripts/04_extract_features.py

baseline:            # member C
	$(PYTHON) scripts/05_run_baseline.py

report:              # member C
	$(PYTHON) scripts/06_make_report.py

all: download preprocess windows features baseline report

integration:
	$(PYTHON) -m pytest -q -m integration

filter-plot:         # member A: regenerate docs/figures/A_fir_response.png
	$(PYTHON) -m eegpipe.preprocessing.filters
