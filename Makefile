# eeg-seizure-loso pipeline. On Windows run make in Git Bash/WSL, or run the python commands directly.
PYTHON ?= python

.PHONY: setup test lint download preprocess windows features baseline report all integration filter-plot

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q -m "not integration and not needs_data"

lint:
	ruff check src tests scripts tools

download:
	$(PYTHON) scripts/01_download_and_index.py

preprocess:
	$(PYTHON) scripts/02_preprocess.py

windows:
	$(PYTHON) scripts/03_make_windows.py

features:
	$(PYTHON) scripts/04_extract_features.py

baseline:
	$(PYTHON) scripts/05_run_baseline.py

report:
	$(PYTHON) scripts/06_make_report.py

all: download preprocess windows features baseline report

integration:
	$(PYTHON) -m pytest -q -m integration

filter-plot:         # regenerate docs/figures/A_fir_response.png
	$(PYTHON) -m eegpipe.preprocessing.filters