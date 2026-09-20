.PHONY: setup test lint clean all integration 01_download 02_preprocess 03_windows 04_features 05_baseline 06_report

setup:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

integration:
	pytest -q -m integration

lint:
	ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

all: 01_download 02_preprocess 03_windows 04_features 05_baseline 06_report

01_download:
	python scripts/01_download_and_index.py

02_preprocess:
	python scripts/02_preprocess.py

03_windows:
	python scripts/03_make_windows.py

04_features:
	python scripts/04_extract_features.py

05_baseline:
	python scripts/05_run_baseline.py

06_report:
	python scripts/06_make_report.py