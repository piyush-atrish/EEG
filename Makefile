.PHONY: setup test lint clean

setup:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

lint:
	ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +