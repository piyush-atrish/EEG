"""Rewrite requirements.txt with exact versions taken from the active environment.

Run it inside the virtual environment where the pipeline and tests already work:

    python tools/pin_requirements.py

The package list below is the single place to add a new dependency.
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

RUNTIME = [
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "PyYAML",
    "mne",
    "requests",
    "joblib",
    "tqdm",
    "matplotlib",
    "numba",
    "PyWavelets",
    "scikit-learn",
]
DEV = ["pytest", "ruff", "edfio"]  # edfio: tests write tiny synthetic EDF files

OUT = Path(__file__).resolve().parents[1] / "requirements.txt"


def pinned(names: list[str]) -> list[str]:
    lines, missing = [], []
    for name in names:
        try:
            lines.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        sys.exit(f"Not installed in this environment: {', '.join(missing)}")
    return lines


def main() -> None:
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    parts = [
        f"# Pinned from a working environment (Python {py}).",
        "# Install with:  pip install -r requirements.txt && pip install -e .",
        "# To refresh after adding a dependency: edit RUNTIME/DEV in tools/pin_requirements.py,",
        "# install it, and re-run `python tools/pin_requirements.py`.",
        "",
        "# --- Runtime ---",
        *pinned(RUNTIME),
        "",
        "# --- Dev / test ---",
        *pinned(DEV),
        "",
    ]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT} (Python {py})")


if __name__ == "__main__":
    main()
