"""Cheap repo-wide guards against merge accidents and stray one-off scripts."""

import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(
    p for d in ("src", "scripts", "tests", "tools") for p in (ROOT / d).rglob("*.py")
)
MARKERS = ("<" * 7 + " ", ">" * 7 + " ")  # git conflict markers, built so this file is clean


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_python_file_compiles(path, tmp_path):
    """A botched merge (e.g. 'accept both sides' pasting two modules into one file) fails here."""
    py_compile.compile(str(path), cfile=str(tmp_path / "x.pyc"), doraise=True)


def test_no_conflict_markers_in_source():
    bad = [
        str(p.relative_to(ROOT))
        for p in FILES
        if any(line.startswith(MARKERS) for line in p.read_text(encoding="utf-8").splitlines())
    ]
    assert not bad, f"git conflict markers found in: {bad}"


def test_no_one_off_fix_scripts_in_repo_root():
    """One-off patch scripts (fix_*.py) must not be committed; diagnostics live in tools/."""
    assert not sorted(p.name for p in ROOT.glob("fix_*.py"))


def test_scripts_default_to_repo_relative_config():
    """Scripts must not depend on the current working directory."""
    for script in sorted((ROOT / "scripts").glob("0*.py")):
        assert 'default="configs/config.yaml"' not in script.read_text(), script.name
