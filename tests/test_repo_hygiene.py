"""Cheap repo-wide guards against merge accidents (joint file)."""

import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(p for d in ("src", "scripts", "tests") for p in (ROOT / d).rglob("*.py"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_python_file_compiles(path, tmp_path):
    """A botched merge (e.g. 'accept both sides' pasting two modules into one file) fails here."""
    py_compile.compile(str(path), cfile=str(tmp_path / "x.pyc"), doraise=True)


def test_scripts_default_to_repo_relative_config():
    """Scripts must not depend on the current working directory."""
    for script in sorted((ROOT / "scripts").glob("0*.py")):
        assert 'default="configs/config.yaml"' not in script.read_text(), script.name
