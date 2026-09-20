from pathlib import Path
from unittest.mock import patch

import pytest

from eegpipe.io import download as dl


class FakeResp:
    def __init__(self, payload=b"x" * 100, content_length=None, status=200):
        self.payload = payload
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise OSError(f"HTTP {self.status}")

    def iter_content(self, chunk_size):
        yield self.payload


@pytest.fixture()
def cfg(tmp_cfg):
    return tmp_cfg


@pytest.fixture(autouse=True)
def no_sleep():
    with patch.object(dl.time, "sleep"):
        yield


def _wfdb_writes(payload=b"edf-bytes"):
    def fake(db, dl_dir, files, keep_subdirs=True, overwrite=False):
        for rel in files:
            target = Path(dl_dir) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return fake


def test_wfdb_success_is_atomic_and_cleans_up(cfg):
    with patch.object(dl.wfdb, "dl_files", side_effect=_wfdb_writes()):
        assert dl.fetch_file(cfg, "chb01/chb01_01.edf")
    raw = Path(cfg["paths"]["raw"])
    assert (raw / "chb01" / "chb01_01.edf").read_bytes() == b"edf-bytes"
    assert not list((raw / ".partial").glob("*"))          # temp dirs removed


def test_existing_nonempty_file_is_not_downloaded_again(cfg):
    target = Path(cfg["paths"]["raw"]) / "chb01" / "a.edf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"data")
    with patch.object(dl.wfdb, "dl_files") as m:
        assert dl.fetch_file(cfg, "chb01/a.edf")
    m.assert_not_called()


def test_empty_existing_file_is_redownloaded(cfg):
    target = Path(cfg["paths"]["raw"]) / "chb01" / "a.edf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    with patch.object(dl.wfdb, "dl_files", side_effect=_wfdb_writes(b"fresh")):
        assert dl.fetch_file(cfg, "chb01/a.edf")
    assert target.read_bytes() == b"fresh"


def test_falls_back_to_https_when_wfdb_raises(cfg):
    with patch.object(dl.wfdb, "dl_files", side_effect=RuntimeError("boom")), \
         patch.object(dl.requests, "get", return_value=FakeResp(b"y" * 10, content_length=10)):
        assert dl.fetch_file(cfg, "chb01/b.edf")
    assert (Path(cfg["paths"]["raw"]) / "chb01" / "b.edf").read_bytes() == b"y" * 10


def test_wfdb_silent_success_without_file_is_a_failure(cfg):
    """wfdb returning normally must not be trusted: the file has to exist afterwards."""
    with patch.object(dl.wfdb, "dl_files", return_value=None), \
         patch.object(dl.requests, "get", return_value=FakeResp(status=404)):
        assert not dl.fetch_file(cfg, "chb01/c.edf", retries=2)
    assert not (Path(cfg["paths"]["raw"]) / "chb01" / "c.edf").exists()


def test_incomplete_https_download_is_rejected(cfg):
    with patch.object(dl.wfdb, "dl_files", side_effect=RuntimeError("boom")), \
         patch.object(dl.requests, "get", return_value=FakeResp(b"y" * 10, content_length=999)):
        assert not dl.fetch_file(cfg, "chb01/d.edf", retries=1)
    assert not (Path(cfg["paths"]["raw"]) / "chb01" / "d.edf").exists()


def test_retries_then_succeeds(cfg):
    calls = {"n": 0}

    def flaky(db, dl_dir, files, keep_subdirs=True, overwrite=False):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temporary")
        _wfdb_writes()(db, dl_dir, files)

    with patch.object(dl.wfdb, "dl_files", side_effect=flaky), \
         patch.object(dl.requests, "get", return_value=FakeResp(status=500)):
        assert dl.fetch_file(cfg, "chb01/e.edf", retries=3)


def test_download_edfs_reports_ok_and_failed(cfg):
    def selective(db, dl_dir, files, keep_subdirs=True, overwrite=False):
        if "bad" in files[0]:
            raise RuntimeError("nope")
        _wfdb_writes()(db, dl_dir, files)

    with patch.object(dl.wfdb, "dl_files", side_effect=selective), \
         patch.object(dl.requests, "get", return_value=FakeResp(status=404)):
        res = dl.download_edfs(cfg, ["chb01/good.edf", "chb01/bad.edf"], max_workers=2)
    assert res["ok"] == ["chb01/good.edf"] and res["failed"] == ["chb01/bad.edf"]


def test_metadata_file_list(cfg):
    files = dl.metadata_files(cfg)
    assert files[:2] == ["RECORDS", "RECORDS-WITH-SEIZURES"]
    assert "chb01/chb01-summary.txt" in files and "chb24/chb24-summary.txt" in files
    assert len(files) == 26
