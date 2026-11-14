import logging
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


def test_https_success_is_atomic_and_cleans_up(cfg):
    resp = FakeResp(b"edf-bytes", content_length=9)
    with patch.object(dl.requests, "get", return_value=resp):
        assert dl.fetch_file(cfg, "chb01/chb01_01.edf")
    raw = Path(cfg["paths"]["raw"])
    assert (raw / "chb01" / "chb01_01.edf").read_bytes() == b"edf-bytes"
    assert not list((raw / ".partial").glob("*"))  # temp dirs removed


def test_existing_nonempty_file_is_not_downloaded_again(cfg):
    target = Path(cfg["paths"]["raw"]) / "chb01" / "a.edf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"data")
    with patch.object(dl.requests, "get") as get:
        assert dl.fetch_file(cfg, "chb01/a.edf")
    get.assert_not_called()


def test_empty_existing_file_is_redownloaded(cfg):
    target = Path(cfg["paths"]["raw"]) / "chb01" / "a.edf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    with patch.object(dl.requests, "get", return_value=FakeResp(b"fresh", content_length=5)):
        assert dl.fetch_file(cfg, "chb01/a.edf")
    assert target.read_bytes() == b"fresh"


def test_http_error_fails_after_all_retries(cfg):
    with patch.object(dl.requests, "get", return_value=FakeResp(status=404)) as get:
        assert not dl.fetch_file(cfg, "chb01/c.edf", retries=2)
    assert get.call_count == 2
    assert not (Path(cfg["paths"]["raw"]) / "chb01" / "c.edf").exists()


def test_incomplete_download_is_rejected(cfg):
    with patch.object(dl.requests, "get", return_value=FakeResp(b"y" * 10, content_length=999)):
        assert not dl.fetch_file(cfg, "chb01/d.edf", retries=1)
    assert not (Path(cfg["paths"]["raw"]) / "chb01" / "d.edf").exists()


def test_retries_then_succeeds(cfg):
    responses = [FakeResp(status=500), FakeResp(status=500), FakeResp(b"ok", content_length=2)]
    with patch.object(dl.requests, "get", side_effect=responses) as get:
        assert dl.fetch_file(cfg, "chb01/e.edf", retries=3)
    assert get.call_count == 3
    assert (Path(cfg["paths"]["raw"]) / "chb01" / "e.edf").read_bytes() == b"ok"


def test_download_edfs_reports_ok_and_failed(cfg):
    def selective(url, **kwargs):
        if "bad" in url:
            return FakeResp(status=404)
        return FakeResp(b"good", content_length=4)

    with patch.object(dl.requests, "get", side_effect=selective):
        res = dl.download_edfs(cfg, ["chb01/good.edf", "chb01/bad.edf"], max_workers=2)
    assert res["ok"] == ["chb01/good.edf"] and res["failed"] == ["chb01/bad.edf"]


def test_progress_is_logged_per_file_and_per_step(cfg, caplog):
    with patch.object(dl, "PROGRESS_STEP", 5), patch.object(
        dl.requests, "get", return_value=FakeResp(b"z" * 20, content_length=20)
    ), caplog.at_level(logging.INFO, logger="eegpipe.io.download"):
        dl.download_edfs(cfg, ["chb01/p.edf"], max_workers=1)
    text = caplog.text
    assert "[1/1] OK" in text and "chb01/p.edf" in text
    assert "p.edf: 0 / 0 MB" in text or "p.edf:" in text  # byte-progress line was emitted


def test_metadata_file_list(cfg):
    files = dl.metadata_files(cfg)
    assert files[:2] == ["RECORDS", "RECORDS-WITH-SEIZURES"]
    assert "chb01/chb01-summary.txt" in files and "chb24/chb24-summary.txt" in files
    assert len(files) == 26
