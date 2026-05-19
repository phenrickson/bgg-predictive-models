from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.collection.gcs_sync as gs


def _blob(name: str):
    """A mocked GCS blob whose download_to_filename actually writes a file."""
    b = MagicMock()
    b.name = name

    def _dl(dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"x")

    b.download_to_filename.side_effect = _dl
    return b


def _client_with_blobs(blobs):
    """Build a mocked storage module whose Client().bucket().list_blobs()
    yields the given blobs and whose bucket().blob() returns fresh mocks."""
    bucket = MagicMock()
    bucket.list_blobs.return_value = iter(blobs)
    bucket.blob.side_effect = lambda name: MagicMock(name=name)
    client = MagicMock()
    client.bucket.return_value = bucket
    storage_mod = MagicMock()
    storage_mod.Client.return_value = client
    return storage_mod, client, bucket


# --- download_prefix ---------------------------------------------------------


def test_download_prefix_empty_is_first_run_not_error(tmp_path):
    """C1/old-behavior crux: empty prefix => 0, no files, no raise."""
    storage_mod, _client, bucket = _client_with_blobs([])
    with patch.object(gs, "storage", storage_mod):
        n = gs.download_prefix("my-bucket", "dev/collections/Gyges", str(tmp_path))
    assert n == 0
    assert list(tmp_path.iterdir()) == []
    bucket.list_blobs.assert_called_once()


def test_download_prefix_downloads_blobs_to_relative_paths(tmp_path):
    prefix = "dev/collections/Gyges"
    blobs = [
        _blob(f"{prefix}/own/model.pkl"),
        _blob(f"{prefix}/own/splits/train.parquet"),
    ]
    storage_mod, _client, _bucket = _client_with_blobs(blobs)
    with patch.object(gs, "storage", storage_mod):
        n = gs.download_prefix("my-bucket", prefix, str(tmp_path))
    assert n == 2
    assert (tmp_path / "own" / "model.pkl").is_file()
    assert (tmp_path / "own" / "splits" / "train.parquet").is_file()


def test_download_prefix_skips_directory_placeholder(tmp_path):
    prefix = "dev/collections/Gyges"
    blobs = [
        _blob(f"{prefix}/"),  # the "directory" placeholder blob
        _blob(f"{prefix}/own/model.pkl"),
    ]
    storage_mod, _client, _bucket = _client_with_blobs(blobs)
    with patch.object(gs, "storage", storage_mod):
        n = gs.download_prefix("my-bucket", prefix, str(tmp_path))
    assert n == 1
    assert (tmp_path / "own" / "model.pkl").is_file()


def test_download_prefix_propagates_client_error(tmp_path):
    """Key safety property: a real/transient client error must propagate,
    NOT be silently swallowed as 'first run'."""
    bucket = MagicMock()
    bucket.list_blobs.side_effect = RuntimeError("transient GCS error")
    client = MagicMock()
    client.bucket.return_value = bucket
    storage_mod = MagicMock()
    storage_mod.Client.return_value = client
    with patch.object(gs, "storage", storage_mod):
        with pytest.raises(RuntimeError, match="transient GCS error"):
            gs.download_prefix("my-bucket", "dev/collections/Gyges", str(tmp_path))


# --- upload_prefix -----------------------------------------------------------


def test_upload_prefix_uploads_all_files_recursively(tmp_path):
    (tmp_path / "own").mkdir()
    (tmp_path / "own" / "model.pkl").write_bytes(b"a")
    (tmp_path / "own" / "splits").mkdir()
    (tmp_path / "own" / "splits" / "train.parquet").write_bytes(b"b")

    bucket = MagicMock()
    made = {}

    def _blob_factory(name):
        m = MagicMock()
        made[name] = m
        return m

    bucket.blob.side_effect = _blob_factory
    client = MagicMock()
    client.bucket.return_value = bucket
    storage_mod = MagicMock()
    storage_mod.Client.return_value = client

    with patch.object(gs, "storage", storage_mod):
        n = gs.upload_prefix("my-bucket", "dev/collections/Gyges", str(tmp_path))

    assert n == 2
    assert "dev/collections/Gyges/own/model.pkl" in made
    assert "dev/collections/Gyges/own/splits/train.parquet" in made
    made["dev/collections/Gyges/own/model.pkl"].upload_from_filename.assert_called_once()
    made[
        "dev/collections/Gyges/own/splits/train.parquet"
    ].upload_from_filename.assert_called_once()


def test_upload_prefix_propagates_error(tmp_path):
    (tmp_path / "model.pkl").write_bytes(b"a")
    bad_blob = MagicMock()
    bad_blob.upload_from_filename.side_effect = RuntimeError("upload failed")
    bucket = MagicMock()
    bucket.blob.return_value = bad_blob
    client = MagicMock()
    client.bucket.return_value = bucket
    storage_mod = MagicMock()
    storage_mod.Client.return_value = client
    with patch.object(gs, "storage", storage_mod):
        with pytest.raises(RuntimeError, match="upload failed"):
            gs.upload_prefix("my-bucket", "dev/collections/Gyges", str(tmp_path))


# --- CLI dispatch ------------------------------------------------------------


def test_cli_dispatches_download(monkeypatch):
    seen = {}

    def fake_download(bucket, prefix, local_dir):
        seen["args"] = (bucket, prefix, local_dir)
        return 5

    monkeypatch.setattr(gs, "download_prefix", fake_download)
    rc = gs.main(
        [
            "download",
            "--bucket",
            "B",
            "--prefix",
            "P",
            "--local-dir",
            "D",
        ]
    )
    assert rc == 0
    assert seen["args"] == ("B", "P", "D")


def test_cli_dispatches_up(monkeypatch):
    seen = {}

    def fake_up(bucket, prefix, local_dir):
        seen["args"] = (bucket, prefix, local_dir)
        return 3

    monkeypatch.setattr(gs, "upload_prefix", fake_up)
    rc = gs.main(["up", "--bucket", "B", "--prefix", "P", "--local-dir", "D"])
    assert rc == 0
    assert seen["args"] == ("B", "P", "D")


def test_cli_returns_nonzero_on_exception(monkeypatch):
    def boom(bucket, prefix, local_dir):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(gs, "upload_prefix", boom)
    rc = gs.main(["up", "--bucket", "B", "--prefix", "P", "--local-dir", "D"])
    assert rc != 0
