"""Tests for SnapshotStorage.

Hermetic: uses pytest's ``tmp_path`` for all I/O. No BigQuery, no network.
"""

from pathlib import Path

import polars as pl

from src.models.snapshot_storage import SnapshotStorage


def test_latest_version_with_no_snapshots(tmp_path: Path) -> None:
    # When no snapshots exist, latest_version returns None.
    base = tmp_path / "snapshots"
    base.mkdir()
    assert SnapshotStorage.latest_version(base_dir=base) is None


def test_next_version_with_no_snapshots(tmp_path: Path) -> None:
    base = tmp_path / "snapshots"
    base.mkdir()
    assert SnapshotStorage.next_version(base_dir=base) == 1


def test_latest_version_picks_highest(tmp_path: Path) -> None:
    base = tmp_path / "snapshots"
    (base / "v1").mkdir(parents=True)
    (base / "v3").mkdir(parents=True)
    (base / "v2").mkdir(parents=True)
    assert SnapshotStorage.latest_version(base_dir=base) == 3
    assert SnapshotStorage.next_version(base_dir=base) == 4


def test_latest_version_ignores_invalid_dirs(tmp_path: Path) -> None:
    base = tmp_path / "snapshots"
    (base / "v1").mkdir(parents=True)
    (base / "vfoo").mkdir()       # not a number after v
    (base / "v2bad").mkdir()       # has trailing junk
    (base / "scratch").mkdir()    # no v prefix
    (base / "v3").mkdir()
    assert SnapshotStorage.latest_version(base_dir=base) == 3


def test_save_and_load_universe_roundtrip(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    df = pl.DataFrame({
        "game_id": [1, 2, 3],
        "year_published": [2018, 2019, 2020],
        "rating": [7.0, 8.0, 6.5],
    })
    storage.save_universe(df)
    loaded = storage.load_universe()
    assert loaded is not None
    assert loaded.equals(df)


def test_save_and_load_metadata_roundtrip(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    meta = {"created_at": "2026-05-08", "n_rows": 3, "use_embeddings": True}
    storage.save_metadata(meta)
    assert storage.load_metadata() == meta


def test_load_universe_when_missing(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    assert storage.load_universe() is None
