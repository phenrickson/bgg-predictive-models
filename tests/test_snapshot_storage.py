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


def test_save_and_load_split_roundtrip(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    train = pl.DataFrame({"game_id": [1, 2], "year_published": [2018, 2019]})
    tune = pl.DataFrame({"game_id": [3], "year_published": [2020]})
    test = pl.DataFrame({"game_id": [4], "year_published": [2021]})
    meta = {"train_through": 2019, "tune_start": 2020, "tune_through": 2020,
            "test_start": 2021, "test_through": 2021, "time_col": "year_published"}

    storage.save_split("standard", train, tune, test, meta)
    loaded = storage.load_split("standard")
    assert loaded is not None
    assert loaded["train"].equals(train)
    assert loaded["tune"].equals(tune)
    assert loaded["test"].equals(test)
    assert loaded["metadata"] == meta


def test_load_split_when_missing(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    assert storage.load_split("standard") is None


def test_list_splits(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    df = pl.DataFrame({"game_id": [1], "year_published": [2018]})
    meta = {"x": 1}
    storage.save_split("standard", df, df, df, meta)
    storage.save_split("yoy_2018", df, df, df, meta)
    storage.save_split("yoy_2019", df, df, df, meta)
    assert sorted(storage.list_splits()) == ["standard", "yoy_2018", "yoy_2019"]


def test_experiment_paths(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")

    exp_dir = storage.experiment_dir("hurdle", "logistic-hurdle", 1)
    assert str(exp_dir).endswith(
        "v1/experiments/hurdle/logistic-hurdle/v1"
    )

    result_dir = storage.result_dir("hurdle", "logistic-hurdle", 1, "standard")
    assert str(result_dir).endswith(
        "v1/experiments/hurdle/logistic-hurdle/v1/results/standard"
    )


def test_next_candidate_version(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")

    assert storage.next_candidate_version("hurdle", "logistic-hurdle") == 1
    # Manually create v1 and v2 dirs
    storage.experiment_dir("hurdle", "logistic-hurdle", 1).mkdir(parents=True)
    storage.experiment_dir("hurdle", "logistic-hurdle", 2).mkdir(parents=True)
    assert storage.next_candidate_version("hurdle", "logistic-hurdle") == 3
