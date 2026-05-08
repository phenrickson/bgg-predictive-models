"""Hermetic test for build_snapshot CLI using a local-parquet input path."""

from pathlib import Path

import polars as pl

from src.models.build_snapshot import build_snapshot
from src.models.snapshot_storage import SnapshotStorage


def test_build_snapshot_from_local_parquet(tmp_path: Path) -> None:
    # Synthetic source data
    source = pl.DataFrame({
        "game_id": [1, 2, 3, 4],
        "year_published": [2018, 2019, 2020, 2021],
        "rating": [7.0, 8.0, 6.5, 7.2],
        "users_rated": [100, 200, 50, 150],
    })
    source_path = tmp_path / "source.parquet"
    source.write_parquet(source_path)

    base_dir = tmp_path / "snaps"

    version = build_snapshot(
        local_data=source_path,
        base_dir=base_dir,
        use_embeddings=False,
    )
    assert version == 1

    storage = SnapshotStorage(snapshot_version=version, base_dir=base_dir)
    universe = storage.load_universe()
    assert universe is not None
    assert universe.equals(source)

    meta = storage.load_metadata()
    assert meta is not None
    assert meta["n_rows"] == 4
    assert meta["use_embeddings"] is False
    assert "created_at" in meta
    assert meta["columns"] == source.columns


def test_build_snapshot_increments_version(tmp_path: Path) -> None:
    base_dir = tmp_path / "snaps"
    source = pl.DataFrame({"game_id": [1], "year_published": [2018]})
    source_path = tmp_path / "src.parquet"
    source.write_parquet(source_path)

    v1 = build_snapshot(local_data=source_path, base_dir=base_dir, use_embeddings=False)
    v2 = build_snapshot(local_data=source_path, base_dir=base_dir, use_embeddings=False)
    assert v1 == 1
    assert v2 == 2
