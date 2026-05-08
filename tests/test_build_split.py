"""Hermetic test for build_split CLI."""

from pathlib import Path

import polars as pl
import pytest

from src.models.build_snapshot import build_snapshot
from src.models.build_split import build_split
from src.models.snapshot_storage import SnapshotStorage


def _make_snapshot(tmp_path: Path) -> int:
    df = pl.DataFrame({
        "game_id": list(range(1, 21)),
        "year_published": [2018]*5 + [2019]*5 + [2020]*5 + [2021]*5,
        "rating": [7.0] * 20,
    })
    src = tmp_path / "src.parquet"
    df.write_parquet(src)
    return build_snapshot(
        local_data=src, base_dir=tmp_path / "snaps", use_embeddings=False,
    )


def test_build_standard_split(tmp_path: Path) -> None:
    base = tmp_path / "snaps"
    v = _make_snapshot(tmp_path)

    build_split(
        snapshot_version=v,
        split_name="standard",
        train_through=2019,
        tune_start=2020,
        tune_through=2020,
        test_start=2021,
        test_through=2021,
        base_dir=base,
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    split = storage.load_split("standard")
    assert split is not None
    assert split["train"].height == 10  # 2018 + 2019
    assert split["tune"].height == 5     # 2020
    assert split["test"].height == 5     # 2021

    meta = split["metadata"]
    assert meta["train_through"] == 2019
    assert meta["tune_start"] == 2020
    assert meta["tune_through"] == 2020


def test_build_split_errors_on_missing_snapshot(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_split(
            snapshot_version=99,
            split_name="standard",
            train_through=2019, tune_start=2020, tune_through=2020,
            test_start=2021, test_through=2021,
            base_dir=tmp_path / "snaps",
        )
