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


def test_build_yoy_splits(tmp_path: Path) -> None:
    from src.models.build_split import build_yoy_splits

    base = tmp_path / "snaps"

    df = pl.DataFrame({
        "game_id": list(range(1, 41)),
        "year_published": sum([[y]*5 for y in range(2014, 2022)], []),
        "rating": [7.0] * 40,
    })
    src = tmp_path / "src.parquet"
    df.write_parquet(src)
    v = build_snapshot(local_data=src, base_dir=base, use_embeddings=False)

    # Years 2018..2020 → splits yoy_2018, yoy_2019, yoy_2020
    # Each test year y → train through y-2, tune y-1, test y
    build_yoy_splits(
        snapshot_version=v,
        yoy_start=2018,
        yoy_end=2020,
        base_dir=base,
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    splits = storage.list_splits()
    assert "yoy_2018" in splits
    assert "yoy_2019" in splits
    assert "yoy_2020" in splits

    # yoy_2019: train≤2017, tune=2018, test=2019
    s = storage.load_split("yoy_2019")
    assert s["metadata"]["train_through"] == 2017
    assert s["metadata"]["tune_start"] == 2018
    assert s["metadata"]["test_start"] == 2019
    # train rows = 2014..2017 = 4 years × 5 = 20
    assert s["train"].height == 20
    assert s["tune"].height == 5
    assert s["test"].height == 5
