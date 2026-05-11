"""Tests for pipeline.finalize writing finalized.pkl per split."""

from pathlib import Path

import polars as pl

from src.models.build_snapshot import build_snapshot
from src.models.build_split import build_split
from src.models.snapshot_storage import SnapshotStorage
from src.pipeline.train import train as run_pipeline_train
from src.pipeline.finalize import finalize as run_pipeline_finalize


def _synthetic_universe(tmp_path: Path) -> tuple[Path, int]:
    base = tmp_path / "snaps"
    n = 200
    n_per_year = n // 4
    df = pl.DataFrame({
        "game_id": list(range(1, n + 1)),
        "year_published": ([2018]*n_per_year + [2019]*n_per_year + [2020]*n_per_year + [2021]*n_per_year),
        "users_rated": [(50 if i % 2 == 0 else 10) for i in range(n)],
        "num_weights": [(5 if i % 3 == 0 else 3) for i in range(n)],
        "complexity": [(2.0 + (i % 5) * 0.5) for i in range(n)],
        "rating": [(6.5 + (i % 10) * 0.2) for i in range(n)],
        "min_players": [(2 if i % 2 == 0 else 3) for i in range(n)],
        "max_players": [(4 if i % 2 == 0 else 5) for i in range(n)],
        "playing_time": [(60 if i % 2 == 0 else 90) for i in range(n)],
        "min_age": [(8 if i % 2 == 0 else 12) for i in range(n)],
        "name": [f"game_{i}" for i in range(n)],
        "categories": [["strategy"] if i % 2 == 0 else ["family"] for i in range(n)],
        "mechanics": [["dice"] if i % 2 == 0 else ["cards"] for i in range(n)],
        "designers": [[f"d{i % 5}"] for i in range(n)],
        "artists": [[f"a{i % 3}"] for i in range(n)],
        "publishers": [[f"p{i % 4}"] for i in range(n)],
        "families": [[f"f{i % 6}"] for i in range(n)],
        "hurdle": [1 if i % 2 == 0 else 0 for i in range(n)],
    })
    src = tmp_path / "src.parquet"
    df.write_parquet(src)
    v = build_snapshot(local_data=src, base_dir=base, use_embeddings=False)
    build_split(
        snapshot_version=v, split_name="standard",
        train_through=2019, tune_start=2020, tune_through=2020,
        test_start=2021, test_through=2021,
        base_dir=base,
    )
    return base, v


def test_finalize_writes_per_split_pipeline(tmp_path: Path) -> None:
    base, v = _synthetic_universe(tmp_path)
    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    cfg = {
        "name": "ard-complexity", "algorithm": "ridge",
        "use_embeddings": False, "use_sample_weights": False,
    }

    run_pipeline_train(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_config=cfg,
        splits=["standard"], upstream={}, base_dir=base,
    )

    run_pipeline_finalize(
        snapshot_version=v,
        model_type="complexity",
        candidate="ard-complexity",
        split_name="standard",
        candidate_version=1,
        base_dir=base,
    )

    finalized = storage.load_finalized_pipeline("complexity", "ard-complexity", 1, "standard")
    assert finalized is not None

    reg = storage.load_candidate_registration("complexity", "ard-complexity", 1)
    assert reg["finalize"]["standard"]["finalize_through"] == 2021
    assert "finalized_at" in reg["finalize"]["standard"]
