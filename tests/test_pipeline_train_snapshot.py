"""Smoke test for the snapshot-aware pipeline.train orchestrator.

Hurdle has no upstream dependencies — start there.
"""

import random
from pathlib import Path

import polars as pl

from src.models.build_snapshot import build_snapshot
from src.models.build_split import build_split
from src.models.snapshot_storage import SnapshotStorage
from src.pipeline.train import train as run_pipeline_train


def _synthetic_universe(tmp_path: Path) -> tuple[Path, int]:
    """Build a snapshot universe rich enough for the hurdle preprocessor.

    Mirrors _synthetic_hurdle_frames() from test_train_one.py: includes
    list-typed columns (categories, mechanics, designers, artists, publishers,
    families) and varied numeric values so VarianceThreshold(0) keeps features.
    200 rows spread across 4 years so each fold has ~50 rows.
    """
    base = tmp_path / "snaps"
    rng = random.Random(42)
    n = 200

    cat_pool = [["Strategy"], ["Party Game"], ["Abstract"], ["Wargame"], ["Economic"]]
    mech_pool = [["Area Control"], ["Deck Building"], ["Worker Placement"],
                 ["Auction"], ["Cooperative Game"]]
    designer_pool = [["Designer A"], ["Designer B"], ["Designer C"],
                     ["Designer D"], ["Designer E"]]

    years = [2018] * 50 + [2019] * 50 + [2020] * 50 + [2021] * 50
    hurdle_vals = [1, 0] * (n // 2)  # alternating so each fold has both classes

    df = pl.DataFrame({
        "game_id": list(range(1, n + 1)),
        "year_published": years,
        "users_rated": [rng.randint(5, 200) for _ in range(n)],
        "hurdle": hurdle_vals,
        "num_weights": [rng.randint(1, 30) for _ in range(n)],
        "complexity": [round(rng.uniform(1.0, 5.0), 2) for _ in range(n)],
        "rating": [round(rng.uniform(4.0, 9.0), 2) for _ in range(n)],
        "min_players": [rng.choice([1, 2, 3]) for _ in range(n)],
        "max_players": [rng.choice([2, 4, 6, 8]) for _ in range(n)],
        "min_playtime": [rng.choice([15, 30, 45, 60]) for _ in range(n)],
        "max_playtime": [rng.choice([60, 90, 120, 180]) for _ in range(n)],
        "min_age": [rng.choice([6, 8, 10, 12, 14]) for _ in range(n)],
        "name": [f"game_{i}" for i in range(n)],
        "categories": [rng.choice(cat_pool) for _ in range(n)],
        "mechanics": [rng.choice(mech_pool) for _ in range(n)],
        "designers": [rng.choice(designer_pool) for _ in range(n)],
        "artists": [["Artist A"]] * n,
        "publishers": [["Publisher A"]] * n,
        "families": [["Family A"]] * n,
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


def test_pipeline_train_writes_result_artifacts(tmp_path: Path) -> None:
    base, v = _synthetic_universe(tmp_path)

    candidate_config = {
        "name": "logistic-hurdle",
        "algorithm": "logistic",
        "use_embeddings": False,
        "use_sample_weights": False,
    }

    run_pipeline_train(
        snapshot_version=v,
        model_type="hurdle",
        candidate="logistic-hurdle",
        candidate_config=candidate_config,
        splits=["standard"],
        upstream={},
        base_dir=base,
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    result = storage.load_result("hurdle", "logistic-hurdle", 1, "standard")
    assert result is not None
    assert "pipeline" in result
    assert "metrics" in result
    assert "tune_predictions" in result
    assert "test_predictions" in result

    # Candidate-level config + registration written
    cfg = storage.load_candidate_config("hurdle", "logistic-hurdle", 1)
    assert cfg == candidate_config

    reg = storage.load_candidate_registration("hurdle", "logistic-hurdle", 1)
    assert reg["snapshot_version"] == v
    assert reg["candidate"] == "logistic-hurdle"
    assert reg["splits"] == ["standard"]
    assert reg["upstream_experiments"] == {}
