"""Smoke test for the snapshot-aware simulation orchestrator.

Builds a synthetic snapshot+split, trains+finalizes the four-model
chain, runs simulation on the test fold, asserts the simulation
artifacts are written with the expected shape.

This test is heavy — it trains the full chain on synthetic data and
then runs a Bayesian simulation. ~30-60s on a typical machine.
"""

from pathlib import Path

import polars as pl

from src.models.build_snapshot import build_snapshot
from src.models.build_split import build_split
from src.models.snapshot_storage import SnapshotStorage
from src.pipeline.train import train as run_pipeline_train
from src.pipeline.score import score as run_pipeline_score
from src.pipeline.finalize import finalize as run_pipeline_finalize
from src.pipeline.evaluate_simulation import evaluate_simulation as run_pipeline_simulate


def _synthetic_universe(tmp_path: Path) -> tuple[Path, int]:
    base = tmp_path / "snaps"
    n = 400
    n_per_year = n // 4
    import math
    users_rated_vals = [(50 if i % 2 == 0 else 10) for i in range(n)]
    df = pl.DataFrame({
        "game_id": list(range(1, n + 1)),
        "year_published": ([2018]*n_per_year + [2019]*n_per_year + [2020]*n_per_year + [2021]*n_per_year),
        "users_rated": users_rated_vals,
        "log_users_rated": [math.log1p(u) for u in users_rated_vals],
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
        "geek_rating": [(6.0 + (i % 10) * 0.15) for i in range(n)],
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


def _train_and_finalize_chain(base: Path, v: int) -> None:
    """Train and finalize the four-model chain on the standard split.

    All upstream models use ARD (Bayesian) so simulate_batch can sample
    from their posteriors. geek_rating uses ARD in 'direct' mode.
    Finalized through 2020 so that eval year = 2021 (the next year in the
    synthetic universe).
    """
    cfg_complexity = {"name": "ard-complexity", "algorithm": "ard",
                      "use_embeddings": False, "use_sample_weights": False}
    cfg_rating = {"name": "ard-ridge-rating", "algorithm": "ard",
                  "use_embeddings": False, "use_sample_weights": False, "min_ratings": 0}
    cfg_users_rated = {"name": "ard-ridge-users_rated", "algorithm": "ard",
                       "use_embeddings": False, "use_sample_weights": False, "min_ratings": 0}
    cfg_geek = {"name": "ard-geek_rating", "algorithm": "ard",
                "use_embeddings": False, "min_ratings": 0,
                "include_predictions": True, "mode": "direct"}

    run_pipeline_train(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_config=cfg_complexity,
        splits=["standard"], upstream={}, base_dir=base,
    )
    run_pipeline_score(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_version=1,
        splits=["standard"], upstream={}, base_dir=base,
    )

    run_pipeline_train(
        snapshot_version=v, model_type="rating",
        candidate="ard-ridge-rating", candidate_config=cfg_rating,
        splits=["standard"], upstream={"complexity": "ard-complexity"}, base_dir=base,
    )
    run_pipeline_train(
        snapshot_version=v, model_type="users_rated",
        candidate="ard-ridge-users_rated", candidate_config=cfg_users_rated,
        splits=["standard"], upstream={"complexity": "ard-complexity"}, base_dir=base,
    )

    run_pipeline_score(
        snapshot_version=v, model_type="rating",
        candidate="ard-ridge-rating", candidate_version=1,
        splits=["standard"], upstream={"complexity": "ard-complexity"}, base_dir=base,
    )
    run_pipeline_score(
        snapshot_version=v, model_type="users_rated",
        candidate="ard-ridge-users_rated", candidate_version=1,
        splits=["standard"], upstream={"complexity": "ard-complexity"}, base_dir=base,
    )

    run_pipeline_train(
        snapshot_version=v, model_type="geek_rating",
        candidate="ard-geek_rating", candidate_config=cfg_geek,
        splits=["standard"],
        upstream={"complexity": "ard-complexity",
                  "rating": "ard-ridge-rating",
                  "users_rated": "ard-ridge-users_rated"},
        base_dir=base,
    )

    run_pipeline_finalize(
        snapshot_version=v, model_type="complexity", candidate="ard-complexity",
        candidate_version=1, finalize_through=2020, base_dir=base,
    )
    run_pipeline_finalize(
        snapshot_version=v, model_type="rating", candidate="ard-ridge-rating",
        candidate_version=1, finalize_through=2020, base_dir=base,
        upstream={"complexity": "ard-complexity"},
    )
    run_pipeline_finalize(
        snapshot_version=v, model_type="users_rated", candidate="ard-ridge-users_rated",
        candidate_version=1, finalize_through=2020, base_dir=base,
        upstream={"complexity": "ard-complexity"},
    )
    run_pipeline_finalize(
        snapshot_version=v, model_type="geek_rating", candidate="ard-geek_rating",
        candidate_version=1, finalize_through=2020, base_dir=base,
        upstream={"complexity": "ard-complexity", "rating": "ard-ridge-rating",
                  "users_rated": "ard-ridge-users_rated"},
    )


def test_simulate_writes_simulation_artifacts(tmp_path: Path) -> None:
    base, v = _synthetic_universe(tmp_path)
    _train_and_finalize_chain(base, v)

    candidates = {
        "complexity": "ard-complexity",
        "rating": "ard-ridge-rating",
        "users_rated": "ard-ridge-users_rated",
        "geek_rating": "ard-geek_rating",
    }
    run_pipeline_simulate(
        snapshot_version=v,
        simulation_name="default",
        candidates=candidates,
        n_samples=50,
        base_dir=base,
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    sim = storage.load_simulation("default", version=1)
    assert sim is not None
    assert "predictions" in sim
    assert "metrics" in sim
    assert "registration" in sim

    # Eval year = finalize_through + 1 = 2020 + 1 = 2021
    universe = storage.load_universe()
    eval_year_rows = universe.filter(pl.col("year_published") == 2021)
    assert sim["predictions"].height == eval_year_rows.height
    assert sim["registration"]["eval_year"] == 2021

    for outcome in ["complexity", "rating", "users_rated", "geek_rating"]:
        assert outcome in sim["metrics"]
