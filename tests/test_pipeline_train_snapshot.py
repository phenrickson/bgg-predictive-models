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


def test_train_multi_split_with_upstream(tmp_path: Path) -> None:
    """Train complexity, then rating with complexity as upstream, on two splits."""
    base, v = _synthetic_universe(tmp_path)
    # Add a yoy_2020 split (train≤2018, tune=2019, test=2020)
    build_split(
        snapshot_version=v, split_name="yoy_2020",
        train_through=2018, tune_start=2019, tune_through=2019,
        test_start=2020, test_through=2020,
        base_dir=base,
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)

    # Train complexity on both splits
    complexity_cfg = {
        "name": "ard-complexity",
        "algorithm": "ridge",  # ridge is fine for tests; faster than ARD
        "use_embeddings": False,
        "use_sample_weights": False,
    }
    run_pipeline_train(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_config=complexity_cfg,
        splits=["standard", "yoy_2020"], upstream={}, base_dir=base,
    )

    # pipeline.score doesn't exist yet (Task 16 wires it). Synthesize
    # score.parquet for both splits manually so rating training has a
    # column to join.
    universe = storage.load_universe()
    for split_name in ["standard", "yoy_2020"]:
        score_df = universe.select(["game_id"]).with_columns(
            pl.lit(2.5).alias("predicted_complexity")
        )
        result = storage.load_result("complexity", "ard-complexity", 1, split_name)
        assert result is not None, f"Expected complexity result for {split_name}"
        # Re-save the result with score_predictions added
        storage.save_result(
            model_type="complexity",
            candidate="ard-complexity",
            version=1,
            split_name=split_name,
            pipeline=result["pipeline"],
            metrics=result["metrics"],
            parameters=result["parameters"],
            tune_predictions=result.get("tune_predictions"),
            test_predictions=result.get("test_predictions"),
            score_predictions=score_df,
        )

    # Train rating with complexity upstream
    rating_cfg = {
        "name": "ard-ridge-rating",
        "algorithm": "ridge",
        "use_embeddings": False,
        "use_sample_weights": False,
        "min_ratings": 0,  # synthetic data is too small for the default of 5
    }
    run_pipeline_train(
        snapshot_version=v, model_type="rating",
        candidate="ard-ridge-rating", candidate_config=rating_cfg,
        splits=["standard", "yoy_2020"],
        upstream={"complexity": "ard-complexity"},
        base_dir=base,
    )

    # Both splits got results
    standard_result = storage.load_result("rating", "ard-ridge-rating", 1, "standard")
    yoy_result = storage.load_result("rating", "ard-ridge-rating", 1, "yoy_2020")
    assert standard_result is not None and yoy_result is not None

    reg = storage.load_candidate_registration("rating", "ard-ridge-rating", 1)
    assert reg["upstream_experiments"] == {"complexity": "ard-complexity"}


def test_summary_json_written_after_multi_split_training(tmp_path: Path) -> None:
    import json
    base, v = _synthetic_universe(tmp_path)
    build_split(
        snapshot_version=v, split_name="yoy_2020",
        train_through=2018, tune_start=2019, tune_through=2019,
        test_start=2020, test_through=2020,
        base_dir=base,
    )

    cfg = {
        "name": "logistic-hurdle", "algorithm": "logistic",
        "use_embeddings": False, "use_sample_weights": False,
    }
    run_pipeline_train(
        snapshot_version=v, model_type="hurdle",
        candidate="logistic-hurdle", candidate_config=cfg,
        splits=["standard", "yoy_2020"], upstream={}, base_dir=base,
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    summary_path = storage.experiment_dir("hurdle", "logistic-hurdle", 1) / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert sorted(summary["per_split"].keys()) == ["standard", "yoy_2020"]
