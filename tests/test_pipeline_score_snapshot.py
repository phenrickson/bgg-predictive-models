"""Tests for pipeline.score writing score.parquet to the snapshot tree."""

from pathlib import Path

import polars as pl

from src.models.build_snapshot import build_snapshot
from src.models.build_split import build_split
from src.models.snapshot_storage import SnapshotStorage
from src.pipeline.train import train as run_pipeline_train
from src.pipeline.score import score as run_pipeline_score


def _synthetic_universe(tmp_path: Path) -> tuple[Path, int]:
    """Same fixture used by test_pipeline_train_snapshot.

    Builds a 200-row synthetic universe across 4 years with feature-rich
    columns the BGG preprocessor can handle, plus a 'standard' split.
    """
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


def test_pipeline_score_writes_score_parquet(tmp_path: Path) -> None:
    base, v = _synthetic_universe(tmp_path)
    storage = SnapshotStorage(snapshot_version=v, base_dir=base)

    # Train complexity first
    complexity_cfg = {
        "name": "ard-complexity", "algorithm": "ridge",
        "use_embeddings": False, "use_sample_weights": False,
    }
    run_pipeline_train(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_config=complexity_cfg,
        splits=["standard"], upstream={}, base_dir=base,
    )

    # Score
    run_pipeline_score(
        snapshot_version=v,
        model_type="complexity",
        candidate="ard-complexity",
        candidate_version=1,
        splits=["standard"],
        upstream={},
        base_dir=base,
    )

    score = storage.load_score_predictions("complexity", "ard-complexity", 1, "standard")
    assert score is not None
    assert score.height == 200  # full universe
    assert "game_id" in score.columns
    assert "predicted_complexity" in score.columns


def test_score_uses_oof_for_upstream_train_rows(tmp_path: Path, monkeypatch) -> None:
    """When scoring an upstream model, kfold_oof_predict is invoked for train rows."""
    base, v = _synthetic_universe(tmp_path)

    # Train complexity with a small k so the test is fast
    cfg = {
        "name": "ard-complexity", "algorithm": "ridge",
        "use_embeddings": False, "use_sample_weights": False,
        "oof_folds": 3,
    }
    run_pipeline_train(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_config=cfg,
        splits=["standard"], upstream={}, base_dir=base,
    )

    calls = []
    from src.models import oof as _oof
    real = _oof.kfold_oof_predict

    def spy(*args, **kwargs):
        calls.append(kwargs.get("k"))
        return real(*args, **kwargs)

    # Patch where the function is LOOKED UP, not where it's defined.
    # If pipeline.score does `from src.models.oof import kfold_oof_predict`
    # then we need to patch `src.pipeline.score.kfold_oof_predict`.
    # If it does `from src.models import oof; oof.kfold_oof_predict(...)`
    # then we patch `src.models.oof.kfold_oof_predict`.
    # Use the latter (module attr) as it's the more flexible default.
    monkeypatch.setattr(_oof, "kfold_oof_predict", spy)

    run_pipeline_score(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_version=1,
        splits=["standard"], upstream={}, base_dir=base,
    )

    assert calls, "kfold_oof_predict was not called for upstream model"
    assert calls[0] == 3, f"OOF was not called with k=3, got k={calls[0]}"
