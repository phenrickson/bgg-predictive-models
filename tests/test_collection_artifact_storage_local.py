"""Tests for local-filesystem CollectionArtifactStorage.

Hermetic: uses pytest's ``tmp_path`` for all I/O. No BigQuery, no network.
"""

from pathlib import Path

import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.collection.collection_artifact_storage import CollectionArtifactStorage


def _make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )


def _make_storage(tmp_path: Path, username: str = "alice") -> CollectionArtifactStorage:
    return CollectionArtifactStorage(
        username=username,
        local_root=tmp_path / "collections",
        environment="test",
    )


def test_save_and_load_model_roundtrip(tmp_path: Path) -> None:
    storage = _make_storage(tmp_path)
    pipeline = _make_pipeline()

    metadata = {
        "task": "classification",
        "metrics": {"auc": 0.8},
        "best_params": {"model__C": 1.0},
    }
    path = storage.save_model(
        outcome="own",
        pipeline=pipeline,
        metadata=metadata,
        threshold=0.42,
    )
    assert isinstance(path, str)
    assert Path(path).exists()

    version = storage.latest_version("own")
    assert version == 1

    loaded_pipeline, loaded_meta, threshold = storage.load_model(
        outcome="own", version=version
    )

    # Pipeline identity: same class, same named steps.
    assert isinstance(loaded_pipeline, Pipeline)
    assert list(loaded_pipeline.named_steps.keys()) == ["scaler", "model"]
    assert isinstance(loaded_pipeline.named_steps["model"], LogisticRegression)

    # Registration metadata has the expected keys.
    assert loaded_meta["username"] == "alice"
    assert loaded_meta["outcome"] == "own"
    assert loaded_meta["version"] == 1
    assert "created_at" in loaded_meta
    assert loaded_meta["task"] == "classification"
    assert loaded_meta["metrics"] == {"auc": 0.8}
    assert loaded_meta["best_params"] == {"model__C": 1.0}
    assert loaded_meta["threshold"] == 0.42

    assert threshold == 0.42


def test_save_model_auto_increments_version(tmp_path: Path) -> None:
    storage = _make_storage(tmp_path)

    storage.save_model(
        outcome="own",
        pipeline=_make_pipeline(),
        metadata={"task": "classification"},
        threshold=0.5,
    )
    v1 = storage.latest_version("own")
    assert v1 == 1

    storage.save_model(
        outcome="own",
        pipeline=_make_pipeline(),
        metadata={"task": "classification"},
        threshold=0.6,
    )
    v2 = storage.latest_version("own")
    assert v2 == 2

    # Both version directories exist.
    assert (storage.base_dir / "own" / "v1" / "model.pkl").exists()
    assert (storage.base_dir / "own" / "v2" / "model.pkl").exists()


def test_save_splits_writes_three_parquets(tmp_path: Path) -> None:
    storage = _make_storage(tmp_path)

    train_df = pl.DataFrame({"game_id": [1, 2], "y": [1, 0]})
    val_df = pl.DataFrame({"game_id": [3], "y": [1]})
    test_df = pl.DataFrame({"game_id": [4, 5], "y": [0, 1]})

    result = storage.save_splits(
        outcome="own",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    assert result["version"] == 1
    splits_dir = storage.base_dir / "own" / "v1" / "splits"
    assert (splits_dir / "train.parquet").exists()
    assert (splits_dir / "validation.parquet").exists()
    assert (splits_dir / "test.parquet").exists()

    loaded = storage.load_splits("own")
    assert loaded is not None
    assert loaded["version"] == 1
    assert loaded["train"].height == 2
    assert loaded["validation"].height == 1
    assert loaded["test"].height == 2


def test_multiple_outcomes_isolated(tmp_path: Path) -> None:
    storage = _make_storage(tmp_path)

    # Save twice for "own".
    storage.save_model(
        outcome="own",
        pipeline=_make_pipeline(),
        metadata={"task": "classification"},
        threshold=0.5,
    )
    storage.save_model(
        outcome="own",
        pipeline=_make_pipeline(),
        metadata={"task": "classification"},
        threshold=0.5,
    )
    # Save once for "want".
    storage.save_model(
        outcome="want",
        pipeline=_make_pipeline(),
        metadata={"task": "classification"},
        threshold=0.5,
    )

    assert storage.latest_version("own") == 2
    assert storage.latest_version("want") == 1

    own_dir = storage.base_dir / "own"
    want_dir = storage.base_dir / "want"
    assert own_dir.exists() and want_dir.exists()
    assert {p.name for p in own_dir.iterdir() if p.is_dir()} == {"v1", "v2"}
    assert {p.name for p in want_dir.iterdir() if p.is_dir()} == {"v1"}


def test_directory_auto_creation(tmp_path: Path) -> None:
    # Point local_root at a path that does not exist yet.
    nonexistent_root = tmp_path / "nested" / "does" / "not" / "exist"
    assert not nonexistent_root.exists()

    storage = CollectionArtifactStorage(
        username="bob",
        local_root=nonexistent_root,
        environment="test",
    )

    # Storage must have created its base directory chain.
    assert storage.base_dir.exists()
    assert storage.base_dir == nonexistent_root / "test" / "bob"
