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


def test_save_and_load_candidate_config_and_registration(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")

    config = {"name": "logistic-hurdle", "algorithm": "logistic", "use_embeddings": True}
    registration = {"snapshot_version": 1, "candidate": "logistic-hurdle",
                    "version": 1, "upstream_experiments": {}}

    storage.save_candidate_config("hurdle", "logistic-hurdle", 1, config)
    storage.save_candidate_registration("hurdle", "logistic-hurdle", 1, registration)

    loaded_cfg = storage.load_candidate_config("hurdle", "logistic-hurdle", 1)
    loaded_reg = storage.load_candidate_registration("hurdle", "logistic-hurdle", 1)
    assert loaded_cfg == config
    assert loaded_reg == registration


def test_save_and_load_finalized_per_split(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")

    obj_a = {"my": "pipeline-a"}
    obj_b = {"my": "pipeline-b"}
    storage.save_finalized_pipeline("hurdle", "logistic-hurdle", 1, "yoy_2021", obj_a)
    storage.save_finalized_pipeline("hurdle", "logistic-hurdle", 1, "yoy_2022", obj_b)

    assert storage.load_finalized_pipeline("hurdle", "logistic-hurdle", 1, "yoy_2021") == obj_a
    assert storage.load_finalized_pipeline("hurdle", "logistic-hurdle", 1, "yoy_2022") == obj_b
    assert storage.load_finalized_pipeline("hurdle", "logistic-hurdle", 1, "missing") is None


def test_save_and_load_result_artifacts(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")

    pipeline_obj = {"pipeline": "obj"}
    metrics = {"train": {"rmse": 0.5}, "tune": {"rmse": 0.6}, "test": {"rmse": 0.7}}
    params = {"alpha": 1.0}
    tune_preds = pl.DataFrame({"game_id": [1, 2], "prediction": [0.5, 0.6], "actual": [0.4, 0.7]})
    test_preds = pl.DataFrame({"game_id": [3], "prediction": [0.8], "actual": [0.7]})
    score_preds = pl.DataFrame({"game_id": [1, 2, 3, 4], "predicted_complexity": [2.0, 2.5, 3.0, 3.5]})

    storage.save_result(
        model_type="complexity",
        candidate="ard-complexity",
        version=1,
        split_name="standard",
        pipeline=pipeline_obj,
        metrics=metrics,
        parameters=params,
        tune_predictions=tune_preds,
        test_predictions=test_preds,
        score_predictions=score_preds,
    )

    loaded = storage.load_result("complexity", "ard-complexity", 1, "standard")
    assert loaded["pipeline"] == pipeline_obj
    assert loaded["metrics"] == metrics
    assert loaded["parameters"] == params
    assert loaded["tune_predictions"].equals(tune_preds)
    assert loaded["test_predictions"].equals(test_preds)
    assert loaded["score_predictions"].equals(score_preds)


def test_load_score_predictions_helper(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    score = pl.DataFrame({"game_id": [1, 2], "predicted_complexity": [2.0, 2.5]})
    storage.save_result(
        model_type="complexity", candidate="ard-complexity", version=1,
        split_name="standard", pipeline={}, metrics={}, parameters={},
        score_predictions=score,
    )
    loaded = storage.load_score_predictions("complexity", "ard-complexity", 1, "standard")
    assert loaded.equals(score)


def test_save_and_load_simulation_roundtrip(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")

    registration = {"split_name": "standard", "n_samples": 100}
    metrics = {"complexity": {"rmse_sim": 0.5, "n": 10}}
    predictions = pl.DataFrame({
        "game_id": [1, 2, 3],
        "complexity_median": [2.0, 3.0, 4.0],
        "complexity_actual": [2.1, 3.0, 3.9],
    })

    assert storage.next_simulation_version("default", "standard") == 1
    storage.save_simulation("default", "standard", 1, registration, metrics, predictions)
    assert storage.next_simulation_version("default", "standard") == 2

    loaded = storage.load_simulation("default", "standard", version=1)
    assert loaded is not None
    assert loaded["registration"] == registration
    assert loaded["metrics"] == metrics
    assert loaded["predictions"].equals(predictions)


def test_load_simulation_when_missing(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    assert storage.load_simulation("default", "standard") is None
