"""Unit tests verifying that ``CollectionPipeline`` is aligned with the
walkthrough notebook in five concrete ways:

1. ``PipelineConfig.processor_config`` flows into ``CollectionProcessor``.
2. ``PipelineConfig.downsample_negatives_ratio`` is applied to the train
   split only (not val/test) for classification outcomes.
3. For classification, ``model.evaluate`` is called with the threshold
   returned by ``find_threshold`` — not with the default 0.5.

These are unit tests; they mock out BigQuery, the BGG API, and the on-disk
artifact storage. No real network calls, no disk writes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from src.collection.collection_pipeline import CollectionPipeline, PipelineConfig
from src.collection.collection_processor import ProcessorConfig
from src.collection.outcomes import DirectColumnRule, OutcomeDefinition


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_outcome(name: str = "own", task: str = "classification") -> OutcomeDefinition:
    return OutcomeDefinition(
        name=name,
        task=task,
        label_rule=DirectColumnRule(column="label"),
        require=None,
    )


def _fake_project_config() -> MagicMock:
    """Fake Config object as returned by ``src.utils.config.load_config``."""
    cfg = MagicMock()
    cfg.get_bigquery_config.return_value = MagicMock(name="BigQueryConfig")
    cfg.get_environment_prefix.return_value = "dev"
    cfg.raw_config = {
        "collections": {
            "outcomes": {
                "own": {"task": "classification", "label_from": "label"},
            }
        }
    }
    return cfg


@pytest.fixture
def patched_pipeline_env(tmp_path, monkeypatch):
    """Patch ``load_config`` and ``CollectionArtifactStorage`` so that
    ``CollectionPipeline.__init__`` does not touch real config files or disk.

    Yields a dict of mocks the test can reach into.
    """
    with patch(
        "src.collection.collection_pipeline.load_config",
        return_value=_fake_project_config(),
    ) as mock_load_config, patch(
        "src.collection.collection_pipeline.CollectionArtifactStorage"
    ) as mock_storage_cls:
        storage_instance = MagicMock()
        storage_instance.next_version.return_value = 1
        mock_storage_cls.return_value = storage_instance
        yield {
            "load_config": mock_load_config,
            "storage_cls": mock_storage_cls,
            "storage_instance": storage_instance,
        }


# ---------------------------------------------------------------------------
# Change 1: processor_config flows through
# ---------------------------------------------------------------------------


def test_pipeline_passes_processor_config_through(patched_pipeline_env):
    """Constructing the pipeline with a ProcessorConfig should construct
    CollectionProcessor with that same ProcessorConfig, and feature
    flags on PipelineConfig should flow through to BGGDataLoader.
    """
    pc = ProcessorConfig()
    config = PipelineConfig(
        processor_config=pc,
        use_predicted_complexity=True,
        use_embeddings=False,
    )

    with patch(
        "src.collection.collection_pipeline.CollectionProcessor"
    ) as mock_processor_cls, patch(
        "src.collection.collection_pipeline.BGGDataLoader"
    ) as mock_loader_cls:
        processor_instance = MagicMock()
        processor_instance.process.return_value = pl.DataFrame({"game_id": [1]})
        mock_processor_cls.return_value = processor_instance

        loader_instance = MagicMock()
        loader_instance.load_features.return_value = pl.DataFrame({"game_id": [1]})
        mock_loader_cls.return_value = loader_instance

        pipeline = CollectionPipeline("alice", config)

        pipeline._process_collection()
        pipeline._load_game_universe()

        # CollectionProcessor receives our processor_config verbatim.
        assert mock_processor_cls.call_count == 1
        _, kwargs = mock_processor_cls.call_args
        assert kwargs.get("processor_config") is pc

        # _load_game_universe forwards the feature flags from PipelineConfig.
        loader_instance.load_features.assert_called_once_with(
            use_predicted_complexity=True,
            use_embeddings=False,
        )

        processor_instance.process.assert_called_once_with("alice")


# ---------------------------------------------------------------------------
# Change 3: downsampling applied to train only
# ---------------------------------------------------------------------------


def _labeled_frame(n_pos: int, n_neg: int) -> pl.DataFrame:
    """Tiny labeled frame with label + a dummy feature column."""
    labels = [True] * n_pos + [False] * n_neg
    return pl.DataFrame(
        {
            "game_id": list(range(n_pos + n_neg)),
            "label": labels,
            "x": [0.0] * (n_pos + n_neg),
        }
    )


def test_downsample_applied_to_train_only(patched_pipeline_env):
    """Set downsample_negatives_ratio=2.0. With 10 pos / 100 neg in the train
    frame, the train frame handed to model.train should have exactly
    10 + 20 = 30 rows (ratio * n_pos negatives). Val and test frames are
    unchanged."""

    config = PipelineConfig(
        downsample_negatives_ratio=2.0,
        downsample_protect_min_ratings=0,
    )

    # Pre-baked splits. The train frame has 10 pos / 100 neg; val and test
    # keep the raw class distribution (big negatives count) to verify they
    # are *not* downsampled.
    train_df = _labeled_frame(n_pos=10, n_neg=100)
    val_df = _labeled_frame(n_pos=5, n_neg=50)
    test_df = _labeled_frame(n_pos=5, n_neg=50)

    splitter = MagicMock()
    splitter.split.return_value = (train_df, val_df, test_df)

    outcome = _make_outcome()

    # apply_outcome is a pure function on a DataFrame, but we short-circuit
    # it so the test DataFrame flows through unchanged regardless of whether
    # it has the exact column an outcome would inspect.
    with patch(
        "src.collection.collection_pipeline.apply_outcome",
        side_effect=lambda df, _outcome: df,
    ), patch(
        "src.collection.collection_pipeline.CollectionModel"
    ) as mock_model_cls:
        model_instance = MagicMock()
        captured: Dict[str, pl.DataFrame] = {}

        def fake_tune(td: pl.DataFrame, vd: pl.DataFrame):
            captured["train"] = td
            captured["val"] = vd
            return (
                MagicMock(name="pipeline"),
                {"param": 1},
                pd.DataFrame([{"params": {"param": 1}, "score": 0.5}]),
            )

        def fake_evaluate(_pipeline, df, threshold=None):
            captured.setdefault("evaluated", []).append((df, threshold))
            return {"f1": 0.5}

        model_instance.tune.side_effect = fake_tune
        model_instance.find_threshold.return_value = 0.4
        model_instance.evaluate.side_effect = fake_evaluate
        mock_model_cls.return_value = model_instance

        pipeline = CollectionPipeline("alice", config)
        # Bypass the processor/universe loaders we aren't testing here.
        pipeline._train_one_outcome(
            joined=_labeled_frame(10, 100),
            outcome=outcome,
            splitter=splitter,
        )

    # Ratio 2.0 × 10 positives = 20 negatives kept in train.
    captured_train = captured["train"]
    n_pos_train = captured_train.filter(pl.col("label")).height
    n_neg_train = captured_train.filter(~pl.col("label")).height
    assert n_pos_train == 10, f"expected 10 positives in train, got {n_pos_train}"
    assert n_neg_train == 20, (
        f"expected 20 negatives after downsampling, got {n_neg_train}"
    )

    # Val frame was preserved.
    assert captured["val"].height == val_df.height
    captured_val = captured["val"]
    assert captured_val.filter(~pl.col("label")).height == 50

    # Model.evaluate was called with the un-downsampled val and test frames.
    evaluated: List[Tuple[pl.DataFrame, Any]] = captured["evaluated"]
    assert len(evaluated) == 2
    heights = sorted(df.height for df, _ in evaluated)
    assert heights == sorted([val_df.height, test_df.height]), (
        "val/test frames must not be downsampled"
    )


# ---------------------------------------------------------------------------
# Change 4: classification evaluate uses the optimized threshold
# ---------------------------------------------------------------------------


def test_threshold_used_in_evaluation_for_classification(patched_pipeline_env):
    """For a classification outcome, ``model.evaluate`` must be called with
    the threshold returned by ``find_threshold``. The default-0.5 call
    (no threshold kwarg) would collapse precision/recall/f1 on imbalanced
    outcomes, which is the bug this change fixes."""

    config = PipelineConfig()

    train_df = _labeled_frame(n_pos=5, n_neg=5)
    val_df = _labeled_frame(n_pos=5, n_neg=5)
    test_df = _labeled_frame(n_pos=5, n_neg=5)

    splitter = MagicMock()
    splitter.split.return_value = (train_df, val_df, test_df)

    outcome = _make_outcome(task="classification")

    with patch(
        "src.collection.collection_pipeline.apply_outcome",
        side_effect=lambda df, _outcome: df,
    ), patch(
        "src.collection.collection_pipeline.CollectionModel"
    ) as mock_model_cls:
        model_instance = MagicMock()
        model_instance.tune.return_value = (
            MagicMock(name="pipeline"),
            {},
            pd.DataFrame(),
        )
        model_instance.find_threshold.return_value = 0.37
        model_instance.evaluate.return_value = {"f1": 0.5}
        mock_model_cls.return_value = model_instance

        pipeline = CollectionPipeline("alice", config)
        pipeline._train_one_outcome(
            joined=_labeled_frame(5, 5),
            outcome=outcome,
            splitter=splitter,
        )

    # evaluate should have been called twice (val + test), each with the
    # threshold kwarg set to find_threshold()'s return value.
    assert model_instance.evaluate.call_count == 2
    for call in model_instance.evaluate.call_args_list:
        assert "threshold" in call.kwargs, (
            "evaluate() must be called with threshold=... for classification"
        )
        assert call.kwargs["threshold"] == 0.37

    # And find_threshold should have been called on the val split.
    model_instance.find_threshold.assert_called_once()
    find_threshold_args, _ = model_instance.find_threshold.call_args
    # Second positional arg is val_df.
    assert find_threshold_args[1].height == val_df.height


def test_no_threshold_passed_for_regression(patched_pipeline_env):
    """For regression, find_threshold is never called and threshold stays None
    in the evaluate() call (the contract CollectionModel.evaluate already
    supports)."""

    config = PipelineConfig()

    train_df = _labeled_frame(n_pos=5, n_neg=5)
    val_df = _labeled_frame(n_pos=5, n_neg=5)
    test_df = _labeled_frame(n_pos=5, n_neg=5)

    splitter = MagicMock()
    splitter.split.return_value = (train_df, val_df, test_df)

    outcome = _make_outcome(task="regression")

    with patch(
        "src.collection.collection_pipeline.apply_outcome",
        side_effect=lambda df, _outcome: df,
    ), patch(
        "src.collection.collection_pipeline.CollectionModel"
    ) as mock_model_cls:
        model_instance = MagicMock()
        model_instance.tune.return_value = (
            MagicMock(name="pipeline"),
            {},
            pd.DataFrame(),
        )
        model_instance.evaluate.return_value = {"rmse": 1.0}
        mock_model_cls.return_value = model_instance

        pipeline = CollectionPipeline("alice", config)
        pipeline._train_one_outcome(
            joined=_labeled_frame(5, 5),
            outcome=outcome,
            splitter=splitter,
        )

    model_instance.find_threshold.assert_not_called()
    # Each evaluate call has threshold=None (explicitly).
    for call in model_instance.evaluate.call_args_list:
        assert call.kwargs.get("threshold") is None
