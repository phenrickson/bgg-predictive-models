"""Unit tests for CollectionProcessor._load_features SQL assembly and
the embedding-explode helper.

These tests stub out the BigQuery client so they don't touch the network.
"""

from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

from src.collection.collection_processor import (
    CollectionProcessor,
    ProcessorConfig,
)
from src.utils.config import DataWarehouseConfig


@pytest.fixture
def fake_config(monkeypatch):
    """A DataWarehouseConfig whose get_client() returns a MagicMock that
    remembers the last SQL string and returns a controllable pandas DF.

    The test code attaches the expected pandas response by assigning to
    `config._next_df`; the mock returns that DataFrame from
    `client.query(sql).to_dataframe()`.
    """
    cfg = DataWarehouseConfig(
        project_id="my-dw-project",
        features_dataset="analytics",
        features_table="games_features",
    )

    state: dict = {"last_sql": None, "next_df": pd.DataFrame({"game_id": []})}

    def fake_client():
        client = MagicMock()

        def query(sql, *args, **kwargs):
            state["last_sql"] = sql
            job = MagicMock()
            job.to_dataframe.return_value = state["next_df"]
            return job

        client.query.side_effect = query
        return client

    monkeypatch.setattr(cfg, "get_client", fake_client)

    # Attach the state dict so tests can inspect last_sql and swap next_df.
    cfg._state = state  # type: ignore[attr-defined]
    return cfg


def _make_processor(cfg, processor_config: ProcessorConfig) -> CollectionProcessor:
    """Construct a CollectionProcessor without triggering storage init."""
    proc = CollectionProcessor.__new__(CollectionProcessor)
    proc.bq_config = cfg
    proc.environment = "dev"
    proc.processor_config = processor_config
    # storage isn't used by _load_features; leave it unset / None.
    proc.storage = None
    return proc


def test_load_features_base_only_sql(fake_config):
    proc = _make_processor(fake_config, ProcessorConfig())
    proc._load_features()

    sql = fake_config._state["last_sql"]
    assert sql is not None
    assert "games_features" in sql
    assert "year_published IS NOT NULL" in sql
    assert "bgg_complexity_predictions" not in sql
    assert "bgg_description_embeddings" not in sql


def test_load_features_with_predicted_complexity_sql(fake_config):
    proc = _make_processor(
        fake_config,
        ProcessorConfig(use_predicted_complexity=True, use_embeddings=False),
    )
    proc._load_features()

    sql = fake_config._state["last_sql"]
    assert "bgg_complexity_predictions" in sql
    assert "ROW_NUMBER()" in sql
    assert "score_ts DESC" in sql
    assert "bgg_description_embeddings" not in sql


def test_load_features_with_embeddings_sql(fake_config):
    # When use_embeddings=True we still need the to_dataframe() result to
    # round-trip through polars and go through _explode_embeddings, which
    # tolerates a missing `embedding` column. Use an empty frame.
    proc = _make_processor(
        fake_config,
        ProcessorConfig(use_predicted_complexity=False, use_embeddings=True),
    )
    proc._load_features()

    sql = fake_config._state["last_sql"]
    assert "bgg_description_embeddings" in sql
    assert "created_ts DESC" in sql
    assert "bgg_complexity_predictions" not in sql


def test_load_features_with_both_sql(fake_config):
    proc = _make_processor(
        fake_config,
        ProcessorConfig(use_predicted_complexity=True, use_embeddings=True),
    )
    proc._load_features()

    sql = fake_config._state["last_sql"]
    assert "bgg_complexity_predictions" in sql
    assert "bgg_description_embeddings" in sql


def test_embedding_explode_produces_emb_columns():
    df = pl.DataFrame(
        {
            "game_id": [1, 2],
            "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        }
    )

    out = CollectionProcessor._explode_embeddings(df)

    assert "embedding" not in out.columns
    assert "emb_0" in out.columns
    assert "emb_1" in out.columns
    assert "emb_2" in out.columns
    # Values preserved in row order.
    assert out["emb_0"].to_list() == pytest.approx([0.1, 0.4])
    assert out["emb_1"].to_list() == pytest.approx([0.2, 0.5])
    assert out["emb_2"].to_list() == pytest.approx([0.3, 0.6])


def test_embedding_explode_handles_all_null():
    df = pl.DataFrame(
        {"game_id": [1, 2], "embedding": [None, None]},
        schema={"game_id": pl.Int64, "embedding": pl.List(pl.Float64)},
    )

    out = CollectionProcessor._explode_embeddings(df)

    # Embedding column is dropped when nothing to explode.
    assert "embedding" not in out.columns
    assert out.columns == ["game_id"]


def test_embedding_explode_noop_without_embedding_column():
    df = pl.DataFrame({"game_id": [1, 2], "x": [10, 20]})
    out = CollectionProcessor._explode_embeddings(df)
    assert set(out.columns) == {"game_id", "x"}
