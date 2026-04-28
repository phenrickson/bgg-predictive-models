"""Unit tests for BGGDataLoader.load_features SQL assembly and the
embedding-explode helper.

These tests stub out the BigQuery client so they don't touch the network.
"""

from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

from src.data.loader import BGGDataLoader, _explode_embeddings
from src.utils.config import DataWarehouseConfig


@pytest.fixture
def fake_config(monkeypatch):
    """A DataWarehouseConfig whose get_client() returns a MagicMock that
    remembers the last SQL string and returns a controllable pandas DF."""
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
    cfg._state = state  # type: ignore[attr-defined]
    return cfg


def test_load_features_base_only_sql(fake_config):
    loader = BGGDataLoader(fake_config)
    loader.load_features()

    sql = fake_config._state["last_sql"]
    assert sql is not None
    assert "games_features" in sql
    assert "year_published IS NOT NULL" in sql
    assert "bgg_complexity_predictions" not in sql
    assert "bgg_description_embeddings" not in sql


def test_load_features_with_predicted_complexity_sql(fake_config):
    loader = BGGDataLoader(fake_config)
    loader.load_features(use_predicted_complexity=True)

    sql = fake_config._state["last_sql"]
    assert "bgg_complexity_predictions" in sql
    assert "ROW_NUMBER()" in sql
    assert "score_ts DESC" in sql
    assert "bgg_description_embeddings" not in sql


def test_load_features_with_embeddings_sql(fake_config):
    loader = BGGDataLoader(fake_config)
    loader.load_features(use_embeddings=True)

    sql = fake_config._state["last_sql"]
    assert "bgg_description_embeddings" in sql
    assert "created_ts DESC" in sql
    assert "bgg_complexity_predictions" not in sql


def test_load_features_with_both_sql(fake_config):
    loader = BGGDataLoader(fake_config)
    loader.load_features(use_predicted_complexity=True, use_embeddings=True)

    sql = fake_config._state["last_sql"]
    assert "bgg_complexity_predictions" in sql
    assert "bgg_description_embeddings" in sql


def test_load_features_with_where_clause(fake_config):
    loader = BGGDataLoader(fake_config)
    loader.load_features(where_clause="users_rated >= 25")

    sql = fake_config._state["last_sql"]
    assert "users_rated >= 25" in sql
    assert "year_published IS NOT NULL" in sql


def test_embedding_explode_produces_emb_columns():
    df = pl.DataFrame(
        {
            "game_id": [1, 2],
            "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        }
    )

    out = _explode_embeddings(df)

    assert "embedding" not in out.columns
    assert "emb_0" in out.columns
    assert "emb_1" in out.columns
    assert "emb_2" in out.columns
    assert out["emb_0"].to_list() == pytest.approx([0.1, 0.4])
    assert out["emb_1"].to_list() == pytest.approx([0.2, 0.5])
    assert out["emb_2"].to_list() == pytest.approx([0.3, 0.6])


def test_embedding_explode_handles_all_null():
    df = pl.DataFrame(
        {"game_id": [1, 2], "embedding": [None, None]},
        schema={"game_id": pl.Int64, "embedding": pl.List(pl.Float64)},
    )

    out = _explode_embeddings(df)

    assert "embedding" not in out.columns
    assert out.columns == ["game_id"]


def test_embedding_explode_noop_without_embedding_column():
    df = pl.DataFrame({"game_id": [1, 2], "x": [10, 20]})
    out = _explode_embeddings(df)
    assert set(out.columns) == {"game_id", "x"}
