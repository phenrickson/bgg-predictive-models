# tests/test_ml_inputs.py
from unittest.mock import MagicMock

from src.data.ml_inputs import (
    complexity_min_score_ts,
    latest_complexity_sql,
    latest_description_embeddings_sql,
)


def test_description_embeddings_sql_filters_latest_version():
    sql = latest_description_embeddings_sql("p")
    assert "`p.raw.description_embeddings`" in sql
    assert "embedding_version = (SELECT MAX(embedding_version)" in sql
    assert "PARTITION BY game_id ORDER BY created_ts DESC, job_id DESC" in sql
    assert "rn = 1" in sql
    assert sql.strip().startswith("(") and sql.strip().endswith(")")


def test_complexity_sql_prunes_with_bound():
    sql = latest_complexity_sql("2026-03-12 04:00:00+00", "p")
    assert "`p.raw.complexity_predictions`" in sql
    assert "score_ts >= TIMESTAMP('2026-03-12 04:00:00+00')" in sql
    assert "PARTITION BY game_id ORDER BY score_ts DESC, job_id DESC" in sql


def test_complexity_sql_without_bound():
    sql = latest_complexity_sql(None, "p")
    assert "TIMESTAMP(" not in sql
    assert "None" not in sql


def test_complexity_min_score_ts_reads_min_of_latest_per_game():
    client = MagicMock()
    client.query.return_value.result.return_value = [{"bound": "2026-03-12 04:00:00+00"}]
    assert complexity_min_score_ts(client) == "2026-03-12 04:00:00+00"
    sql = client.query.call_args.args[0]
    assert "MAX(score_ts)" in sql and "GROUP BY game_id" in sql and "MIN(" in sql


def test_complexity_min_score_ts_empty_table():
    client = MagicMock()
    client.query.return_value.result.return_value = [{"bound": None}]
    assert complexity_min_score_ts(client) is None
