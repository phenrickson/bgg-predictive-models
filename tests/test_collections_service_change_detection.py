"""Tests for services.collections.change_detection."""

from unittest.mock import MagicMock

from services.collections.change_detection import build_unscored_query, find_unscored


def test_build_unscored_query_uses_left_anti_join_against_landing():
    sql = build_unscored_query(
        landing_table="proj.raw.collection_predictions_landing",
        candidate_table="proj.analytics.games_features",
    )
    # Must filter by username, outcome, model_version on the landing side
    assert "username = @username" in sql
    assert "outcome = @outcome" in sql
    assert "model_version = @model_version" in sql
    # Must reference both tables
    assert "proj.raw.collection_predictions_landing" in sql
    assert "proj.analytics.games_features" in sql


def test_find_unscored_returns_game_ids_only_for_missing_rows():
    bq_client = MagicMock()
    row1 = MagicMock(); row1.game_id = 7
    row2 = MagicMock(); row2.game_id = 42
    bq_client.query.return_value.result.return_value = [row1, row2]

    unscored = find_unscored(
        username="alice",
        outcome="own",
        model_version=3,
        landing_table="proj.raw.collection_predictions_landing",
        candidate_table="proj.analytics.games_features",
        bq_client=bq_client,
    )

    assert unscored == [7, 42]
    # Verify the parameters were threaded through
    call = bq_client.query.call_args
    job_config = call.kwargs["job_config"]
    params = {p.name: p.value for p in job_config.query_parameters}
    assert params == {"username": "alice", "outcome": "own", "model_version": 3}
