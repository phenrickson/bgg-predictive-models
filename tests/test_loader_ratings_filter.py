"""Tests for the min_users_rated predicate in change-detection loading.

Scheduled runs score upcoming games regardless of ratings; only the backfill
restricts to rated games, so the predicate must be absent unless asked for.
"""

from unittest.mock import MagicMock

import pandas as pd

from src.data.loader import BGGDataLoader


def _loader_capturing_sql():
    """Return (loader, get_sql) where get_sql yields the query that was issued."""
    config = MagicMock()
    config.project_id = "proj"
    config.dataset = "analytics"
    config.table = "games_features"

    query_job = MagicMock()
    query_job.to_dataframe.return_value = pd.DataFrame()  # short-circuits the load
    config.get_client.return_value.query.return_value = query_job

    loader = BGGDataLoader(config)
    return loader, lambda: loader.client.query.call_args[0][0]


def _load(loader, **kwargs):
    return loader.load_changed_games_with_embeddings(
        start_year=1900,
        end_year=2030,
        ml_project_id="ml-proj",
        **kwargs,
    )


def test_no_ratings_predicate_by_default():
    loader, get_sql = _loader_capturing_sql()

    _load(loader)

    assert "users_rated >=" not in get_sql()


def test_ratings_predicate_applied_when_requested():
    loader, get_sql = _loader_capturing_sql()

    _load(loader, min_users_rated=25)

    assert "gf.users_rated >= 25" in get_sql()


def test_ratings_predicate_does_not_replace_year_filter():
    loader, get_sql = _loader_capturing_sql()

    _load(loader, min_users_rated=25)

    sql = get_sql()
    assert "gf.year_published >= 1900" in sql
    assert "gf.year_published < 2030" in sql


def test_ratings_predicate_is_not_string_interpolated_raw():
    # Guards against a non-integer reaching the SQL text
    loader, get_sql = _loader_capturing_sql()

    _load(loader, min_users_rated="25")

    assert "gf.users_rated >= 25" in get_sql()
