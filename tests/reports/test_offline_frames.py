"""Unit tests for the schema-correct offline empty-frame helper.

`empty_offline_frame` exists so the report's join/filter cells render
empty (instead of raising `ColumnNotFoundError`) when BQ-backed fetchers
are stubbed in offline/test mode. These tests pin the columns the report
actually reads and prove the representative failing op now succeeds.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.reports.collection_data import empty_offline_frame


def test_games_frame_schema():
    df = empty_offline_frame("games")
    assert df.height == 0
    for col in ("game_id", "name", "year_published", "image",
                "description", "users_rated"):
        assert col in df.columns, f"games stub missing {col}"
    assert df.schema["game_id"] == pl.Int64


def test_upcoming_frame_schema():
    df = empty_offline_frame("upcoming")
    assert df.height == 0
    for col in ("game_id", "predicted_prob"):
        assert col in df.columns, f"upcoming stub missing {col}"
    assert df.schema["game_id"] == pl.Int64


def test_collection_frame_schema():
    df = empty_offline_frame("collection")
    assert df.height == 0
    assert "game_id" in df.columns
    assert df.schema["game_id"] == pl.Int64


def test_join_then_filter_does_not_raise():
    """The exact op that crashed the smoke test on bare pl.DataFrame()."""
    result = (
        empty_offline_frame("upcoming")
        .join(empty_offline_frame("games"), on="game_id", how="inner")
        .filter(pl.col("year_published") > 2024)
    )
    assert result.height == 0


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        empty_offline_frame("nope")
