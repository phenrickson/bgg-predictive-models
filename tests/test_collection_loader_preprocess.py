"""Unit tests for BGGCollectionLoader._preprocess (dedupe step)."""

from datetime import datetime

import polars as pl
import pytest

from src.collection.collection_loader import BGGCollectionLoader


@pytest.fixture
def loader(monkeypatch):
    monkeypatch.setenv("BGG_API_TOKEN", "fake-token-for-test")
    return BGGCollectionLoader(username="anyuser")


def _row(game_id, collection_id, last_modified=None, owned=False, rating=None):
    return {
        "game_id": game_id,
        "game_name": "G",
        "subtype": "boardgame",
        "collection_id": collection_id,
        "owned": owned,
        "previously_owned": False,
        "for_trade": False,
        "want": False,
        "want_to_play": False,
        "want_to_buy": False,
        "wishlist": False,
        "wishlist_priority": None,
        "preordered": False,
        "last_modified": last_modified,
        "user_rating": rating,
        "user_comment": None,
    }


def test_preprocess_passes_through_unique_rows(loader):
    df = pl.DataFrame(
        [
            _row(1, 100, "2025-01-01 00:00:00"),
            _row(2, 200, "2025-01-02 00:00:00"),
        ]
    )
    out = loader._preprocess(df)
    assert out.height == 2
    assert set(out["game_id"].to_list()) == {1, 2}


def test_preprocess_dedupes_keeping_most_recent_last_modified(loader):
    df = pl.DataFrame(
        [
            _row(1, 100, "2025-01-01 00:00:00", owned=True, rating=5.0),
            _row(1, 200, "2025-06-15 00:00:00", owned=False, rating=9.5),
        ]
    )
    out = loader._preprocess(df)
    assert out.height == 1
    surviving = out.row(0, named=True)
    assert surviving["collection_id"] == 200  # newer row
    assert surviving["owned"] is False
    assert surviving["user_rating"] == 9.5


def test_preprocess_null_last_modified_loses_to_non_null(loader):
    df = pl.DataFrame(
        [
            _row(1, 100, None, owned=True),
            _row(1, 200, "2025-01-01 00:00:00", owned=False),
        ]
    )
    out = loader._preprocess(df)
    assert out.height == 1
    assert out.row(0, named=True)["collection_id"] == 200


def test_preprocess_ties_broken_by_higher_collection_id(loader):
    df = pl.DataFrame(
        [
            _row(1, 100, "2025-01-01 00:00:00", owned=True),
            _row(1, 200, "2025-01-01 00:00:00", owned=False),
        ]
    )
    out = loader._preprocess(df)
    assert out.height == 1
    assert out.row(0, named=True)["collection_id"] == 200


def test_preprocess_both_null_last_modified_tie_broken_deterministically(loader):
    df = pl.DataFrame(
        [
            _row(1, 100, None, owned=True),
            _row(1, 200, None, owned=False),
        ]
    )
    out = loader._preprocess(df)
    assert out.height == 1
    # Higher collection_id wins when last_modified ties (both NULL).
    assert out.row(0, named=True)["collection_id"] == 200


def test_preprocess_multiple_games_each_deduped_independently(loader):
    df = pl.DataFrame(
        [
            _row(1, 100, "2025-01-01 00:00:00"),
            _row(1, 101, "2025-06-01 00:00:00"),  # wins for game_id=1
            _row(2, 200, "2025-06-01 00:00:00"),  # wins for game_id=2
            _row(2, 201, "2025-01-01 00:00:00"),
            _row(3, 300, "2025-01-01 00:00:00"),
        ]
    )
    out = loader._preprocess(df)
    assert out.height == 3
    result_by_game = {r["game_id"]: r["collection_id"] for r in out.iter_rows(named=True)}
    assert result_by_game == {1: 101, 2: 200, 3: 300}
