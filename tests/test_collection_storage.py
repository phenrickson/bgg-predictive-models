"""Integration tests for CollectionStorage against dev BigQuery.

Requires ADC (gcloud auth application-default login) and ML_PROJECT_ID set.
Uses `_test_*` username prefixes and cleans them up at teardown.
"""

from typing import Iterable

import polars as pl
import pytest

from src.collection.collection_storage import CollectionStorage


pytestmark = pytest.mark.integration


TEST_USER_A = "_test_user_a"
TEST_USER_B = "_test_user_b"


def _row(game_id: int, *, name: str = "G", owned: bool = True,
         user_rating: float | None = None,
         subtype: str = "boardgame") -> dict:
    """Build a minimal collection row matching what the loader emits."""
    return {
        "game_id": game_id,
        "game_name": name,
        "subtype": subtype,
        "collection_id": None,
        "owned": owned,
        "previously_owned": False,
        "for_trade": False,
        "want": False,
        "want_to_play": False,
        "want_to_buy": False,
        "wishlist": False,
        "wishlist_priority": None,
        "preordered": False,
        "last_modified": None,
        "user_rating": user_rating,
        "user_comment": None,
    }


def _df(rows: Iterable[dict]) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


@pytest.fixture
def storage():
    return CollectionStorage(environment="dev")


@pytest.fixture(autouse=True)
def cleanup(storage):
    yield
    storage.delete_user_rows(TEST_USER_A)
    storage.delete_user_rows(TEST_USER_B)


def test_initial_load_inserts_all_rows(storage):
    df = _df([_row(1), _row(2), _row(3)])
    storage.save_collection(TEST_USER_A, df)

    result = storage.get_latest_collection(TEST_USER_A)
    assert result is not None
    assert result.height == 3
    assert set(result["game_id"].to_list()) == {1, 2, 3}
    # first_seen_at == updated_at on fresh insert
    assert result.filter(
        pl.col("first_seen_at") != pl.col("updated_at")
    ).height == 0


def test_idempotent_repull(storage):
    df = _df([_row(1), _row(2)])
    storage.save_collection(TEST_USER_A, df)
    first = storage.get_latest_collection(TEST_USER_A)

    storage.save_collection(TEST_USER_A, df)
    second = storage.get_latest_collection(TEST_USER_A)

    # Still 2 rows, same game_ids, no soft-deletes.
    assert second.height == 2
    assert set(second["game_id"].to_list()) == {1, 2}
    # first_seen_at unchanged; updated_at advanced.
    first_by_id = {row["game_id"]: row for row in first.iter_rows(named=True)}
    second_by_id = {row["game_id"]: row for row in second.iter_rows(named=True)}
    for gid in (1, 2):
        assert first_by_id[gid]["first_seen_at"] == second_by_id[gid]["first_seen_at"]
        assert second_by_id[gid]["updated_at"] >= first_by_id[gid]["updated_at"]


def test_modified_row_updates_data(storage):
    storage.save_collection(TEST_USER_A, _df([_row(1, user_rating=5.0), _row(2)]))
    storage.save_collection(TEST_USER_A, _df([_row(1, user_rating=9.5), _row(2)]))

    result = storage.get_latest_collection(TEST_USER_A)
    rating = result.filter(pl.col("game_id") == 1)["user_rating"].item()
    assert rating == 9.5


def test_new_row_inserts(storage):
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(2)]))
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(2), _row(3)]))

    result = storage.get_latest_collection(TEST_USER_A)
    assert result.height == 3
    assert set(result["game_id"].to_list()) == {1, 2, 3}


def test_removed_row_soft_deletes(storage):
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(2), _row(3)]))
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(3)]))

    # get_latest_collection filters removed_at IS NULL.
    visible = storage.get_latest_collection(TEST_USER_A)
    assert set(visible["game_id"].to_list()) == {1, 3}

    # Raw query to verify game_id=2 has removed_at set.
    raw = storage.get_all_rows_including_removed(TEST_USER_A)
    row2 = raw.filter(pl.col("game_id") == 2)
    assert row2.height == 1
    assert row2["removed_at"].item() is not None


def test_readded_row_clears_removed_at(storage):
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(2)]))
    storage.save_collection(TEST_USER_A, _df([_row(1)]))  # soft-delete 2
    storage.save_collection(TEST_USER_A, _df([_row(1), _row(2)]))  # re-add

    visible = storage.get_latest_collection(TEST_USER_A)
    assert set(visible["game_id"].to_list()) == {1, 2}

    # first_seen_at on game_id=2 should be the original insert, not the re-add.
    raw = storage.get_all_rows_including_removed(TEST_USER_A)
    row2 = raw.filter(pl.col("game_id") == 2)
    assert row2["removed_at"].item() is None


def test_cross_user_isolation(storage):
    storage.save_collection(TEST_USER_A, _df([_row(10), _row(11)]))
    storage.save_collection(TEST_USER_B, _df([_row(20)]))

    # Re-pull B with a different set.
    storage.save_collection(TEST_USER_B, _df([_row(21)]))

    # User A untouched by anything that happened to user B.
    a = storage.get_latest_collection(TEST_USER_A)
    assert set(a["game_id"].to_list()) == {10, 11}

    b = storage.get_latest_collection(TEST_USER_B)
    assert set(b["game_id"].to_list()) == {21}


def test_empty_dataframe_is_rejected(storage):
    with pytest.raises(ValueError, match=r"empty"):
        storage.save_collection(TEST_USER_A, _df([]))


def test_duplicate_rows_are_rejected(storage):
    with pytest.raises(ValueError, match=r"duplicate"):
        storage.save_collection(TEST_USER_A, _df([_row(1), _row(1)]))


def test_expansion_rows_are_filtered_out(storage):
    df = _df([_row(1), _row(2, subtype="boardgameexpansion"), _row(3)])
    storage.save_collection(TEST_USER_A, df)

    result = storage.get_latest_collection(TEST_USER_A)
    assert set(result["game_id"].to_list()) == {1, 3}


def test_get_owned_game_ids_filters_to_owned(storage):
    df = _df([_row(1, owned=True), _row(2, owned=False), _row(3, owned=True)])
    storage.save_collection(TEST_USER_A, df)

    ids = storage.get_owned_game_ids(TEST_USER_A)
    assert set(ids) == {1, 3}
