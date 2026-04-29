"""Tests for src.collection.collection_processor."""

import polars as pl

from src.collection.collection_processor import (
    BGG_TO_CANONICAL,
    CollectionProcessor,
)


def _raw_bgg_df() -> pl.DataFrame:
    """A dataframe with the exact column names the BGG loader emits.

    Mirrors the status fields parsed in CollectionLoader._parse_collection_xml:
    note `previously_owned` (BGG native), which downstream code expects as
    `prev_owned`.
    """
    return pl.DataFrame({
        "game_id": [1, 2, 3],
        "subtype": ["boardgame", "boardgameexpansion", "boardgame"],
        "owned": [True, False, True],
        "previously_owned": [False, True, False],
        "user_rating": [8.0, None, 7.5],
    })


def test_to_canonical_renames_previously_owned():
    out = CollectionProcessor._to_canonical(_raw_bgg_df())
    assert "prev_owned" in out.columns
    assert "previously_owned" not in out.columns


def test_to_canonical_preserves_other_columns():
    out = CollectionProcessor._to_canonical(_raw_bgg_df())
    for col in ("game_id", "subtype", "owned", "user_rating"):
        assert col in out.columns
    assert out["owned"].to_list() == [True, False, True]
    assert out["prev_owned"].to_list() == [False, True, False]


def test_to_canonical_is_noop_when_source_column_absent():
    df = pl.DataFrame({"game_id": [1], "owned": [True]})
    out = CollectionProcessor._to_canonical(df)
    assert out.columns == ["game_id", "owned"]


def test_bgg_to_canonical_map_covers_outcomes_config_columns():
    """The canonical column names referenced in collections.outcomes must either
    come through the loader unchanged or be produced by BGG_TO_CANONICAL."""
    outcome_columns = {"owned", "prev_owned", "user_rating"}
    loader_raw_columns = {
        "game_id", "game_name", "subtype", "collection_id",
        "owned", "previously_owned", "for_trade", "want", "want_to_play",
        "want_to_buy", "wishlist", "wishlist_priority", "preordered",
        "last_modified", "user_rating", "user_comment",
    }
    produced = (loader_raw_columns - set(BGG_TO_CANONICAL)) | set(BGG_TO_CANONICAL.values())
    missing = outcome_columns - produced
    assert not missing, f"Outcomes config references columns not produced by processor: {missing}"
