"""Tests for CollectionSplitter.

The splitter takes a single labeled feature frame (universe ⟕ collection,
with the outcome's label applied) and returns (train, val, test). It does
not build negatives, sample, or join anything.
"""

import polars as pl
import pytest

from src.collection.collection_split import (
    ClassificationSplitConfig,
    CollectionSplitter,
    RegressionSplitConfig,
    downsample_negatives,
)
from src.collection.outcomes import DirectColumnRule, OutcomeDefinition


def _outcome(task: str) -> OutcomeDefinition:
    return OutcomeDefinition(
        name="own", task=task, label_rule=DirectColumnRule(column="label"), require=None
    )


def _classification_frame(n_pos: int, n_neg: int, years=None) -> pl.DataFrame:
    n = n_pos + n_neg
    years = years or [2010] * n
    return pl.DataFrame({
        "game_id": list(range(1, n + 1)),
        "users_rated": [100 * i for i in range(1, n + 1)],
        "year_published": years,
        "label": [True] * n_pos + [False] * n_neg,
    })


# --- classification: stratified_random ---


def test_stratified_random_preserves_label_ratio():
    df = _classification_frame(n_pos=20, n_neg=80)
    splitter = CollectionSplitter(
        classification_config=ClassificationSplitConfig(
            split_mode="stratified_random",
            validation_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
        )
    )
    train, val, test = splitter.split(df, _outcome("classification"))

    assert train.height + val.height + test.height == 100
    # Each split should be ~20% positives (within rounding tolerance).
    for d in (train, val, test):
        if d.height == 0:
            continue
        pos_rate = d.filter(pl.col("label") == True).height / d.height
        assert 0.15 <= pos_rate <= 0.25


def test_stratified_random_rejects_empty_input():
    df = _classification_frame(0, 0)
    splitter = CollectionSplitter(
        classification_config=ClassificationSplitConfig(split_mode="stratified_random")
    )
    with pytest.raises(ValueError, match="No rows"):
        splitter.split(df, _outcome("classification"))


def test_random_seed_reproducible():
    df = _classification_frame(20, 80)
    cfg = ClassificationSplitConfig(split_mode="stratified_random", random_seed=42)

    a_train, a_val, a_test = CollectionSplitter(classification_config=cfg).split(df, _outcome("classification"))
    b_train, b_val, b_test = CollectionSplitter(classification_config=cfg).split(df, _outcome("classification"))

    assert a_train["game_id"].to_list() == b_train["game_id"].to_list()
    assert a_val["game_id"].to_list() == b_val["game_id"].to_list()
    assert a_test["game_id"].to_list() == b_test["game_id"].to_list()


# --- classification: time_based ---


def test_time_based_split_uses_year_cutoff():
    years = [2005, 2006, 2007, 2008, 2009] * 10  # 50 rows
    df = _classification_frame(25, 25, years=years)
    splitter = CollectionSplitter(
        classification_config=ClassificationSplitConfig(
            split_mode="time_based",
            train_through=2007,
            prediction_window=1,  # val = 2008
            test_window=1,        # test = 2009
        )
    )
    train, val, test = splitter.split(df, _outcome("classification"))

    assert train["year_published"].max() <= 2007
    assert val["year_published"].min() == 2008 and val["year_published"].max() == 2008
    assert test["year_published"].min() == 2009 and test["year_published"].max() == 2009


def test_time_based_split_without_test_window_yields_empty_test():
    years = [2005, 2006, 2007, 2008, 2009] * 10
    df = _classification_frame(25, 25, years=years)
    splitter = CollectionSplitter(
        classification_config=ClassificationSplitConfig(
            split_mode="time_based",
            train_through=2007,
            prediction_window=2,
            test_window=None,
        )
    )
    train, val, test = splitter.split(df, _outcome("classification"))
    assert train.height > 0
    assert val.height > 0
    assert test.height == 0


def test_time_based_split_rejects_missing_train_through():
    df = _classification_frame(10, 10)
    splitter = CollectionSplitter(
        classification_config=ClassificationSplitConfig(
            split_mode="time_based", train_through=None
        )
    )
    with pytest.raises(ValueError, match="train_through"):
        splitter.split(df, _outcome("classification"))


# --- regression ---


def test_regression_shuffle_split_three_way():
    df = pl.DataFrame({
        "game_id": list(range(1, 101)),
        "users_rated": [100] * 100,
        "year_published": [2010] * 100,
        "label": [7.0] * 100,
    })
    splitter = CollectionSplitter(
        regression_config=RegressionSplitConfig(
            validation_ratio=0.2, test_ratio=0.2, random_seed=42
        )
    )
    train, val, test = splitter.split(df, _outcome("regression"))
    assert train.height + val.height + test.height == 100
    assert val.height == 20
    assert test.height == 20


def test_regression_rejects_empty_input():
    df = pl.DataFrame(schema={"game_id": pl.Int64, "label": pl.Float64})
    splitter = CollectionSplitter()
    with pytest.raises(ValueError, match="No rows"):
        splitter.split(df, _outcome("regression"))


# --- dispatcher ---


def test_splitter_dispatches_on_outcome_task():
    classification_df = _classification_frame(10, 10)
    regression_df = pl.DataFrame({
        "game_id": list(range(1, 11)),
        "users_rated": [100] * 10,
        "year_published": [2010] * 10,
        "label": [7.0] * 10,
    })
    splitter = CollectionSplitter()

    train_c, _, _ = splitter.split(classification_df, _outcome("classification"))
    train_r, _, _ = splitter.split(regression_df, _outcome("regression"))

    assert train_c.height > 0
    assert train_r.height > 0


# --- downsampling helper ---


def test_downsample_negatives_preserves_positives():
    df = pl.DataFrame({
        "game_id": list(range(1, 101)),
        "users_rated": [50] * 100,
        "label": [True] * 10 + [False] * 90,
    })
    out = downsample_negatives(df, ratio=2.0, protect_min_ratings=0)
    assert out.filter(pl.col("label") == True).height == 10
    # 10 positives × ratio=2 = 20 negatives.
    assert out.filter(pl.col("label") == False).height == 20


def test_downsample_protect_min_ratings_keeps_high_rating_negatives():
    df = pl.DataFrame({
        "game_id": list(range(1, 51)),
        "users_rated": [10] * 30 + [100] * 20,  # 30 low, 20 high
        "label": [True] * 5 + [False] * 45,     # 5 pos, 45 neg
    })
    # ratio=2 → target 10 negatives. 20 high-rating ones are protected
    # (more than the target), so all of them are kept and zero low-rating
    # ones come along.
    out = downsample_negatives(df, ratio=2.0, protect_min_ratings=50)
    high = out.filter((pl.col("label") == False) & (pl.col("users_rated") >= 50))
    low = out.filter((pl.col("label") == False) & (pl.col("users_rated") < 50))
    assert high.height == 20
    assert low.height == 0
