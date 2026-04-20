"""Tests for CollectionSplitter: classification (random + time_based) and regression."""

import polars as pl
import pytest

from src.collection.outcomes import OutcomeDefinition, DirectColumnRule
from src.collection.collection_split import (
    CollectionSplitter,
    ClassificationSplitConfig,
    RegressionSplitConfig,
)


def _universe_df() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": list(range(1, 21)),
        "users_rated": [100 * i for i in range(1, 21)],
        "year_published": [2000 + (i % 10) for i in range(1, 21)],
    })


def _labeled_classification_df(years=None) -> pl.DataFrame:
    years = years or [2005] * 10
    return pl.DataFrame({
        "game_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "users_rated": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "year_published": years,
        "label": [True] * 10,
    })


def _labeled_regression_df() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "users_rated": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "year_published": [2005] * 10,
        "label": [6.0, 7.0, 8.0, 9.0, 10.0, 6.5, 7.5, 8.5, 9.5, 7.0],
    })


def _outcome(name: str, task: str) -> OutcomeDefinition:
    return OutcomeDefinition(
        name=name, task=task, label_rule=DirectColumnRule(column="label"), require=None
    )


# --- classification: stratified_random ---

def test_stratified_random_returns_three_splits_with_matched_negatives():
    splitter = CollectionSplitter(
        universe_df=_universe_df(),
        classification_config=ClassificationSplitConfig(
            split_mode="stratified_random",
            negative_sampling_ratio=2.0,
            min_ratings_for_negatives=0,
            random_seed=42,
        ),
    )
    train, val, test = splitter.split(_labeled_classification_df(), _outcome("own", "classification"))
    assert train.height > 0 and val.height > 0 and test.height > 0
    # After negative sampling: label column has both True and False
    assert set(train["label"].to_list()).issubset({True, False})
    n_pos = train.filter(pl.col("label") == True).height
    n_neg = train.filter(pl.col("label") == False).height
    # Negatives are requested at ratio*positives but capped by eligible pool size.
    assert 0 < n_neg <= 2 * n_pos


def test_stratified_random_rejects_empty_positives():
    splitter = CollectionSplitter(
        universe_df=_universe_df(),
        classification_config=ClassificationSplitConfig(
            split_mode="stratified_random", min_ratings_for_negatives=0
        ),
    )
    empty = pl.DataFrame({"game_id": [], "users_rated": [], "year_published": [], "label": []}, schema={
        "game_id": pl.Int64, "users_rated": pl.Int64, "year_published": pl.Int64, "label": pl.Boolean
    })
    with pytest.raises(ValueError, match="positive"):
        splitter.split(empty, _outcome("own", "classification"))


# --- classification: time_based ---

def test_time_based_split_uses_year_cutoff():
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2024, 2023]
    splitter = CollectionSplitter(
        universe_df=_universe_df(),
        classification_config=ClassificationSplitConfig(
            split_mode="time_based",
            train_through=2022,
            prediction_window=2,  # val = 2023-2024
            test_window=2,        # test = 2025-2026
            negative_sampling_ratio=1.0,
            min_ratings_for_negatives=0,
            random_seed=42,
        ),
    )
    train, val, test = splitter.split(
        _labeled_classification_df(years=years), _outcome("own", "classification")
    )
    # Positive year check (negatives come from universe with different year distributions;
    # filter each split's positives to verify)
    def _pos_years(df):
        return sorted(
            df.filter(pl.col("label") == True)["year_published"].to_list()
        )
    assert all(y <= 2022 for y in _pos_years(train))
    assert all(2023 <= y <= 2024 for y in _pos_years(val))
    assert all(2025 <= y <= 2026 for y in _pos_years(test))


def test_time_based_split_without_test_window_yields_empty_test():
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2023, 2024, 2023]
    splitter = CollectionSplitter(
        universe_df=_universe_df(),
        classification_config=ClassificationSplitConfig(
            split_mode="time_based",
            train_through=2022,
            prediction_window=2,
            test_window=None,
            negative_sampling_ratio=1.0,
            min_ratings_for_negatives=0,
            random_seed=42,
        ),
    )
    train, val, test = splitter.split(
        _labeled_classification_df(years=years), _outcome("own", "classification")
    )
    assert train.height > 0
    assert val.height > 0
    # test is empty (no test_window) → no positives and no negatives sampled for it
    assert test.height == 0


def test_time_based_split_rejects_missing_train_through():
    splitter = CollectionSplitter(
        universe_df=_universe_df(),
        classification_config=ClassificationSplitConfig(
            split_mode="time_based", train_through=None, min_ratings_for_negatives=0
        ),
    )
    with pytest.raises(ValueError, match="train_through"):
        splitter.split(_labeled_classification_df(), _outcome("own", "classification"))


# --- regression ---

def test_regression_split_returns_three_splits_without_negatives():
    splitter = CollectionSplitter(
        universe_df=_universe_df(),
        regression_config=RegressionSplitConfig(
            validation_ratio=0.2, test_ratio=0.2, random_seed=42
        ),
    )
    train, val, test = splitter.split(
        _labeled_regression_df(), _outcome("rating", "regression")
    )
    assert train.height + val.height + test.height == 10
    for split_df in (train, val, test):
        assert split_df["label"].dtype == pl.Float64
    all_labels = (
        train["label"].to_list() + val["label"].to_list() + test["label"].to_list()
    )
    assert all(6.0 <= v <= 10.0 for v in all_labels)


# --- dispatcher ---

def test_splitter_dispatches_on_outcome_task():
    splitter = CollectionSplitter(
        universe_df=_universe_df(),
        classification_config=ClassificationSplitConfig(
            split_mode="stratified_random", min_ratings_for_negatives=0, random_seed=42
        ),
        regression_config=RegressionSplitConfig(random_seed=42),
    )
    c_train, _, _ = splitter.split(
        _labeled_classification_df(), _outcome("own", "classification")
    )
    assert "label" in c_train.columns
    r_train, _, _ = splitter.split(
        _labeled_regression_df(), _outcome("rating", "regression")
    )
    assert "label" in r_train.columns
