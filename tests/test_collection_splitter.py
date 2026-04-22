"""Tests for CollectionSplitter: classification (random + time_based) and regression.

The classification splitter keeps *every* eligible universe game as a
``label=False`` row and stratified-splits by label so each of train/val/test
preserves the overall positive:negative ratio. No downsampling occurs here.
"""

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

def test_classification_split_uses_all_eligible_negatives():
    """Every universe game not in positives (and passing filters) shows up
    as label=False somewhere in the splits — no sampling."""
    universe = pl.DataFrame({
        "game_id": list(range(1, 11)),
        "users_rated": [100] * 10,
        "year_published": [2010] * 10,
    })
    labeled = pl.DataFrame({
        "game_id": [1, 2, 3],
        "users_rated": [100, 100, 100],
        "year_published": [2010, 2010, 2010],
        "label": [True, True, True],
    })
    splitter = CollectionSplitter(
        universe_df=universe,
        classification_config=ClassificationSplitConfig(
            split_mode="stratified_random",
            min_ratings_for_negatives=0,
            min_year_for_negatives=None,
            max_year_for_negatives=None,
            random_seed=42,
        ),
    )
    train, val, test = splitter.split(labeled, _outcome("own", "classification"))
    all_rows = pl.concat([train, val, test], how="vertical_relaxed")
    assert all_rows.filter(pl.col("label") == True).height == 3
    assert all_rows.filter(pl.col("label") == False).height == 7


def test_classification_split_stratified_preserves_ratio():
    """Each of train/val/test should carry ~ the overall positive:negative ratio."""
    # 100-game universe; game_ids 1..20 are positives, 21..100 are available negatives.
    universe = pl.DataFrame({
        "game_id": list(range(1, 101)),
        "users_rated": [100] * 100,
        "year_published": [2010] * 100,
    })
    labeled = pl.DataFrame({
        "game_id": list(range(1, 21)),
        "users_rated": [100] * 20,
        "year_published": [2010] * 20,
        "label": [True] * 20,
    })
    splitter = CollectionSplitter(
        universe_df=universe,
        classification_config=ClassificationSplitConfig(
            split_mode="stratified_random",
            validation_ratio=0.2,
            test_ratio=0.2,
            min_ratings_for_negatives=0,
            random_seed=42,
        ),
    )
    train, val, test = splitter.split(labeled, _outcome("own", "classification"))

    overall_ratio = 20 / 80  # positives / negatives = 0.25
    for name, df in [("train", train), ("val", val), ("test", test)]:
        n_pos = df.filter(pl.col("label") == True).height
        n_neg = df.filter(pl.col("label") == False).height
        assert n_pos > 0 and n_neg > 0, f"{name} missing a class"
        ratio = n_pos / n_neg
        # Rounding from int() slicing can skew small buckets; allow ~25% tolerance.
        assert abs(ratio - overall_ratio) < 0.1, (
            f"{name} ratio {ratio:.3f} != overall {overall_ratio:.3f}"
        )


def test_classification_split_respects_min_ratings_for_negatives():
    """Universe games below the ratings threshold are excluded entirely."""
    universe = pl.DataFrame({
        "game_id": list(range(1, 21)),
        # game_ids 1..10: low users_rated (10); 11..20: high (100)
        "users_rated": [10] * 10 + [100] * 10,
        "year_published": [2010] * 20,
    })
    # Positives are game_ids 1..3 (the positives themselves are not filtered by
    # min_ratings_for_negatives — that filter only applies to the negative pool).
    labeled = pl.DataFrame({
        "game_id": [1, 2, 3],
        "users_rated": [10, 10, 10],
        "year_published": [2010, 2010, 2010],
        "label": [True, True, True],
    })
    splitter = CollectionSplitter(
        universe_df=universe,
        classification_config=ClassificationSplitConfig(
            split_mode="stratified_random",
            min_ratings_for_negatives=50,
            random_seed=42,
        ),
    )
    train, val, test = splitter.split(labeled, _outcome("own", "classification"))
    all_rows = pl.concat([train, val, test], how="vertical_relaxed")
    negs = all_rows.filter(pl.col("label") == False)
    # Only game_ids 11..20 should be negatives (game_ids 4..10 fail min_ratings).
    neg_ids = set(negs["game_id"].to_list())
    assert neg_ids.issubset(set(range(11, 21)))
    # And the low-ratings non-positive games (4..10) must not appear.
    assert neg_ids.isdisjoint({4, 5, 6, 7, 8, 9, 10})


def test_classification_split_respects_year_bounds():
    """min_year_for_negatives excludes older games from the negative pool."""
    universe = pl.DataFrame({
        "game_id": list(range(1, 37)),
        "users_rated": [100] * 36,
        "year_published": list(range(1990, 2026)),  # 1 game per year 1990-2025
    })
    labeled = pl.DataFrame({
        "game_id": [1, 2, 3],
        "users_rated": [100, 100, 100],
        "year_published": [1990, 1991, 1992],
        "label": [True, True, True],
    })
    splitter = CollectionSplitter(
        universe_df=universe,
        classification_config=ClassificationSplitConfig(
            split_mode="stratified_random",
            min_ratings_for_negatives=0,
            min_year_for_negatives=2020,
            random_seed=42,
        ),
    )
    train, val, test = splitter.split(labeled, _outcome("own", "classification"))
    for df in (train, val, test):
        negs = df.filter(pl.col("label") == False)
        if negs.height == 0:
            continue
        assert negs["year_published"].min() >= 2020


def test_stratified_random_rejects_empty_positives():
    splitter = CollectionSplitter(
        universe_df=_universe_df(),
        classification_config=ClassificationSplitConfig(
            split_mode="stratified_random", min_ratings_for_negatives=0
        ),
    )
    empty = pl.DataFrame(
        {"game_id": [], "users_rated": [], "year_published": [], "label": []},
        schema={
            "game_id": pl.Int64, "users_rated": pl.Int64,
            "year_published": pl.Int64, "label": pl.Boolean,
        },
    )
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
            min_ratings_for_negatives=0,
            random_seed=42,
        ),
    )
    train, val, test = splitter.split(
        _labeled_classification_df(years=years), _outcome("own", "classification")
    )

    def _pos_years(df):
        return sorted(df.filter(pl.col("label") == True)["year_published"].to_list())

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


def test_time_based_split_negatives_bucketed_by_year():
    """Negatives are bucketed by year cutoffs the same way positives are."""
    # Universe: one game per year 2018..2024, users_rated high so all eligible.
    universe = pl.DataFrame({
        "game_id": list(range(100, 107)),
        "users_rated": [500] * 7,
        "year_published": list(range(2018, 2025)),
    })
    # Positives across the same year range. Separate game_ids from universe so
    # the negative pool is not empty.
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    labeled = pl.DataFrame({
        "game_id": list(range(1, 8)),
        "users_rated": [500] * 7,
        "year_published": years,
        "label": [True] * 7,
    })
    splitter = CollectionSplitter(
        universe_df=universe,
        classification_config=ClassificationSplitConfig(
            split_mode="time_based",
            train_through=2021,
            prediction_window=2,  # val = 2022-2023
            test_window=2,        # test = 2024-2025
            min_ratings_for_negatives=0,
            random_seed=42,
        ),
    )
    train, val, test = splitter.split(labeled, _outcome("own", "classification"))

    def _neg_years(df):
        return df.filter(pl.col("label") == False)["year_published"].to_list()

    train_neg_years = _neg_years(train)
    val_neg_years = _neg_years(val)
    test_neg_years = _neg_years(test)

    assert train_neg_years, "train should have negatives"
    assert val_neg_years, "val should have negatives"
    assert test_neg_years, "test should have negatives"
    assert all(y <= 2021 for y in train_neg_years)
    assert all(2022 <= y <= 2023 for y in val_neg_years)
    assert all(y >= 2024 for y in test_neg_years)


def test_random_seed_reproducible():
    """Two splitters constructed with the same seed produce identical frames."""
    universe = pl.DataFrame({
        "game_id": list(range(1, 51)),
        "users_rated": [100] * 50,
        "year_published": [2010] * 50,
    })
    labeled = pl.DataFrame({
        "game_id": list(range(1, 11)),
        "users_rated": [100] * 10,
        "year_published": [2010] * 10,
        "label": [True] * 10,
    })
    cfg = ClassificationSplitConfig(
        split_mode="stratified_random",
        validation_ratio=0.2,
        test_ratio=0.2,
        min_ratings_for_negatives=0,
        random_seed=7,
    )
    a = CollectionSplitter(universe_df=universe, classification_config=cfg)
    b = CollectionSplitter(universe_df=universe, classification_config=cfg)
    a_train, a_val, a_test = a.split(labeled, _outcome("own", "classification"))
    b_train, b_val, b_test = b.split(labeled, _outcome("own", "classification"))
    assert a_train.equals(b_train)
    assert a_val.equals(b_val)
    assert a_test.equals(b_test)


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
