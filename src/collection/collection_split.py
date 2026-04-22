"""Train/val/test splitting for user collection models.

Dispatches on OutcomeDefinition.task:
- classification: _ClassificationSplitter
  - mode "stratified_random": split positives and full-universe negatives
    per-class so each split preserves the overall positive:negative ratio
  - mode "time_based": year cutoff via src.models.splitting.time_based_split,
    applied independently to positives and to eligible negatives
- regression: _RegressionSplitter (no negatives, rated rows only)

Downsampling the negative class for training-time class balance is NOT this
module's concern. The split preserves the real class distribution; class
balancing happens at training time on the model side.
"""

import logging
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import polars as pl

from src.collection.outcomes import OutcomeDefinition
from src.models.splitting import time_based_split

logger = logging.getLogger(__name__)


@dataclass
class ClassificationSplitConfig:
    """Config for classification splitting.

    split_mode selects how positives and negatives are bucketed:
    - "stratified_random": random split per label class (preserves ratio)
    - "time_based": year cutoff; reuses src.models.splitting.time_based_split

    Eligibility filters (``min_ratings_for_negatives``, ``min_year_for_negatives``,
    ``max_year_for_negatives``) restrict which universe games count as valid
    negatives. They are NOT sampling parameters — every eligible negative goes
    into the split; the filters just define "eligible".
    """
    split_mode: Literal["stratified_random", "time_based"] = "stratified_random"

    # stratified_random params
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # time_based params (train_through required when split_mode == "time_based")
    train_through: Optional[int] = None
    """Last year (inclusive) to include in training data. Required for time_based."""
    prediction_window: int = 2
    """Validation window in years."""
    test_window: Optional[int] = None
    """Test window in years; None means train/val only (2 splits)."""
    time_column: str = "year_published"

    # Negative eligibility filters (applied before split in both modes)
    min_ratings_for_negatives: int = 50
    min_year_for_negatives: Optional[int] = None
    max_year_for_negatives: Optional[int] = None

    random_seed: int = 42


@dataclass
class RegressionSplitConfig:
    """Config for regression splitting. No negative sampling."""
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


SplitTriple = Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]


class _ClassificationSplitter:
    """Split a labeled classification dataframe using full-universe negatives.

    Expected input: labeled_df has ``game_id`` and ``label`` columns. Positives
    are rows where label == True (truthy). Every game in ``universe_df`` whose
    ``game_id`` is not a positive and passes the eligibility filters becomes a
    ``label=False`` row. Positives and negatives are then split independently
    (stratified on label) and concatenated per split, so each of train/val/test
    preserves the overall positive:negative ratio.
    """

    def __init__(self, universe_df: pl.DataFrame, config: ClassificationSplitConfig):
        self.universe_df = universe_df
        self.config = config

    def split(self, labeled_df: pl.DataFrame) -> SplitTriple:
        positives = labeled_df.filter(pl.col("label").cast(pl.Boolean) == True)
        if positives.height == 0:
            raise ValueError("No positive rows (label=True) in classification split input")

        negatives = self._build_negatives(positives)

        if self.config.split_mode == "stratified_random":
            train_pos, val_pos, test_pos = self._stratified_random_split(positives)
            train_neg, val_neg, test_neg = self._stratified_random_split(negatives)
        elif self.config.split_mode == "time_based":
            train_pos, val_pos, test_pos = self._time_based_split(positives)
            train_neg, val_neg, test_neg = self._time_based_split(negatives)
        else:
            raise ValueError(f"Unknown split_mode: {self.config.split_mode!r}")

        return (
            self._align(train_pos, train_neg),
            self._align(val_pos, val_neg),
            self._align(test_pos, test_neg),
        )

    def _build_negatives(self, positives: pl.DataFrame) -> pl.DataFrame:
        """Every eligible universe game not in positives becomes a label=False row."""
        positive_ids = positives["game_id"].to_list()
        eligible = self.universe_df.filter(~pl.col("game_id").is_in(positive_ids))
        if self.config.min_ratings_for_negatives > 0:
            eligible = eligible.filter(
                pl.col("users_rated") >= self.config.min_ratings_for_negatives
            )
        if self.config.min_year_for_negatives is not None:
            eligible = eligible.filter(
                pl.col("year_published") >= self.config.min_year_for_negatives
            )
        if self.config.max_year_for_negatives is not None:
            eligible = eligible.filter(
                pl.col("year_published") <= self.config.max_year_for_negatives
            )
        return eligible.with_columns(pl.lit(False).alias("label"))

    def _stratified_random_split(self, df: pl.DataFrame) -> SplitTriple:
        """Shuffle-and-slice into (train, val, test) by configured ratios.

        Applied separately to positives and negatives so the concat in
        :meth:`split` preserves the overall positive:negative ratio in each
        bucket.
        """
        if df.height == 0:
            empty = df.head(0)
            return empty, empty, empty
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=self.config.random_seed)
        n = shuffled.height
        n_test = int(n * self.config.test_ratio)
        n_val = int(n * self.config.validation_ratio)
        test_df = shuffled[:n_test]
        val_df = shuffled[n_test : n_test + n_val]
        train_df = shuffled[n_test + n_val :]
        return train_df, val_df, test_df

    def _time_based_split(self, df: pl.DataFrame) -> SplitTriple:
        cfg = self.config
        if cfg.train_through is None:
            raise ValueError("time_based split requires config.train_through")
        if cfg.time_column not in df.columns:
            raise ValueError(
                f"time_based split requires column {cfg.time_column!r}"
            )
        if df.height == 0:
            empty = df.head(0)
            return empty, empty, empty
        result = time_based_split(
            df,
            train_through=cfg.train_through,
            prediction_window=cfg.prediction_window,
            test_window=cfg.test_window,
            time_col=cfg.time_column,
            return_dict=False,
        )
        # time_based_split returns 2-tuple when test_window is None, 3-tuple otherwise.
        if cfg.test_window is None:
            train_df, val_df = result
            test_df = df.head(0)  # empty, same schema
        else:
            train_df, val_df, test_df = result
        return train_df, val_df, test_df

    @staticmethod
    def _align(pos: pl.DataFrame, neg: pl.DataFrame) -> pl.DataFrame:
        """Concat positives + negatives keeping only columns present in both.

        Positives carry collection-specific columns (user_rating, owned, etc.) that
        negatives lack. Dropping them via intersection avoids label leakage through
        features that could never be present at prediction time.
        """
        common = [c for c in pos.columns if c in neg.columns]
        return pl.concat([pos.select(common), neg.select(common)], how="vertical_relaxed")


class _RegressionSplitter:
    """Split a labeled regression dataframe with no negative sampling.

    Expected input: labeled_df has a numeric `label` column already filtered to
    qualifying rows (via the outcome's require filter).
    """

    def __init__(self, universe_df: pl.DataFrame, config: RegressionSplitConfig):
        self.universe_df = universe_df
        self.config = config

    def split(self, labeled_df: pl.DataFrame) -> SplitTriple:
        if labeled_df.height == 0:
            raise ValueError("No rows in regression split input")

        shuffled = labeled_df.sample(
            fraction=1.0, shuffle=True, seed=self.config.random_seed
        )
        n = shuffled.height
        n_test = int(n * self.config.test_ratio)
        n_val = int(n * self.config.validation_ratio)
        test_df = shuffled[:n_test]
        val_df = shuffled[n_test : n_test + n_val]
        train_df = shuffled[n_test + n_val :]
        return train_df, val_df, test_df


class CollectionSplitter:
    """Public dispatcher: picks classification or regression strategy from outcome.task."""

    def __init__(
        self,
        universe_df: pl.DataFrame,
        classification_config: Optional[ClassificationSplitConfig] = None,
        regression_config: Optional[RegressionSplitConfig] = None,
    ):
        self.universe_df = universe_df
        self._classification = _ClassificationSplitter(
            universe_df, classification_config or ClassificationSplitConfig()
        )
        self._regression = _RegressionSplitter(
            universe_df, regression_config or RegressionSplitConfig()
        )

    def split(self, labeled_df: pl.DataFrame, outcome: OutcomeDefinition) -> SplitTriple:
        if outcome.task == "classification":
            return self._classification.split(labeled_df)
        if outcome.task == "regression":
            return self._regression.split(labeled_df)
        raise ValueError(f"Unsupported outcome task: {outcome.task!r}")


def downsample_negatives(
    df: pl.DataFrame,
    ratio: float,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Downsample the negatives (label=False rows) to a target ratio of
    negatives-per-positive. Positives are preserved.

    Use this at training time on the training DataFrame only — never on
    validation or test. The split itself should preserve the real class
    distribution; downsampling is a training-time class-balance concern.

    Args:
        df: DataFrame with a boolean `label` column.
        ratio: Target negatives-per-positive (e.g., 5.0 → 5 negatives per positive).
        random_seed: Seed for the negative sampling RNG.

    Returns:
        DataFrame with all positives plus `int(n_positives * ratio)` negatives,
        or all negatives if the pool is smaller than the target.
    """
    if "label" not in df.columns:
        raise ValueError("downsample_negatives requires a 'label' column")
    if ratio <= 0:
        raise ValueError(f"ratio must be > 0, got {ratio!r}")

    positives = df.filter(pl.col("label").cast(pl.Boolean) == True)
    negatives = df.filter(pl.col("label").cast(pl.Boolean) == False)

    target = int(positives.height * ratio)
    if target >= negatives.height:
        return df

    sampled = negatives.sample(n=target, shuffle=True, seed=random_seed)
    return pl.concat([positives, sampled], how="vertical_relaxed")
