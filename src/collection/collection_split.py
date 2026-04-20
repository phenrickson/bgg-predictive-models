"""Train/val/test splitting for user collection models.

Dispatches on OutcomeDefinition.task:
- classification: _ClassificationSplitter (with negative sampling)
  - mode "stratified_random": random split per label, matched negatives per split
  - mode "time_based": year cutoff via src.models.splitting.time_based_split
- regression: _RegressionSplitter (no negatives, rated rows only)
"""

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import polars as pl

from src.collection.outcomes import OutcomeDefinition
from src.models.splitting import time_based_split

logger = logging.getLogger(__name__)


@dataclass
class ClassificationSplitConfig:
    """Config for classification splitting.

    split_mode selects how positives (and their matched negatives) are bucketed:
    - "stratified_random": random split per label class
    - "time_based": year cutoff; reuses src.models.splitting.time_based_split
    """
    split_mode: Literal["stratified_random", "time_based"] = "stratified_random"

    # stratified_random params
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # time_based params (all required when split_mode == "time_based")
    train_through: Optional[int] = None
    """Last year (inclusive) to include in training data. Required for time_based."""
    prediction_window: int = 2
    """Validation window in years."""
    test_window: Optional[int] = None
    """Test window in years; None means train/val only (2 splits)."""
    time_column: str = "year_published"

    # negative sampling (both modes)
    negative_sampling_ratio: float = 5.0
    negative_sampling_strategy: str = "popularity_weighted"  # "random" | "popularity_weighted" | "uniform"
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
    """Split a labeled classification dataframe, adding negative samples per split.

    Expected input: labeled_df has `game_id` and `label` columns. Positives are
    rows where label == True (or truthy). Negatives are sampled from universe_df,
    which must have `game_id`, `users_rated`, `year_published`.
    """

    def __init__(self, universe_df: pl.DataFrame, config: ClassificationSplitConfig):
        self.universe_df = universe_df
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)

    def split(self, labeled_df: pl.DataFrame) -> SplitTriple:
        positives = labeled_df.filter(pl.col("label").cast(pl.Boolean) == True)
        if positives.height == 0:
            raise ValueError("No positive rows (label=True) in classification split input")

        if self.config.split_mode == "stratified_random":
            train_pos, val_pos, test_pos = self._stratified_random_split(positives)
        elif self.config.split_mode == "time_based":
            train_pos, val_pos, test_pos = self._time_based_split(positives)
        else:
            raise ValueError(f"Unknown split_mode: {self.config.split_mode!r}")

        positive_ids = set(positives["game_id"].to_list())

        train_neg = self._sample_neg(
            int(train_pos.height * self.config.negative_sampling_ratio), positive_ids
        )
        val_neg = self._sample_neg(
            int(val_pos.height * self.config.negative_sampling_ratio),
            positive_ids | set(train_neg["game_id"].to_list()),
        )
        test_neg = self._sample_neg(
            int(test_pos.height * self.config.negative_sampling_ratio),
            positive_ids
            | set(train_neg["game_id"].to_list())
            | set(val_neg["game_id"].to_list()),
        )

        return (
            self._align(train_pos, train_neg),
            self._align(val_pos, val_neg),
            self._align(test_pos, test_neg),
        )

    def _stratified_random_split(self, positives: pl.DataFrame) -> SplitTriple:
        shuffled = positives.sample(
            fraction=1.0, shuffle=True, seed=self.config.random_seed
        )
        n = shuffled.height
        n_test = int(n * self.config.test_ratio)
        n_val = int(n * self.config.validation_ratio)
        test_pos = shuffled[:n_test]
        val_pos = shuffled[n_test : n_test + n_val]
        train_pos = shuffled[n_test + n_val :]
        return train_pos, val_pos, test_pos

    def _time_based_split(self, positives: pl.DataFrame) -> SplitTriple:
        cfg = self.config
        if cfg.train_through is None:
            raise ValueError("time_based split requires config.train_through")
        if cfg.time_column not in positives.columns:
            raise ValueError(
                f"time_based split requires column {cfg.time_column!r} in positives"
            )
        result = time_based_split(
            positives,
            train_through=cfg.train_through,
            prediction_window=cfg.prediction_window,
            test_window=cfg.test_window,
            time_col=cfg.time_column,
            return_dict=False,
        )
        # time_based_split returns 2-tuple when test_window is None, 3-tuple otherwise.
        if cfg.test_window is None:
            train_pos, val_pos = result
            test_pos = positives.head(0)  # empty, same schema
        else:
            train_pos, val_pos, test_pos = result
        return train_pos, val_pos, test_pos

    def _sample_neg(self, n_samples: int, excluded: set) -> pl.DataFrame:
        eligible = self.universe_df.filter(~pl.col("game_id").is_in(list(excluded)))
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
        if eligible.height == 0 or n_samples == 0:
            return eligible.head(0).with_columns(pl.lit(False).alias("label"))

        k = min(n_samples, eligible.height)
        if self.config.negative_sampling_strategy == "popularity_weighted":
            weights = eligible["users_rated"].to_numpy().astype(float)
            total = weights.sum()
            p = weights / total if total > 0 else None
            indices = self._rng.choice(eligible.height, size=k, replace=False, p=p)
        elif self.config.negative_sampling_strategy in ("random", "uniform"):
            indices = self._rng.choice(eligible.height, size=k, replace=False)
        else:
            raise ValueError(
                f"Unknown negative_sampling_strategy: {self.config.negative_sampling_strategy!r}"
            )

        sampled = eligible[indices.tolist()]
        return sampled.with_columns(pl.lit(False).alias("label"))

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


# ---------------------------------------------------------------------------
# Backward-compatibility shims (used by collection_pipeline.py and cli.py)
# Task 8 will remove these once the pipeline is fully rewritten.
# ---------------------------------------------------------------------------

# SplitConfig is aliased to ClassificationSplitConfig so existing code that
# does `SplitConfig(negative_sampling_ratio=..., negative_sampling_strategy=...)`
# continues to import without error.  The field names are identical.
SplitConfig = ClassificationSplitConfig


class CollectionSplit:
    """Thin shim preserving the old CollectionSplit interface.

    Delegates to the new CollectionSplitter under the hood.  Only the
    create_ownership_splits() path that collection_pipeline currently calls is
    preserved; other methods are not supported.
    """

    def __init__(self, collection_df, game_universe_df, config=None):
        self.collection_df = collection_df
        self.game_universe_df = game_universe_df
        self.config = config or SplitConfig()

    def create_ownership_splits(self, train_through=None, time_column="year_published", return_dict=False):
        import warnings
        warnings.warn(
            "CollectionSplit.create_ownership_splits is deprecated; use CollectionSplitter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Delegate to the old implementation logic inline so the shim is self-contained.
        import numpy as np

        owned_ids = set(
            self.collection_df.filter(pl.col("owned") == True)
            .select("game_id")
            .to_series()
            .to_list()
        )

        owned_df = self.game_universe_df.filter(pl.col("game_id").is_in(list(owned_ids)))

        if len(owned_df) == 0:
            raise ValueError("No owned games found in collection")

        cfg = self.config
        np.random.seed(cfg.random_seed)

        # Split owned games
        if train_through is not None and time_column in owned_df.columns:
            val_year = train_through + 1
            test_start = val_year + 1
            train_owned = owned_df.filter(pl.col(time_column) <= train_through)
            val_owned = owned_df.filter(pl.col(time_column) == val_year)
            test_owned = owned_df.filter(pl.col(time_column) >= test_start)
            if len(val_owned) == 0 or len(test_owned) == 0:
                train_owned, val_owned, test_owned = self._random_split(owned_df, cfg)
        else:
            train_owned, val_owned, test_owned = self._random_split(owned_df, cfg)

        # Sample negatives
        splitter = _ClassificationSplitter(self.game_universe_df, cfg)
        train_neg = splitter._sample_neg(int(len(train_owned) * cfg.negative_sampling_ratio), owned_ids)
        val_neg = splitter._sample_neg(
            int(len(val_owned) * cfg.negative_sampling_ratio),
            owned_ids | set(train_neg["game_id"].to_list()),
        )
        test_neg = splitter._sample_neg(
            int(len(test_owned) * cfg.negative_sampling_ratio),
            owned_ids | set(train_neg["game_id"].to_list()) | set(val_neg["game_id"].to_list()),
        )

        train_owned = train_owned.with_columns(pl.lit(1).alias("target"))
        val_owned = val_owned.with_columns(pl.lit(1).alias("target"))
        test_owned = test_owned.with_columns(pl.lit(1).alias("target"))
        train_neg = train_neg.with_columns(pl.lit(0).alias("target"))
        val_neg = val_neg.with_columns(pl.lit(0).alias("target"))
        test_neg = test_neg.with_columns(pl.lit(0).alias("target"))

        common_train = [c for c in train_owned.columns if c in train_neg.columns]
        common_val = [c for c in val_owned.columns if c in val_neg.columns]
        common_test = [c for c in test_owned.columns if c in test_neg.columns]

        train_df = pl.concat([train_owned.select(common_train), train_neg.select(common_train)])
        val_df = pl.concat([val_owned.select(common_val), val_neg.select(common_val)])
        test_df = pl.concat([test_owned.select(common_test), test_neg.select(common_test)])

        train_df = train_df.sample(fraction=1.0, seed=cfg.random_seed)
        val_df = val_df.sample(fraction=1.0, seed=cfg.random_seed + 1)
        test_df = test_df.sample(fraction=1.0, seed=cfg.random_seed + 2)

        if return_dict:
            return {"train": train_df, "validation": val_df, "test": test_df}
        return train_df, val_df, test_df

    @staticmethod
    def _random_split(df, cfg):
        n = len(df)
        n_val = int(n * cfg.validation_ratio)
        n_test = int(n * cfg.test_ratio)
        n_train = n - n_val - n_test
        shuffled = df.sample(fraction=1.0, seed=cfg.random_seed)
        return shuffled.head(n_train), shuffled.slice(n_train, n_val), shuffled.tail(n_test)
