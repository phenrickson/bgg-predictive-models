"""Train/val/test splitting for user collection ownership prediction."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple, Union

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for collection data splits."""

    negative_sampling_ratio: float = 5.0
    """Ratio of negative samples to positive samples (owned games)."""

    negative_sampling_strategy: str = "popularity_weighted"
    """Strategy for sampling negatives: 'random', 'popularity_weighted', or 'uniform'."""

    min_ratings_for_negatives: int = 50
    """Minimum users_rated for a game to be sampled as negative."""

    min_year_for_negatives: Optional[int] = None
    """Minimum year_published for negative samples."""

    max_year_for_negatives: Optional[int] = None
    """Maximum year_published for negative samples."""

    validation_ratio: float = 0.15
    """Fraction of owned games to use for validation (when not time-based)."""

    test_ratio: float = 0.15
    """Fraction of owned games to use for test (when not time-based)."""

    random_seed: int = 42
    """Random seed for reproducibility."""


class CollectionSplit:
    """Create train/val/test splits for user collection ownership prediction.

    This class handles the creation of training data for ownership prediction models.
    The key challenge is class imbalance: users own ~100-500 games out of ~30,000+
    available games. This is addressed through negative sampling strategies.

    Positive class: Games the user owns (owned=True)
    Negative class: Sampled from games the user doesn't own

    Supports two splitting approaches:
    1. Time-based: Split owned games by when they were added to collection
    2. Random: Stratified random split of owned games

    Example usage:
        >>> splitter = CollectionSplit(collection_df, game_universe_df)
        >>> train_df, val_df, test_df = splitter.create_ownership_splits()
    """

    def __init__(
        self,
        collection_df: pl.DataFrame,
        game_universe_df: pl.DataFrame,
        config: Optional[SplitConfig] = None,
    ):
        """Initialize splitter with collection and game universe.

        Args:
            collection_df: User's collection with game features.
                Required columns: game_id, owned
                Optional: last_modified (for time-based splits)
            game_universe_df: All games with features for negative sampling.
                Required columns: game_id, users_rated, year_published
            config: Split configuration
        """
        self.collection_df = collection_df
        self.game_universe_df = game_universe_df
        self.config = config or SplitConfig()

        # Validate required columns
        self._validate_inputs()

        # Set random seed
        np.random.seed(self.config.random_seed)

        # Get owned game IDs
        self.owned_game_ids = set(
            self.collection_df.filter(pl.col("owned") == True)
            .select("game_id")
            .to_series()
            .to_list()
        )

        logger.info(f"Initialized splitter with {len(self.owned_game_ids)} owned games")
        logger.info(f"Game universe has {len(self.game_universe_df)} games")

    def _validate_inputs(self) -> None:
        """Validate input DataFrames have required columns."""
        required_collection = {"game_id", "owned"}
        required_universe = {"game_id", "users_rated", "year_published"}

        missing_collection = required_collection - set(self.collection_df.columns)
        if missing_collection:
            raise ValueError(f"Collection missing columns: {missing_collection}")

        missing_universe = required_universe - set(self.game_universe_df.columns)
        if missing_universe:
            raise ValueError(f"Game universe missing columns: {missing_universe}")

    def create_ownership_splits(
        self,
        train_end_year: Optional[int] = None,
        time_column: str = "year_published",
        return_dict: bool = False,
    ) -> Union[
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame], Dict[str, pl.DataFrame]
    ]:
        """Create train/val/test splits for ownership prediction.

        Args:
            train_end_year: For time-based splits, the year to end training data.
                If None, uses random stratified splitting.
            time_column: Column to use for time-based splitting.
                Default is 'year_published' (splits by game publication year).
            return_dict: If True, return dict; if False, return tuple.

        Returns:
            If return_dict is False:
                (train_df, val_df, test_df)
            If return_dict is True:
                {"train": train_df, "validation": val_df, "test": test_df}
        """
        # Get owned games with features
        owned_df = self._get_owned_with_features()

        if len(owned_df) == 0:
            raise ValueError("No owned games found in collection")

        logger.info(f"Creating splits for {len(owned_df)} owned games")

        # Split owned games into train/val/test
        if train_end_year is not None and time_column in owned_df.columns:
            train_owned, val_owned, test_owned = self._time_based_split_owned(
                owned_df, train_end_year, time_column
            )
        else:
            train_owned, val_owned, test_owned = self._random_split_owned(owned_df)

        # Sample negatives for each split
        train_negatives = self._sample_negatives(
            n_samples=int(len(train_owned) * self.config.negative_sampling_ratio),
            excluded_ids=self.owned_game_ids,
        )

        val_negatives = self._sample_negatives(
            n_samples=int(len(val_owned) * self.config.negative_sampling_ratio),
            excluded_ids=self.owned_game_ids | set(train_negatives["game_id"].to_list()),
        )

        test_negatives = self._sample_negatives(
            n_samples=int(len(test_owned) * self.config.negative_sampling_ratio),
            excluded_ids=self.owned_game_ids
            | set(train_negatives["game_id"].to_list())
            | set(val_negatives["game_id"].to_list()),
        )

        # Add ownership target column
        train_owned = train_owned.with_columns(pl.lit(1).alias("target"))
        val_owned = val_owned.with_columns(pl.lit(1).alias("target"))
        test_owned = test_owned.with_columns(pl.lit(1).alias("target"))

        train_negatives = train_negatives.with_columns(pl.lit(0).alias("target"))
        val_negatives = val_negatives.with_columns(pl.lit(0).alias("target"))
        test_negatives = test_negatives.with_columns(pl.lit(0).alias("target"))

        # Combine positives and negatives
        train_df = pl.concat([train_owned, train_negatives])
        val_df = pl.concat([val_owned, val_negatives])
        test_df = pl.concat([test_owned, test_negatives])

        # Shuffle
        train_df = train_df.sample(fraction=1.0, seed=self.config.random_seed)
        val_df = val_df.sample(fraction=1.0, seed=self.config.random_seed + 1)
        test_df = test_df.sample(fraction=1.0, seed=self.config.random_seed + 2)

        logger.info(f"Train: {len(train_df)} ({train_owned.height} pos, {train_negatives.height} neg)")
        logger.info(f"Val: {len(val_df)} ({val_owned.height} pos, {val_negatives.height} neg)")
        logger.info(f"Test: {len(test_df)} ({test_owned.height} pos, {test_negatives.height} neg)")

        if return_dict:
            return {"train": train_df, "validation": val_df, "test": test_df}
        return train_df, val_df, test_df

    def _get_owned_with_features(self) -> pl.DataFrame:
        """Get owned games with features from game universe.

        Returns features from game_universe_df for owned games.
        Collection-specific columns (user_rating, etc.) are NOT included
        since they're not used as model features - only for labeling.
        """
        owned_ids = list(self.owned_game_ids)

        # Get features from universe for owned games
        owned_features = self.game_universe_df.filter(
            pl.col("game_id").is_in(owned_ids)
        )

        return owned_features

    def _time_based_split_owned(
        self,
        owned_df: pl.DataFrame,
        train_end_year: int,
        time_column: str,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split owned games by time (e.g., year published).

        Args:
            owned_df: Owned games with features
            train_end_year: Year to end training data (exclusive)
            time_column: Column with year or datetime for splitting

        Returns:
            (train_df, val_df, test_df)
        """
        col_dtype = owned_df[time_column].dtype

        # Handle different column types
        if col_dtype == pl.Utf8:
            # String datetime - parse and extract year
            owned_df = owned_df.with_columns(
                pl.col(time_column).str.to_datetime().dt.year().alias("_split_year")
            )
        elif col_dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
            # Already a numeric year
            owned_df = owned_df.with_columns(
                pl.col(time_column).cast(pl.Int64).alias("_split_year")
            )
        elif col_dtype == pl.Datetime or str(col_dtype).startswith("Datetime"):
            # Datetime - extract year
            owned_df = owned_df.with_columns(
                pl.col(time_column).dt.year().alias("_split_year")
            )
        else:
            raise ValueError(f"Unsupported dtype for time column: {col_dtype}")

        # Handle nulls - put them in training
        owned_df = owned_df.with_columns(
            pl.col("_split_year").fill_null(train_end_year - 1)
        )

        # Calculate validation and test years
        val_end_year = train_end_year + 1
        test_end_year = val_end_year + 1

        train_df = owned_df.filter(pl.col("_split_year") < train_end_year)
        val_df = owned_df.filter(
            (pl.col("_split_year") >= train_end_year)
            & (pl.col("_split_year") < val_end_year)
        )
        test_df = owned_df.filter(pl.col("_split_year") >= val_end_year)

        # Drop helper column
        train_df = train_df.drop("_split_year")
        val_df = val_df.drop("_split_year")
        test_df = test_df.drop("_split_year")

        # If val or test is empty, fall back to random split
        if len(val_df) == 0 or len(test_df) == 0:
            logger.warning(
                "Time-based split produced empty val/test sets. "
                "Falling back to random split."
            )
            return self._random_split_owned(owned_df.drop("_split_year"))

        logger.info(
            f"Time-based split: train < {train_end_year}, "
            f"val {train_end_year}-{val_end_year}, test >= {val_end_year}"
        )

        return train_df, val_df, test_df

    def _random_split_owned(
        self, owned_df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Randomly split owned games into train/val/test.

        Args:
            owned_df: Owned games with features

        Returns:
            (train_df, val_df, test_df)
        """
        n = len(owned_df)
        n_val = int(n * self.config.validation_ratio)
        n_test = int(n * self.config.test_ratio)
        n_train = n - n_val - n_test

        # Shuffle and split
        shuffled = owned_df.sample(fraction=1.0, seed=self.config.random_seed)

        train_df = shuffled.head(n_train)
        val_df = shuffled.slice(n_train, n_val)
        test_df = shuffled.tail(n_test)

        logger.info(f"Random split: {n_train} train, {n_val} val, {n_test} test")

        return train_df, val_df, test_df

    def _sample_negatives(
        self,
        n_samples: int,
        excluded_ids: Optional[Set[int]] = None,
    ) -> pl.DataFrame:
        """Sample negative examples from non-owned games.

        Args:
            n_samples: Number of negative samples to draw
            excluded_ids: Game IDs to exclude from sampling

        Returns:
            DataFrame with sampled negative games
        """
        excluded_ids = excluded_ids or set()

        # Filter universe to eligible games
        eligible = self.game_universe_df.filter(
            ~pl.col("game_id").is_in(list(excluded_ids))
            & (pl.col("users_rated") >= self.config.min_ratings_for_negatives)
        )

        # Apply year filters if specified
        if self.config.min_year_for_negatives is not None:
            eligible = eligible.filter(
                pl.col("year_published") >= self.config.min_year_for_negatives
            )
        if self.config.max_year_for_negatives is not None:
            eligible = eligible.filter(
                pl.col("year_published") <= self.config.max_year_for_negatives
            )

        if len(eligible) == 0:
            raise ValueError("No eligible games for negative sampling after filtering")

        # Ensure we don't sample more than available
        n_samples = min(n_samples, len(eligible))

        # Apply sampling strategy
        if self.config.negative_sampling_strategy == "random":
            sampled = eligible.sample(n=n_samples, seed=self.config.random_seed)

        elif self.config.negative_sampling_strategy == "popularity_weighted":
            # Weight by inverse popularity (log scale to avoid extreme weights)
            # More obscure games are more likely to be sampled
            eligible = eligible.with_columns(
                (1.0 / (pl.col("users_rated").log() + 1)).alias("_weight")
            )
            # Normalize weights
            total_weight = eligible["_weight"].sum()
            eligible = eligible.with_columns(
                (pl.col("_weight") / total_weight).alias("_prob")
            )

            # Sample with weights
            probs = eligible["_prob"].to_numpy()
            indices = np.random.choice(
                len(eligible),
                size=n_samples,
                replace=False,
                p=probs,
            )
            sampled = eligible[indices].drop(["_weight", "_prob"])

        elif self.config.negative_sampling_strategy == "uniform":
            # Uniform sampling - all games equally likely
            sampled = eligible.sample(n=n_samples, seed=self.config.random_seed)

        else:
            raise ValueError(
                f"Unknown sampling strategy: {self.config.negative_sampling_strategy}"
            )

        logger.debug(f"Sampled {len(sampled)} negative examples")
        return sampled

    def get_split_stats(self) -> Dict[str, any]:
        """Get statistics about the collection for splitting.

        Returns:
            Dictionary with split statistics
        """
        owned_df = self._get_owned_with_features()

        stats = {
            "total_owned": len(self.owned_game_ids),
            "owned_with_features": len(owned_df),
            "total_universe": len(self.game_universe_df),
            "negative_sampling_ratio": self.config.negative_sampling_ratio,
            "sampling_strategy": self.config.negative_sampling_strategy,
        }

        # Year distribution of owned games
        if "year_published" in owned_df.columns:
            stats["owned_year_range"] = {
                "min": owned_df["year_published"].min(),
                "max": owned_df["year_published"].max(),
                "median": owned_df["year_published"].median(),
            }

        # Time-based split availability
        if "last_modified" in owned_df.columns:
            non_null = owned_df.filter(pl.col("last_modified").is_not_null())
            stats["has_last_modified"] = len(non_null)
            stats["last_modified_coverage"] = len(non_null) / len(owned_df)

        return stats
