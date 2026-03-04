"""Centralized data loading for outcome models."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import polars as pl

from src.models.outcomes.base import DataConfig
from src.utils.config import load_config
from src.data.loader import BGGDataLoader
from src.models.splitting import time_based_split


logger = logging.getLogger(__name__)


def load_data(
    data_config: DataConfig,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    use_embeddings: bool = False,
    complexity_predictions_path: Optional[Union[str, Path]] = None,
    local_data_path: Optional[Union[str, Path]] = None,
    apply_filters: bool = True,
) -> pl.DataFrame:
    """Unified data loader for training and scoring.

    This is the centralized data loader that handles:
    1. Loading base features (from BigQuery or local parquet)
    2. Applying year filters (start_year, end_year)
    3. Applying data filters (min_ratings, min_weights from data_config)
    4. Joining complexity predictions (if required by data_config)
    5. Joining embeddings (if enabled and supported by data_config)

    Args:
        data_config: Model's data requirements specification.
        start_year: Minimum year to include (inclusive). None for no lower bound.
        end_year: Maximum year to include (inclusive). None for no upper bound.
        use_embeddings: Whether to load and join embeddings.
        complexity_predictions_path: Path to complexity predictions parquet.
            Required if data_config.requires_complexity_predictions is True.
        local_data_path: Optional path to local parquet file instead of BigQuery.
        apply_filters: Whether to apply min_ratings/min_weights filters from data_config.
            Set to False for scoring to predict on all games.

    Returns:
        Polars DataFrame with all required features joined.

    Raises:
        ValueError: If complexity predictions required but not provided.
    """
    # Determine if we should load embeddings
    load_embeddings = use_embeddings and data_config.supports_embeddings
    if use_embeddings and not data_config.supports_embeddings:
        logger.warning(
            "Embeddings requested but not supported by this model. Skipping."
        )

    # Load base features (with embeddings if requested)
    # Only apply min_ratings/min_weights filters if apply_filters is True
    df = _load_base_features(
        local_data_path=local_data_path,
        start_year=start_year,
        end_year=end_year,
        min_ratings=data_config.min_ratings if apply_filters else None,
        min_weights=data_config.min_weights if apply_filters else None,
        use_embeddings=load_embeddings,
    )
    logger.info(f"Loaded base features: {len(df)} rows")

    # Join complexity predictions if required
    if data_config.requires_complexity_predictions:
        if complexity_predictions_path is None:
            raise ValueError(
                "complexity_predictions_path required for this model but not provided"
            )
        df = _join_complexity_predictions(df, complexity_predictions_path)
        logger.info(f"After joining complexity predictions: {len(df)} rows")

    return df


def load_training_data(
    data_config: DataConfig,
    end_year: int,
    use_embeddings: bool = False,
    complexity_predictions_path: Optional[Union[str, Path]] = None,
    local_data_path: Optional[Union[str, Path]] = None,
) -> pl.DataFrame:
    """Load training data (convenience wrapper around load_data).

    Loads data up to and including end_year.

    Args:
        data_config: Model's data requirements specification.
        end_year: Maximum year to include in the data.
        use_embeddings: Whether to load and join embeddings.
        complexity_predictions_path: Path to complexity predictions parquet.
        local_data_path: Optional path to local parquet file instead of BigQuery.

    Returns:
        Polars DataFrame with all required features joined.
    """
    return load_data(
        data_config=data_config,
        start_year=None,
        end_year=end_year,
        use_embeddings=use_embeddings,
        complexity_predictions_path=complexity_predictions_path,
        local_data_path=local_data_path,
    )


def load_scoring_data(
    data_config: DataConfig,
    start_year: int,
    end_year: Optional[int] = None,
    use_embeddings: bool = False,
    complexity_predictions_path: Optional[Union[str, Path]] = None,
    local_data_path: Optional[Union[str, Path]] = None,
) -> pl.DataFrame:
    """Load scoring data (convenience wrapper around load_data).

    Loads data from start_year onwards (typically after the model's training cutoff).
    Does NOT apply min_ratings/min_weights filters - scoring should predict on all games.

    Args:
        data_config: Model's data requirements specification.
        start_year: Minimum year to include (inclusive).
        end_year: Maximum year to include (inclusive). None for no upper bound.
        use_embeddings: Whether to load and join embeddings.
        complexity_predictions_path: Path to complexity predictions parquet.
        local_data_path: Optional path to local parquet file instead of BigQuery.

    Returns:
        Polars DataFrame with all required features joined.
    """
    return load_data(
        data_config=data_config,
        start_year=start_year,
        end_year=end_year,
        use_embeddings=use_embeddings,
        complexity_predictions_path=complexity_predictions_path,
        local_data_path=local_data_path,
        apply_filters=False,  # Don't filter during scoring - predict on all games
    )


def _load_base_features(
    local_data_path: Optional[Union[str, Path]],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    min_ratings: Optional[int] = None,
    min_weights: Optional[int] = None,
    use_embeddings: bool = False,
) -> pl.DataFrame:
    """Load base game features from BigQuery or local file.

    Args:
        local_data_path: Path to local parquet file, or None for BigQuery.
        start_year: Minimum year to include (inclusive). None for no lower bound.
        end_year: Maximum year to include (inclusive). None for no upper bound.
        min_ratings: Minimum user ratings filter.
        min_weights: Minimum complexity weights filter.
        use_embeddings: Whether to join embeddings in the same query.

    Returns:
        Polars DataFrame with base features (and embeddings if requested).
    """
    if local_data_path is not None:
        logger.info(f"Loading data from local file: {local_data_path}")
        df = pl.read_parquet(local_data_path)

        # Apply filters
        df = df.filter(pl.col("year_published").is_not_null())

        if start_year is not None:
            df = df.filter(pl.col("year_published") >= start_year)

        if end_year is not None:
            df = df.filter(pl.col("year_published") <= end_year)

        if min_ratings is not None:
            df = df.filter(pl.col("users_rated") >= min_ratings)

        if min_weights is not None:
            df = df.filter(pl.col("num_weights") >= min_weights)

        # For local data, embeddings would need to be joined separately
        if use_embeddings:
            logger.warning(
                "Embeddings not supported with local data. "
                "Use BigQuery or pre-join embeddings in local file."
            )

        return df

    # Load from BigQuery
    config = load_config()

    # Build where clause
    where_clauses = []

    if start_year is not None:
        where_clauses.append(f"f.year_published >= {start_year}")

    if end_year is not None:
        where_clauses.append(f"f.year_published <= {end_year}")

    if min_ratings is not None:
        where_clauses.append(f"f.users_rated >= {min_ratings}")

    if min_weights is not None:
        where_clauses.append(f"f.num_weights >= {min_weights}")

    where_clause = " AND ".join(where_clauses) if where_clauses else "TRUE"

    loader = BGGDataLoader(config.get_data_warehouse_config())

    if use_embeddings:
        logger.info(f"Loading features with embeddings from BigQuery: {where_clause}")
        return loader.load_data_with_embeddings(where_clause=where_clause if where_clause != "TRUE" else "")
    else:
        # Remove table alias prefix for simple query
        simple_where = where_clause.replace("f.", "")
        logger.info(f"Loading data from BigQuery with filter: {simple_where}")
        return loader.load_data(where_clause=simple_where if simple_where != "TRUE" else "")



def _join_complexity_predictions(
    df: pl.DataFrame,
    predictions_path: Union[str, Path],
) -> pl.DataFrame:
    """Join complexity predictions to the base dataframe.

    Args:
        df: Base dataframe with game features.
        predictions_path: Path to complexity predictions parquet file.

    Returns:
        DataFrame with predicted_complexity column joined.

    Raises:
        ValueError: If no games remain after join.
    """
    logger.info(f"Loading complexity predictions from {predictions_path}")
    complexity_df = pl.read_parquet(predictions_path)

    if len(complexity_df) == 0:
        raise ValueError(f"No complexity predictions found in {predictions_path}")

    # Join on game_id
    df = df.join(complexity_df, on="game_id", how="inner")

    if len(df) == 0:
        raise ValueError("No games remain after joining complexity predictions")

    return df


def create_data_splits(
    df: pl.DataFrame,
    train_through: int,
    tune_start: int,
    tune_through: int,
    test_start: int,
    test_through: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create time-based data splits for model training.

    Args:
        df: Input dataframe.
        train_through: Inclusive end year for training data.
        tune_start: Inclusive start year for tuning data.
        tune_through: Inclusive end year for tuning data.
        test_start: Inclusive start year for test data.
        test_through: Inclusive end year for test data.

    Returns:
        Tuple of (train_df, tune_df, test_df).

    Raises:
        ValueError: If any split is empty.
    """
    validation_window = tune_through - tune_start + 1
    test_window = test_through - test_start + 1

    train_df, tune_df, test_df = time_based_split(
        df=df,
        train_through=train_through,
        prediction_window=validation_window,
        test_window=test_window,
        time_col="year_published",
        return_dict=False,
    )

    if len(train_df) == 0:
        raise ValueError(f"Training set is empty. Check train_through={train_through}")
    if len(tune_df) == 0:
        raise ValueError(
            f"Tuning set is empty. Check tune years: {tune_start}-{tune_through}"
        )
    if len(test_df) == 0:
        raise ValueError(
            f"Test set is empty. Check test years: {test_start}-{test_through}"
        )

    logger.info(f"Training data: {len(train_df)} rows (years through {train_through})")
    logger.info(
        f"Tuning data: {len(tune_df)} rows (years {tune_start}-{tune_through})"
    )
    logger.info(
        f"Test data: {len(test_df)} rows (years {test_start}-{test_through})"
    )

    return train_df, tune_df, test_df


def select_X_y(
    df: pl.DataFrame,
    target_column: str,
    to_pandas: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features (X) and target (y) from a dataframe.

    Args:
        df: Polars DataFrame.
        target_column: Name of the target column.
        to_pandas: Whether to convert to pandas (default True).

    Returns:
        Tuple of (X, y) as pandas DataFrame/Series.

    Raises:
        ValueError: If target column not found.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {df.columns}"
        )

    X = df.drop(target_column)
    y = df.select(target_column)

    if to_pandas:
        X = X.to_pandas()
        y = y.to_pandas().squeeze()

    return X, y
