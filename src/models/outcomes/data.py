"""Centralized data loading for outcome models."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from src.models.outcomes.base import DataConfig
from src.utils.config import load_config
from src.data.loader import BGGDataLoader
from src.models.splitting import time_based_split


logger = logging.getLogger(__name__)


def load_training_data(
    data_config: DataConfig,
    end_year: int,
    use_embeddings: bool = False,
    complexity_predictions_path: Optional[Union[str, Path]] = None,
    local_data_path: Optional[Union[str, Path]] = None,
) -> pl.DataFrame:
    """Load training data based on model's data configuration.

    This is the centralized data loader that handles:
    1. Loading base features (from BigQuery or local parquet)
    2. Applying filters (min_ratings, min_weights)
    3. Joining complexity predictions (if required)
    4. Joining embeddings (if enabled and supported)

    Args:
        data_config: Model's data requirements specification.
        end_year: Maximum year to include in the data.
        use_embeddings: Whether to load and join embeddings.
        complexity_predictions_path: Path to complexity predictions parquet.
            Required if data_config.requires_complexity_predictions is True.
        local_data_path: Optional path to local parquet file instead of BigQuery.

    Returns:
        Polars DataFrame with all required features joined.

    Raises:
        ValueError: If complexity predictions required but not provided.
    """
    # Load base features
    df = _load_base_features(
        local_data_path=local_data_path,
        end_year=end_year,
        min_ratings=data_config.min_ratings,
        min_weights=data_config.min_weights,
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

    # Join embeddings if enabled and supported
    if use_embeddings and data_config.supports_embeddings:
        df = _join_embeddings(df)
        logger.info(f"After joining embeddings: {len(df)} rows")
    elif use_embeddings and not data_config.supports_embeddings:
        logger.warning(
            f"Embeddings requested but not supported by this model. Skipping."
        )

    return df


def _load_base_features(
    local_data_path: Optional[Union[str, Path]],
    end_year: int,
    min_ratings: Optional[int] = None,
    min_weights: Optional[int] = None,
) -> pl.DataFrame:
    """Load base game features from BigQuery or local file.

    Args:
        local_data_path: Path to local parquet file, or None for BigQuery.
        end_year: Maximum year to include.
        min_ratings: Minimum user ratings filter.
        min_weights: Minimum complexity weights filter.

    Returns:
        Polars DataFrame with base features.
    """
    if local_data_path is not None:
        logger.info(f"Loading data from local file: {local_data_path}")
        df = pl.read_parquet(local_data_path)

        # Apply filters
        df = df.filter(pl.col("year_published").is_not_null())
        df = df.filter(pl.col("year_published") <= end_year)

        if min_ratings is not None:
            df = df.filter(pl.col("users_rated") >= min_ratings)

        if min_weights is not None:
            df = df.filter(pl.col("num_weights") >= min_weights)

        return df

    # Load from BigQuery
    config = load_config()
    loader = BGGDataLoader(config.get_bigquery_config())

    # Build where clause
    where_clauses = [f"year_published <= {end_year}"]

    if min_ratings is not None:
        where_clauses.append(f"users_rated >= {min_ratings}")

    if min_weights is not None:
        where_clauses.append(f"num_weights >= {min_weights}")

    where_clause = " AND ".join(where_clauses)

    logger.info(f"Loading data from BigQuery with filter: {where_clause}")
    return loader.load_data(where_clause=where_clause)


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


def _join_embeddings(df: pl.DataFrame) -> pl.DataFrame:
    """Join text embeddings from BigQuery.

    Loads embeddings from bgg-data-warehouse.predictions.bgg_description_embeddings
    and expands the embedding array into individual columns.

    Args:
        df: Base dataframe with game features.

    Returns:
        DataFrame with embedding columns joined (emb_0, emb_1, ..., emb_N).

    Raises:
        ValueError: If no games remain after join.
    """
    config = load_config()
    client = config.get_bigquery_config().get_client()

    # Query embeddings from data warehouse
    # The embedding column is an ARRAY<FLOAT64>
    query = """
    SELECT
        game_id,
        embedding
    FROM `bgg-data-warehouse.predictions.bgg_description_embeddings`
    """

    logger.info("Loading embeddings from BigQuery")
    query_job = client.query(query)
    query_job.result()
    embeddings_pandas = query_job.to_dataframe()

    if len(embeddings_pandas) == 0:
        logger.warning("No embeddings found in BigQuery. Skipping embedding join.")
        return df

    # Expand embedding array into columns
    embeddings_expanded = _expand_embedding_array(embeddings_pandas)
    embeddings_df = pl.from_pandas(embeddings_expanded)

    logger.info(
        f"Loaded {len(embeddings_df)} embeddings with "
        f"{len(embeddings_df.columns) - 1} dimensions"
    )

    # Join on game_id
    df = df.join(embeddings_df, on="game_id", how="inner")

    if len(df) == 0:
        raise ValueError("No games remain after joining embeddings")

    return df


def _expand_embedding_array(df: pd.DataFrame) -> pd.DataFrame:
    """Expand embedding array column into individual columns.

    Args:
        df: DataFrame with 'game_id' and 'embedding' columns,
            where embedding is a list/array of floats.

    Returns:
        DataFrame with game_id and emb_0, emb_1, ..., emb_N columns.
    """
    # Get embedding dimension from first non-null embedding
    sample_embedding = None
    for emb in df["embedding"]:
        if emb is not None and len(emb) > 0:
            sample_embedding = emb
            break

    if sample_embedding is None:
        raise ValueError("No valid embeddings found in data")

    embedding_dim = len(sample_embedding)
    logger.info(f"Embedding dimension: {embedding_dim}")

    # Create column names
    emb_columns = [f"emb_{i}" for i in range(embedding_dim)]

    # Expand embeddings into columns
    embeddings_matrix = np.vstack(df["embedding"].values)
    embeddings_df = pd.DataFrame(embeddings_matrix, columns=emb_columns)
    embeddings_df["game_id"] = df["game_id"].values

    # Reorder columns to have game_id first
    return embeddings_df[["game_id"] + emb_columns]


def create_data_splits(
    df: pl.DataFrame,
    train_end_year: int,
    tune_start_year: int,
    tune_end_year: int,
    test_start_year: int,
    test_end_year: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create time-based data splits for model training.

    Args:
        df: Input dataframe.
        train_end_year: Exclusive end year for training data.
        tune_start_year: Inclusive start year for tuning data.
        tune_end_year: Inclusive end year for tuning data.
        test_start_year: Inclusive start year for test data.
        test_end_year: Inclusive end year for test data.

    Returns:
        Tuple of (train_df, tune_df, test_df).

    Raises:
        ValueError: If any split is empty.
    """
    validation_window = tune_end_year - tune_start_year + 1
    test_window = test_end_year - test_start_year + 1

    train_df, tune_df, test_df = time_based_split(
        df=df,
        train_end_year=train_end_year,
        prediction_window=validation_window,
        test_window=test_window,
        time_col="year_published",
        return_dict=False,
    )

    if len(train_df) == 0:
        raise ValueError(f"Training set is empty. Check train_end_year={train_end_year}")
    if len(tune_df) == 0:
        raise ValueError(
            f"Tuning set is empty. Check tune years: {tune_start_year}-{tune_end_year}"
        )
    if len(test_df) == 0:
        raise ValueError(
            f"Test set is empty. Check test years: {test_start_year}-{test_end_year}"
        )

    logger.info(f"Training data: {len(train_df)} rows (years < {train_end_year})")
    logger.info(
        f"Tuning data: {len(tune_df)} rows (years {tune_start_year}-{tune_end_year})"
    )
    logger.info(
        f"Test data: {len(test_df)} rows (years {test_start_year}-{test_end_year})"
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
