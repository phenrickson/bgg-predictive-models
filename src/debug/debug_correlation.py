"""Debug script for correlation matrix computation."""

import logging
import time
from pathlib import Path

import pandas as pd
import polars as pl

from src.models.outcomes.train import get_model_class
from src.models.outcomes.data import select_X_y
from src.models.training import create_preprocessing_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Load hurdle training data
    data_path = Path("data/training/hurdle.parquet")
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info(f"Loading data from {data_path}")
    start = time.time()
    df = pl.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows in {time.time() - start:.2f}s")

    # Get model class for target column
    model_class = get_model_class("hurdle")
    model = model_class()

    # Extract X, y
    X, y = select_X_y(df, model.target_column)
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")

    # Create preprocessor with correlation filter enabled
    logger.info("Creating preprocessor...")
    preprocessor = create_preprocessing_pipeline(
        model_type="linear",
        preserve_columns=["year_published"],
        include_description_embeddings=False,
        remove_correlated=True,
        correlation_threshold=0.99,
    )

    # Fit and transform
    logger.info("Fitting preprocessor...")
    start = time.time()
    X_processed = preprocessor.fit_transform(X)
    logger.info(f"Preprocessor fit_transform in {time.time() - start:.2f}s")
    logger.info(f"Processed features shape: {X_processed.shape}")

    # X_processed is already a DataFrame (due to set_output(transform="pandas"))
    if isinstance(X_processed, pd.DataFrame):
        X_processed_df = X_processed
    else:
        X_processed_df = pd.DataFrame(X_processed)
    logger.info(f"DataFrame shape: {X_processed_df.shape}")

    # Sample 10k rows for correlation computation
    sample_size = 10000
    if len(X_processed_df) > sample_size:
        logger.info(f"Sampling {sample_size} rows for correlation matrix...")
        X_processed_df = X_processed_df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled DataFrame shape: {X_processed_df.shape}")

    # Check for NaNs
    nan_counts = X_processed_df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        logger.warning(f"Found {len(nan_cols)} columns with NaNs")
        # Sort by NaN count descending
        nan_cols_sorted = nan_cols.sort_values(ascending=False)
        logger.warning(f"Top 20 NaN columns:\n{nan_cols_sorted.head(20)}")
        logger.warning(f"Total NaN values: {nan_counts.sum()}")
        logger.warning(f"Rows with any NaN: {X_processed_df.isna().any(axis=1).sum()}")

    # Compute correlation matrix
    logger.info("Computing correlation matrix...")
    start = time.time()
    corr_matrix = X_processed_df.corr()
    logger.info(f"Correlation matrix computed in {time.time() - start:.2f}s")
    logger.info(f"Correlation matrix shape: {corr_matrix.shape}")

    # Save correlation matrix to CSV
    output_path = Path("data/debug/correlation_matrix.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(output_path)
    logger.info(f"Saved correlation matrix to {output_path}")

    # Find high correlations
    logger.info("Finding high correlation pairs (|r| >= 0.7)...")
    start = time.time()
    high_corr_pairs = []
    n_features = len(corr_matrix.columns)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= 0.7:
                high_corr_pairs.append({
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": corr_val,
                })
    logger.info(f"Found {len(high_corr_pairs)} high correlation pairs in {time.time() - start:.2f}s")

    # Save high correlations
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("correlation", key=abs, ascending=False)
        high_corr_path = Path("data/debug/high_correlations.csv")
        high_corr_df.to_csv(high_corr_path, index=False)
        logger.info(f"Saved high correlations to {high_corr_path}")
        logger.info(f"Top 20 highest correlations:\n{high_corr_df.head(20).to_string()}")

    logger.info("Debug complete!")


if __name__ == "__main__":
    main()
