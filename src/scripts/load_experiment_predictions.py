"""
Script to load test predictions from time-based evaluation experiments.

This script helps retrieve and load predictions from experiments that end in specific years,
typically produced by the time_based_evaluation module.
"""

import os
import logging
from typing import List, Optional, Union

import polars as pl
import pandas as pd


def load_experiment_predictions(
    base_dir: str = "./models/experiments",
    model_types: Optional[Union[str, List[str]]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    return_format: str = "polars",
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Load test_predictions.parquet from time-based evaluation experiments.

    Args:
        base_dir (str): Base directory containing experiment subdirectories.
        model_types (str or List[str], optional): Filter for specific model types.
        start_year (int, optional): Minimum year for experiment splits.
        end_year (int, optional): Maximum year for experiment splits.
        return_format (str, optional): Format of returned dataframe.

    Returns:
        DataFrame containing loaded predictions.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Normalize model_types to list
    if isinstance(model_types, str):
        model_types = [model_types]

    # Find experiment subdirectories
    predictions_list = []
    for model_type in model_types or []:
        model_dir = os.path.join(base_dir, model_type)

        # Check if model directory exists
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory not found: {model_dir}")
            continue

        # Find experiment subdirectories
        for exp_dir in os.listdir(model_dir):
            full_exp_path = os.path.join(model_dir, exp_dir)

            # Check if this is an experiment directory
            if not os.path.isdir(full_exp_path):
                continue

            # Extract years from experiment directory name
            try:
                train_end_year, test_start_year = map(int, exp_dir.split("_")[-2:])

                # Apply year filters
                if (start_year is not None and test_start_year < start_year) or (
                    end_year is not None and test_start_year > end_year
                ):
                    continue
            except (ValueError, IndexError):
                continue

            # Find the latest version directory
            version_dirs = [
                d
                for d in os.listdir(full_exp_path)
                if d.startswith("v") and d[1:].isdigit()
            ]

            if not version_dirs:
                logger.warning(f"No version directories found in {full_exp_path}")
                continue

            # Sort versions and get the latest
            latest_version_dir = sorted(
                version_dirs, key=lambda x: int(x[1:]), reverse=True
            )[0]

            full_version_path = os.path.join(full_exp_path, latest_version_dir)

            # Look for test_predictions.parquet
            test_pred_path = os.path.join(full_version_path, "test_predictions.parquet")

            if not os.path.exists(test_pred_path):
                logger.warning(
                    f"No test_predictions.parquet found in {full_version_path}"
                )
                continue

            try:
                # Log detailed file information
                logger.info(f"Loading predictions from: {test_pred_path}")
                logger.info(f"  Model Type: {model_type}")
                logger.info(f"  Experiment: {exp_dir}")
                logger.info(f"  Version: {latest_version_dir}")

                df = pl.read_parquet(test_pred_path)

                logger.info(f"  Number of rows: {len(df)}")
                logger.info(f"  Columns: {', '.join(df.columns)}")

                # Select columns based on model type
                base_columns = ["game_id", "name", "year_published"]

                # Standardize column names
                if model_type in ["complexity", "rating", "users_rated"]:
                    # Regression models
                    specific_columns = ["prediction", "actual"]
                    # Add None for predicted_proba_class_1
                    df = df.with_columns(
                        [pl.lit(None).alias("predicted_proba_class_1")]
                    )
                elif model_type == "hurdle":
                    # Classification model
                    specific_columns = [
                        "prediction",
                        "actual",
                        "predicted_proba_class_1",
                    ]
                else:
                    specific_columns = []

                # Combine columns
                selected_columns = base_columns + specific_columns

                # Filter DataFrame to selected columns
                df = df.select(selected_columns)

                # Add metadata columns
                df = df.with_columns(
                    [
                        pl.lit(exp_dir).alias("experiment_name"),
                        pl.lit(model_type).alias("model_type"),
                    ]
                )

                predictions_list.append(df)
            except Exception as e:
                logger.warning(f"Could not load {test_pred_path}: {e}")

    # Combine predictions
    if not predictions_list:
        logger.warning("No predictions found matching the criteria")
        return pl.DataFrame() if return_format == "polars" else pd.DataFrame()

    # Ensure consistent columns
    standard_columns = [
        "game_id",
        "name",
        "year_published",
        "prediction",
        "actual",
        "predicted_proba_class_1",
        "experiment_name",
        "model_type",
    ]

    # Standardize DataFrames
    standardized_predictions = []
    for df in predictions_list:
        # Create a new DataFrame with standard columns and consistent types
        std_df_dict = {}
        for col in standard_columns:
            if col in df.columns:
                # Use the column from the original DataFrame
                col_data = df.get_column(col)
            else:
                # Create a column of None with appropriate type
                col_data = [None] * len(df)

            # Standardize types
            if col in ["game_id", "year_published"]:
                std_df_dict[col] = pl.Series(col, col_data, dtype=pl.Int64)
            elif col in ["prediction", "actual", "predicted_proba_class_1"]:
                std_df_dict[col] = pl.Series(col, col_data, dtype=pl.Float64)
            elif col in ["experiment_name", "model_type"]:
                std_df_dict[col] = pl.Series(col, col_data, dtype=pl.String)
            else:
                std_df_dict[col] = pl.Series(col, col_data)

        std_df = pl.DataFrame(std_df_dict)
        standardized_predictions.append(std_df)

    # Concatenate standardized DataFrames
    combined_predictions = pl.concat(standardized_predictions)

    # Log summary of combined predictions
    logger.info(f"Total number of predictions loaded: {len(combined_predictions)}")
    logger.info(f"Unique model types: {combined_predictions['model_type'].unique()}")
    logger.info(
        f"Unique experiment names: {combined_predictions['experiment_name'].unique()}"
    )

    # Convert to pandas if requested
    if return_format == "pandas":
        combined_predictions = combined_predictions.to_pandas()

    return combined_predictions


def main():
    """
    CLI entry point for loading experiment predictions.
    Demonstrates usage of the load_experiment_predictions function.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Load Experiment Predictions")
    parser.add_argument(
        "--predictions-dir",
        default="./models/experiments",
        help="Base directory containing model experiment subdirectories",
    )
    parser.add_argument(
        "--model-types", nargs="+", help="Filter for specific model types"
    )
    parser.add_argument(
        "--start-year", type=int, help="Minimum year for experiment splits"
    )
    parser.add_argument(
        "--end-year", type=int, help="Maximum year for experiment splits"
    )
    parser.add_argument(
        "--output",
        default="experiment_predictions.parquet",
        help="Output file for combined predictions",
    )

    args = parser.parse_args()

    # Load predictions
    predictions = load_experiment_predictions(
        base_dir=args.predictions_dir,
        model_types=args.model_types,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    # Print column names and details
    if not predictions.is_empty():
        print("Columns in the predictions:")
        print(predictions.columns)
        print("\nColumn types:")
        print(predictions.dtypes)
        print(f"\nTotal number of predictions: {len(predictions)}")

        # Save predictions
        predictions.write_parquet(args.output)
        print(f"Saved {len(predictions)} predictions to {args.output}")
    else:
        print("No predictions found.")


if __name__ == "__main__":
    main()
