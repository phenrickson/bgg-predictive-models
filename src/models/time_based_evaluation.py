"""Time-based Model Evaluation and Prediction Pipeline"""

import logging
import argparse
from typing import Dict, Optional, List, Any

import polars as pl
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from src.models.training import load_data, create_data_splits
from src.models.finalize_model import finalize_model

import subprocess
import sys
import os
import json


def generate_time_splits(
    start_year: int = 2015,
    end_year: int = 2025,
    prediction_window: int = 2,
    train_window: Optional[int] = None,
) -> List[Dict[str, int]]:
    """
    Generate time-based splits for model evaluation.

    Args:
        start_year: First year to start generating splits
        end_year: Last year to generate splits
        prediction_window: Number of years to predict
        train_window: Optional fixed training window. If None, uses all previous data.

    Returns:
        List of dictionaries with split configurations
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating time splits from {start_year} to {end_year}")

    splits = []
    for train_through in range(start_year, end_year):
        split_config = {
            "train_through": train_through,
            "tune_start": train_through,
            "tune_through": train_through + 1,
            "test_start": train_through + prediction_window,
            "test_through": train_through + prediction_window + 1,
        }
        splits.append(split_config)

    logger.info(f"Generated {len(splits)} time splits")
    return splits


def evaluate_geek_rating_performance(
    output_dir: str,
    experiment_base: str,
    predicted_ratings_path: str,
    test_df: pd.DataFrame,
    df: pd.DataFrame,
):
    """
    Evaluate geek rating performance by comparing predicted and actual ratings.

    Args:
        output_dir: Base directory for storing experiments
        experiment_base: Base name for the experiment
        predicted_ratings_path: Path to the predicted ratings file
        test_df: Test dataset
        df: Full original dataset
    """
    logger = logging.getLogger(__name__)

    # Load predicted geek ratings
    predicted_ratings = pl.read_parquet(predicted_ratings_path).to_pandas()

    # Calculate actual geek ratings by joining with the original dataset
    test_df_with_geek_rating = test_df.merge(
        df[["game_id", "geek_rating"]], on="game_id", how="left"
    )
    test_df_with_geek_rating.rename(
        columns={"geek_rating": "actual_geek_rating"}, inplace=True
    )

    # Merge predicted and actual ratings
    merged_ratings = predicted_ratings.merge(
        test_df_with_geek_rating[["game_id", "actual_geek_rating"]],
        on="game_id",
        how="inner",
    )

    # Calculate performance metrics
    mae = metrics.mean_absolute_error(
        merged_ratings["actual_geek_rating"],
        merged_ratings["predicted_geek_rating"],
    )
    rmse = np.sqrt(
        metrics.mean_squared_error(
            merged_ratings["actual_geek_rating"],
            merged_ratings["predicted_geek_rating"],
        )
    )
    r2 = metrics.r2_score(
        merged_ratings["actual_geek_rating"],
        merged_ratings["predicted_geek_rating"],
    )

    # Log performance metrics
    logger.info(f"Geek Rating Performance Metrics for Split {experiment_base}:")
    logger.info(f"  Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    logger.info(f"  R-squared (R²): {r2:.4f}")

    # Optional: Save performance metrics to a file
    performance_metrics = {
        "split": experiment_base,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    performance_path = os.path.join(
        output_dir,
        "metrics",
        f"geek_rating_metrics_{experiment_base}.json",
    )
    os.makedirs(os.path.dirname(performance_path), exist_ok=True)

    with open(performance_path, "w") as f:
        json.dump(performance_metrics, f, indent=2)


def run_time_based_evaluation(
    splits: Optional[List[Dict[str, int]]] = None,
    min_ratings: int = 0,
    output_dir: str = "./models/experiments",
    local_data_path: Optional[str] = None,
    model_args: Optional[Dict[str, Dict[str, Any]]] = None,
    additional_args: Optional[List[str]] = None,
):
    """
    Run comprehensive time-based model evaluation pipeline.

    Args:
        splits: Optional list of time splits. If None, generates default splits.
        min_ratings: Minimum number of ratings threshold
        output_dir: Base directory for storing experiments
        local_data_path: Optional path to local data file
        model_args: Optional dictionary of additional arguments for each model
        additional_args: Optional list of additional CLI arguments to pass to all model scripts
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file_path = os.path.join(output_dir, "time_based_evaluation.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                log_file_path, mode="w"
            ),  # 'w' mode to overwrite previous log
        ],
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Generate splits if not provided
    if splits is None:
        splits = generate_time_splits()

    # Prepare additional arguments
    global_additional_args = additional_args or []

    # Load full dataset
    df = load_data(
        local_data_path=local_data_path,
        min_ratings=min_ratings,
        end_train_year=max(split["test_through"] for split in splits),
    )

    # Iterate through splits
    for split_config in splits:
        logger.info(f"Processing split: {split_config}")

        # Create data splits
        train_df, tune_df, test_df = create_data_splits(
            df,
            train_through=split_config["train_through"],
            tune_start=split_config["tune_start"],
            tune_through=split_config["tune_through"],
            test_start=split_config["test_start"],
            test_through=split_config["test_through"],
        )

        # Experiment naming
        experiment_base = (
            f"{split_config['train_through']}_{split_config['test_start']}"
        )

        # Run individual model training scripts
        model_scripts = [
            ("hurdle", "src/models/hurdle.py"),
            ("complexity", "src/models/complexity.py"),
            ("rating", "src/models/rating.py"),
            ("users_rated", "src/models/users_rated.py"),
        ]

        complexity_local_path = None
        for model_name, script_path in model_scripts:
            # Convert script path to module path
            module_path = script_path.replace("/", ".").replace(".py", "")

            cmd = [
                sys.executable,
                "-m",
                module_path,
                f"--train-through={split_config['train_through']}",
                f"--tune-start={split_config['tune_start']}",
                f"--tune-through={split_config['tune_through']}",
                f"--test-start={split_config['test_start']}",
                f"--test-through={split_config['test_through']}",
                f"--experiment={model_name}_{experiment_base}",
                f"--output-dir={output_dir}",
            ]

            # Add model-specific arguments if provided
            if model_args and model_name in model_args:
                for key, value in model_args[model_name].items():
                    # Special handling for boolean flags
                    if isinstance(value, bool):
                        if value:
                            cmd.append(f"--{key}")
                    else:
                        cmd.append(f"--{key}={value}")

            # Add complexity experiment reference for rating/users_rated
            if model_name in ["rating", "users_rated"]:
                # Ensure complexity local path exists
                if complexity_local_path is None:
                    logger.warning("No complexity predictions found for this split")
                    continue

                cmd.extend(
                    [
                        f"--complexity-experiment=complexity_{experiment_base}",
                        f"--local-complexity-path={complexity_local_path}",
                    ]
                )

            # Add any global additional arguments
            cmd.extend(global_additional_args)

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.info(f"{model_name.capitalize()} Model Training Completed")
                logger.debug(result.stdout)

                # If this is the complexity model, save its predictions
                if model_name == "complexity":
                    complexity_local_path = os.path.join(
                        output_dir,
                        "predictions",
                        f"complexity_{experiment_base}.parquet",
                    )
                    logger.info(
                        f"Complexity predictions saved to {complexity_local_path}"
                    )

                # Finalize the trained model
                try:
                    finalize_model(
                        model_type=model_name,
                        experiment_name=f"{model_name}_{experiment_base}",
                        end_year=split_config["train_through"],
                    )
                    logger.info(f"{model_name.capitalize()} Model Finalized")
                except Exception as finalize_error:
                    logger.error(
                        f"Error finalizing {model_name} model: {finalize_error}"
                    )
                    raise

            except subprocess.CalledProcessError as e:
                logger.error(f"Error training {model_name} model: {e}")
                logger.error(f"STDOUT: {e.stdout}")
                logger.error(f"STDERR: {e.stderr}")
                raise

        # Geek Rating Calculation
        try:
            geek_rating_cmd = [
                sys.executable,
                "-m",
                "src.models.geek_rating",
                f"--hurdle=hurdle_{experiment_base}",
                f"--complexity=complexity_{experiment_base}",
                f"--rating=rating_{experiment_base}",
                f"--users-rated=users_rated_{experiment_base}",
                f"--experiment=geek_rating_{experiment_base}",
                f"--output={os.path.join('geek_rating', f'geek_rating_{experiment_base}', 'v1', 'test_predictions.parquet')}",
                f"--start-year={split_config['test_start']}",
                f"--end-year={split_config['test_through']}",
                f"--local-complexity-path={complexity_local_path}",
            ]

            # Add any global additional arguments
            geek_rating_cmd.extend(global_additional_args)

            result = subprocess.run(
                geek_rating_cmd, capture_output=True, text=True, check=True
            )
            logger.info("Geek Rating Calculation Completed")
            logger.debug(result.stdout)

            # # Performance Evaluation
            # evaluate_geek_rating_performance(
            #     output_dir=output_dir,
            #     experiment_base=experiment_base,
            #     predicted_ratings_path=os.path.join(
            #         output_dir,
            #         "geek_rating",
            #         f"geek_rating_{experiment_base}",
            #         "v1",
            #         "test_predictions.parquet",
            #     ),
            #     test_df=test_df,
            #     df=df,
            # )

        except subprocess.CalledProcessError as e:
            logger.error(f"Error calculating geek ratings: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise


def main():
    """CLI entry point for time-based evaluation."""
    parser = argparse.ArgumentParser(description="Run Time-Based Model Evaluation")
    parser.add_argument(
        "--start-year", type=int, default=2015, help="First year for evaluation"
    )
    parser.add_argument(
        "--end-year", type=int, default=2025, help="Last year for evaluation"
    )
    parser.add_argument(
        "--min-ratings", type=int, default=0, help="Minimum ratings threshold"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/experiments",
        help="Output directory for experiments",
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Optional path to local data file",
    )
    parser.add_argument(
        "--model-args",
        nargs="+",
        help="Model-specific arguments in format 'model.key=value'",
    )
    parser.add_argument(
        "--additional-args",
        nargs="+",
        help="Additional arguments to pass to all model scripts",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Parse model-specific arguments
    model_args = {}
    if args.model_args:
        for arg in args.model_args:
            try:
                # Split into model.key and value
                full_key, value = arg.split("=")
                model, key = full_key.split(".")

                # Initialize model dict if not exists
                if model not in model_args:
                    model_args[model] = {}

                # Convert value to appropriate type
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)

                # Add key-value pair
                model_args[model][key] = value
            except ValueError:
                print(f"Warning: Skipping invalid model argument: {arg}")

    # Generate splits
    splits = generate_time_splits(start_year=args.start_year, end_year=args.end_year)

    # Run evaluation
    run_time_based_evaluation(
        splits=splits,
        min_ratings=args.min_ratings,
        output_dir=args.output_dir,
        local_data_path=args.local_data,
        model_args=model_args,
        additional_args=args.additional_args,
    )


if __name__ == "__main__":
    main()
