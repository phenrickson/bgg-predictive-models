"""Estimate complexity for board games using a trained complexity model."""

import argparse
import numpy as np
import pandas as pd
import polars as pl
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from src.models.score import load_model, load_scoring_data, predict_data
from src.models.experiments import ExperimentTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_complexity_results(
    df: pl.DataFrame, predicted_complexity: np.ndarray, experiment_name: str
) -> pl.DataFrame:
    """
    Prepare results DataFrame for complexity predictions.

    Args:
        df: Original input DataFrame
        predicted_complexity: Predicted complexity values
        experiment_name: Name of the experiment used for predictions

    Returns:
        Results DataFrame with predictions and metadata
    """
    # Validate input data
    logger.info(f"Input DataFrame shape: {df.shape}")
    logger.info(f"Predicted complexity shape: {predicted_complexity.shape}")
    logger.info(f"Predicted complexity length: {len(predicted_complexity)}")

    if len(predicted_complexity) != len(df):
        raise ValueError(
            f"Mismatch in data lengths: DataFrame has {len(df)} rows, "
            f"but predictions have {len(predicted_complexity)} rows"
        )

    # Current timestamp for load_ts
    load_ts = datetime.now().isoformat()  # Convert to ISO format string

    # Get experiment details for score_ts
    tracker = ExperimentTracker("complexity")
    experiments = tracker.list_experiments()

    # Debug logging
    logger.info(f"All experiments: {experiments}")
    logger.info(f"Searching for experiment name: {experiment_name}")

    matching_experiments = [
        exp
        for exp in experiments
        if exp["name"] == experiment_name or exp["full_name"] == experiment_name
    ]

    # Debug logging
    logger.info(f"Matching experiments: {matching_experiments}")

    if not matching_experiments:
        raise ValueError(f"No experiments found matching {experiment_name}")

    latest_experiment = max(matching_experiments, key=lambda x: x.get("version", 0))
    logger.info(f"Selected experiment: {latest_experiment}")

    # Use current timestamp as score_ts if no created_at is available
    score_ts = datetime.now().isoformat()  # Convert to ISO format string

    # Prepare results DataFrame
    results_dict = {
        "game_id": df["game_id"].to_numpy(),
        "name": df["name"].to_numpy(),
        "year_published": df["year_published"].to_numpy(),
        "predicted_complexity": predicted_complexity,
        "model_id": np.full(len(df), str(experiment_name)),  # Ensure string type
        "score_ts": np.full(len(df), score_ts),
        "load_ts": np.full(len(df), load_ts),
    }

    # Optionally include original complexity if available
    if "complexity" in df.columns:
        results_dict["original_complexity"] = df["complexity"].to_numpy()

    # Create DataFrame using results_dict
    try:
        results = pl.DataFrame(results_dict)
    except Exception as e:
        # Log detailed error information
        logger.error(f"Error creating DataFrame: {e}")
        for key, value in results_dict.items():
            logger.error(f"Column '{key}': type={type(value)}, length={len(value)}")
            if len(value) > 0:
                logger.error(f"First value type: {type(value[0])}")
        raise

    return results


def generate_complexity_predictions(
    experiment_name: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Generate complexity predictions for all games.

    Args:
        experiment_name: Name of complexity experiment (optional)
        start_year: First year of data to include (optional)
        end_year: Last year of data to include (optional)
        output_path: Path to save predictions (optional)

    Returns:
        DataFrame with complexity predictions
    """
    # Determine experiment name if not provided
    if experiment_name is None:
        tracker = ExperimentTracker("complexity")
        experiments = tracker.list_experiments()
        if not experiments:
            raise ValueError("No complexity model experiments found.")
        experiment = max(experiments, key=lambda x: x.get("version", 0))
        experiment_name = experiment["name"]

    logger.info(f"Using complexity experiment: {experiment_name}")

    # Load pipeline
    pipeline = load_model(experiment_name, model_type="complexity")

    # Load data
    from src.data.loader import BGGDataLoader
    from src.data.config import load_config

    config = load_config()
    loader = BGGDataLoader(config)

    # Load all games with non-null year_published
    df = loader.load_data(where_clause="year_published IS NOT NULL", preprocessor=None)

    # Predict data
    prediction_results = predict_data(
        pipeline, df, experiment_name, model_type="complexity"
    )

    # Unpack prediction results
    if prediction_results is None:
        raise ValueError("Prediction data returned None")

    # Unpack prediction results based on the expected tuple structure
    if len(prediction_results) == 3:
        predicted_complexity, _, _ = prediction_results
    elif len(prediction_results) == 2:
        predicted_complexity, _ = prediction_results
    else:
        raise ValueError(f"Unexpected prediction results format: {prediction_results}")

    # Validate predictions
    if predicted_complexity is None:
        raise ValueError("Predicted complexity is None")

    # Prepare results
    results = prepare_complexity_results(df, predicted_complexity, experiment_name)

    # Determine output path
    if output_path is None:
        base_predictions_dir = Path("data/estimates")
        base_predictions_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_predictions_dir / f"{experiment_name}_estimates.parquet"

    # Save results
    results.write_parquet(str(output_path))
    logger.info(f"Complexity predictions saved to {output_path}")

    # Display sample of results
    logger.info("\nSample predictions:")
    logger.info(results.head().to_pandas().to_string())

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate Complexity Predictions")
    parser.add_argument(
        "--experiment", help="Name of complexity experiment (default: latest)"
    )
    parser.add_argument("--start-year", type=int, help="First year of data to include")
    parser.add_argument("--end-year", type=int, help="Last year of data to include")
    parser.add_argument("--output", help="Path to save predictions parquet file")

    args = parser.parse_args()

    generate_complexity_predictions(
        experiment_name=args.experiment,
        start_year=args.start_year,
        end_year=args.end_year,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
