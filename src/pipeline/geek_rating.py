"""Geek rating prediction entry point.

Usage:
    uv run -m src.pipeline.geek_rating \
        --hurdle lightgbm-hurdle \
        --complexity catboost-complexity \
        --rating catboost-rating \
        --users-rated ridge-users_rated \
        --start-year 2024 \
        --end-year 2026
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl

from src.data.loader import BGGDataLoader
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.models.experiments import ExperimentTracker
from src.models.outcomes.geek_rating import GeekRatingModel

logger = setup_logging()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    config = load_config()
    complexity_predictions_path = config.models["complexity"].predictions_path

    parser = argparse.ArgumentParser(description="Predict geek ratings for board games")

    # Model experiment arguments
    parser.add_argument(
        "--hurdle",
        required=True,
        help="Experiment name for hurdle model",
    )
    parser.add_argument(
        "--complexity",
        required=True,
        help="Experiment name for complexity model",
    )
    parser.add_argument(
        "--rating",
        required=True,
        help="Experiment name for rating model",
    )
    parser.add_argument(
        "--users-rated",
        required=True,
        help="Experiment name for users rated model",
    )

    # Data arguments
    parser.add_argument(
        "--local-complexity-path",
        type=str,
        default=complexity_predictions_path,
        help="Path to local complexity predictions parquet file",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="Start year for filtering games (inclusive)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="End year for filtering games (inclusive)",
    )

    # Prediction parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold for hurdle model (default: from experiment)",
    )
    parser.add_argument(
        "--prior-rating",
        type=float,
        default=5.5,
        help="Prior mean rating for Bayesian average (default: 5.5)",
    )
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=2000,
        help="Weight given to prior rating (default: 2000)",
    )

    # Output arguments
    parser.add_argument(
        "--experiment",
        default="geek_rating_prediction",
        help="Name of the experiment for tracking",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/predictions",
        help="Base directory for output files",
    )

    return parser.parse_args()


def main():
    """Main entry point for geek rating prediction."""
    args = parse_arguments()

    # Create experiment tracker
    tracker = ExperimentTracker(model_type="geek_rating")

    # Create experiment with metadata
    experiment = tracker.create_experiment(
        name=args.experiment,
        description="Geek rating predictions using multiple model experiments",
        metadata={
            "model_experiments": {
                "hurdle": args.hurdle,
                "complexity": args.complexity,
                "rating": args.rating,
                "users_rated": args.users_rated,
            },
            "prediction_parameters": {
                "threshold": args.threshold,
                "prior_rating": args.prior_rating,
                "prior_weight": args.prior_weight,
            },
        },
    )

    # Load configuration and data
    config = load_config()
    bigquery_config = config.get_bigquery_config()
    loader = BGGDataLoader(bigquery_config)

    # Load complexity predictions
    logger.info(f"Loading complexity predictions from {args.local_complexity_path}")
    complexity_df = pl.read_parquet(args.local_complexity_path)
    logger.info(f"Loaded {len(complexity_df)} complexity predictions")

    # Construct WHERE clause for year filtering
    where_clauses = []
    if args.start_year is not None:
        where_clauses.append(f"year_published >= {args.start_year}")
    if args.end_year is not None:
        where_clauses.append(f"year_published <= {args.end_year}")

    where_clause = " AND ".join(where_clauses) if where_clauses else ""
    df = loader.load_data(where_clause=where_clause, preprocessor=None)

    # Join with complexity predictions
    df = df.join(complexity_df, on="game_id", how="inner")
    logger.info(f"After joining with complexity predictions: {len(df)} games")

    logger.info(
        f"Filtered to {len(df)} games between years "
        f"{args.start_year or 'min'} and {args.end_year or 'max'}"
    )

    # Convert to pandas for prediction
    df_pandas = df.to_pandas()

    # Create and load model
    model = GeekRatingModel.from_experiments(
        hurdle_experiment=args.hurdle,
        complexity_experiment=args.complexity,
        rating_experiment=args.rating,
        users_rated_experiment=args.users_rated,
        prior_rating=args.prior_rating,
        prior_weight=args.prior_weight,
        hurdle_threshold=args.threshold,
    )

    # Generate predictions
    predictions = model.predict(df_pandas)

    # Rename for consistency with old code
    predictions = predictions.rename(columns={"predicted_geek_rating": "prediction"})

    # Convert to Polars for saving
    results = pl.from_pandas(predictions)

    # Ensure predictions directory exists
    predictions_dir = Path(args.output_dir) / tracker.model_type / args.experiment
    predictions_dir.mkdir(parents=True, exist_ok=True)

    output_path = predictions_dir / "predictions.parquet"
    logger.info(f"Saving predictions to: {output_path}")

    results.write_parquet(str(output_path))

    # Get actual values for logging
    if "geek_rating" in df.columns:
        actuals = np.where(
            df["geek_rating"].to_numpy() == 0, np.nan, df["geek_rating"].to_numpy()
        )
    else:
        actuals = np.full(len(results), np.nan)

    # Log predictions to experiment
    experiment.log_predictions(
        predictions=results["prediction"].to_numpy(),
        actuals=actuals,
        df=results,
        dataset="test",
    )

    logger.info("Experiment tracked:")
    logger.info(f"  Name: {args.experiment}")
    logger.info("  Experiments used:")
    logger.info(f"    Hurdle: {args.hurdle}")
    logger.info(f"    Complexity: {args.complexity}")
    logger.info(f"    Rating: {args.rating}")
    logger.info(f"    Users Rated: {args.users_rated}")


if __name__ == "__main__":
    main()
