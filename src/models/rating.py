"""Train/Tune/Test Rating Regression Model for Board Game Average Rating Prediction"""

import logging
import argparse
from typing import Dict

import numpy as np
import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Project imports
# Project imports
from src.utils.logging import setup_logging
from src.models.experiments import (
    ExperimentTracker,
    log_experiment,
)
from src.utils.config import load_config
from src.data.loader import BGGDataLoader
from src.models.training import (
    load_data,
    create_data_splits,
    select_X_y,
    create_preprocessing_pipeline,
    tune_model,
    evaluate_model,
    calculate_sample_weights,
    configure_model,
)


def constrain_predictions(predictions: np.ndarray) -> np.ndarray:
    """Constrain predictions to be between 1 and 10."""
    return np.clip(predictions, 1, 10)


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train/Tune/Test Rating Regression Model"
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2022,
        help="End year for training (exclusive)",
    )
    parser.add_argument(
        "--tune-start-year",
        type=int,
        default=2022,
        help="Start year for tuning (inclusive)",
    )
    parser.add_argument(
        "--tune-end-year",
        type=int,
        default=2023,
        help="End year for tuning (inclusive)",
    )
    parser.add_argument(
        "--test-start-year",
        type=int,
        default=2024,
        help="Start year for testing (inclusive)",
    )
    parser.add_argument(
        "--test-end-year",
        type=int,
        default=2025,
        help="End year for testing (inclusive)",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=5,
        help="Minimum number of ratings threshold",
    )
    parser.add_argument("--output-dir", type=str, default="./models/experiments")
    parser.add_argument("--experiment", type=str, default="rating_regression")
    parser.add_argument(
        "--description", type=str, default=None, help="Description of the experiment"
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to local parquet file for training data",
    )
    parser.add_argument(
        "--complexity-experiment",
        type=str,
        required=True,
        help="Name of complexity experiment to use for predictions",
    )
    parser.add_argument(
        "--local-complexity-path",
        type=str,
        default=None,
        help="Path to local complexity predictions parquet file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        help="Regression model type to use",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.5,
        help="Quantile to use when model is 'quantile' (default: 0.5, median)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rmse",
        choices=["rmse", "mae", "r2", "mape"],
        help="Metric to optimize during hyperparameter tuning",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Number of iterations without improvement before early stopping",
    )
    parser.add_argument(
        "--use-sample-weights",
        action="store_true",
        default=False,
        help="Enable sample weights based on number of ratings",
    )
    parser.add_argument(
        "--sample-weight-column",
        type=str,
        default="users_rated",
        help="Column to use for calculating sample weights (default: users_rated)",
    )
    parser.add_argument(
        "--preprocessor-type",
        type=str,
        default="auto",
        choices=["auto", "linear", "tree"],
        help="Type of preprocessor to use (auto: automatically select based on model type)",
    )

    args = parser.parse_args()

    # Validate year ranges
    if args.tune_start_year != args.train_end_year:
        raise ValueError(
            f"tune_start_year ({args.tune_start_year}) must equal train_end_year ({args.train_end_year})"
        )

    if not (
        args.tune_start_year
        <= args.tune_end_year
        < args.test_start_year
        <= args.test_end_year
    ):
        raise ValueError(
            "Invalid year ranges. Must satisfy: tune_start <= tune_end < test_start <= test_end"
        )

    # Validate quantile argument
    if args.model == "quantile" and (args.quantile < 0 or args.quantile > 1):
        raise ValueError("Quantile must be between 0 and 1")

    return args


def stratified_evaluation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    users_rated: pd.Series,
    dataset_name: str = "test",
) -> Dict[str, Dict[str, float]]:
    """
    Perform stratified evaluation of model performance across different rating count buckets.

    Args:
        model: Trained model
        X: Features
        y: True target values
        users_rated: Number of users who rated each game
        dataset_name: Name of the dataset for logging

    Returns:
        Dictionary of performance metrics for each rating count bucket
    """
    logger = logging.getLogger(__name__)

    # Define rating count buckets
    buckets = {
        "high_confidence": users_rated >= 25,
        "medium_confidence": (users_rated >= 15) & (users_rated < 25),
        "low_confidence": (users_rated >= 10) & (users_rated < 15),
    }

    # Compute metrics for each bucket
    stratified_metrics = {}
    for bucket_name, mask in buckets.items():
        # Convert mask to boolean index
        mask_index = mask.to_numpy()

        if mask_index.sum() == 0:
            logger.info(f"No games in {bucket_name} bucket for {dataset_name}")
            continue

        X_bucket = X.loc[mask_index]
        y_bucket = y.loc[mask_index]

        # Compute metrics for this bucket
        bucket_metrics = {
            "mse": mean_squared_error(y_bucket, model.predict(X_bucket)),
            "rmse": np.sqrt(mean_squared_error(y_bucket, model.predict(X_bucket))),
            "mae": mean_absolute_error(y_bucket, model.predict(X_bucket)),
            "r2": r2_score(y_bucket, model.predict(X_bucket)),
            "n_samples": len(y_bucket),
        }

        logger.info(f"{dataset_name} - {bucket_name} Metrics:")
        for metric, value in bucket_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        stratified_metrics[bucket_name] = bucket_metrics

    return stratified_metrics


def main():
    """Main script for training, tuning, and testing a rating regression model."""
    # Parse arguments and setup logging
    args = parse_arguments()
    logger = setup_logging()

    # Load and split data
    # load full data
    df = load_data(
        local_data_path=args.local_data,
        min_ratings=args.min_ratings,
        end_train_year=args.test_end_year,
    )

    # Load complexity predictions
    logger.info(
        f"Loading complexity predictions for experiment: {args.complexity_experiment}"
    )

    # Try to load from local path first
    if args.local_complexity_path:
        try:
            complexity_df = pl.read_parquet(args.local_complexity_path)
            logger.info(
                f"Loaded local complexity predictions from {args.local_complexity_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load local complexity predictions: {e}")
            raise
    else:
        # If no local path, load from BigQuery
        try:
            config = load_config()
            loader = BGGDataLoader(config)

            # Construct query to get complexity predictions for a specific experiment
            query = f"""
            SELECT game_id, predicted_complexity
            FROM `{config.project_id}.{config.dataset}.complexity_predictions`
            WHERE model_id = '{args.complexity_experiment}'
            """

            complexity_df = pl.from_pandas(loader.client.query(query).to_dataframe())
            logger.info(
                f"Loaded complexity predictions from BigQuery for experiment {args.complexity_experiment}"
            )
        except Exception as e:
            logger.error(f"Failed to load complexity predictions from BigQuery: {e}")
            raise

    # Validate complexity predictions
    if len(complexity_df) == 0:
        raise ValueError(
            f"No complexity predictions found for experiment {args.complexity_experiment}"
        )

    # Join complexity predictions with main dataframe
    df = df.join(complexity_df, on="game_id", how="inner")

    # Validate join result
    if len(df) == 0:
        raise ValueError("No games remain after joining with complexity predictions")

    logger.info(f"Joined complexity predictions: {len(df)} games remain")

    # filtered for training/evaluation
    logger.info(f"Training on games with at least {args.min_ratings} ratings")
    train_df, tune_df, test_df = create_data_splits(
        df,
        train_end_year=args.train_end_year,
        tune_start_year=args.tune_start_year,
        tune_end_year=args.tune_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )

    # Get X, y splits
    train_X, train_y = select_X_y(train_df, y_column="rating")
    tune_X, tune_y = select_X_y(tune_df, y_column="rating")
    test_X, test_y = select_X_y(test_df, y_column="rating")

    # Setup model and pipeline
    model, param_grid = configure_model(args.model)
    preprocessor = create_preprocessing_pipeline(
        model_type=args.preprocessor_type,
        model_name=args.model,
        preserve_columns=["year_published", "predicted_complexity"],
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Log experiment details
    logger.info(f"Training experiment: {args.experiment}")
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"Preprocessor Type: {args.preprocessor_type}")
    logger.info(f"Parameter Grid: {param_grid}")
    logger.info(f"Optimization metric: {args.metric}")
    logger.info(f"Feature dimensions: Train {train_X.shape}, Tune {tune_X.shape}")

    # Calculate sample weights if enabled
    train_sample_weights = None
    tune_sample_weights = None
    if args.use_sample_weights:
        train_sample_weights = calculate_sample_weights(
            train_df, weight_column=args.sample_weight_column
        )
        tune_sample_weights = calculate_sample_weights(
            tune_df, weight_column=args.sample_weight_column
        )

        logger.info("Sample Weights Diagnostic:")
        logger.info(
            f"  Train weights - min: {train_sample_weights.min():.4f}, max: {train_sample_weights.max():.4f}, mean: {train_sample_weights.mean():.4f}"
        )
        logger.info(
            f"  Tune weights - min: {tune_sample_weights.min():.4f}, max: {tune_sample_weights.max():.4f}, mean: {tune_sample_weights.mean():.4f}"
        )

    # Tune model
    tuned_pipeline, best_params = tune_model(
        pipeline=pipeline,
        train_X=train_X,
        train_y=train_y,
        tune_X=tune_X,
        tune_y=tune_y,
        param_grid=param_grid,
        metric=args.metric,
        patience=args.patience,
        sample_weights=train_sample_weights,
    )

    # Fit on train data
    train_pipeline = clone(tuned_pipeline).fit(train_X, train_y)
    train_metrics = evaluate_model(train_pipeline, train_X, train_y, "training")

    # Evaluate tuning set
    tune_metrics = evaluate_model(train_pipeline, tune_X, tune_y, "tuning")

    # Fit final model on combined train+tune data
    logger.info("Fitting final model on combined training + validation data...")
    X_combined = pd.concat([train_X, tune_X])
    y_combined = pd.concat([train_y, tune_y])

    # Combine sample weights if used
    combined_sample_weights = None
    if args.use_sample_weights:
        combined_sample_weights = np.concatenate(
            [train_sample_weights, tune_sample_weights]
        )
        logger.info("Combined Sample Weights Diagnostic:")
        logger.info(
            f"  Combined weights - min: {combined_sample_weights.min():.4f}, max: {combined_sample_weights.max():.4f}, mean: {combined_sample_weights.mean():.4f}"
        )

    # Fit final model with step-specific sample weights
    final_pipeline = clone(tuned_pipeline).fit(
        X_combined,
        y_combined,
        model__sample_weight=(
            np.asarray(combined_sample_weights)
            if combined_sample_weights is not None
            else None
        ),
    )

    # Evaluate on test set (filtered)
    test_metrics = evaluate_model(final_pipeline, test_X, test_y, "test")

    # Perform stratified evaluation
    stratified_test_metrics = stratified_evaluation(
        final_pipeline, test_X, test_y, test_df["users_rated"], dataset_name="test"
    )

    # Log experiment results
    tracker = ExperimentTracker("rating", args.output_dir)

    # Create experiment
    experiment_metadata = {
        "train_end_year_exclusive": args.train_end_year,
        "tune_start_year": args.tune_start_year,
        "tune_end_year": args.tune_end_year,
        "test_start_year": args.test_start_year,
        "test_end_year": args.test_end_year,
        "model_type": "rating_regression",
        "target": "rating",
        "min_ratings": args.min_ratings,
    }

    # Add sample weight metadata if used
    # Add sample weight metadata if used
    if args.use_sample_weights:
        experiment_metadata.update(
            {
                "sample_weights": {
                    "column": "users_rated",
                }
            }
        )

    experiment = tracker.create_experiment(
        name=args.experiment,
        description=args.description,
        metadata=experiment_metadata,
        config={
            "model_params": best_params,
            "preprocessing": {
                "category_min_freq": 0,
                "mechanic_min_freq": 0,
                "designer_min_freq": 10,
                "artist_min_freq": 10,
                "publisher_min_freq": 5,
                "family_min_freq": 10,
                "max_artist_features": 500,
                "max_publisher_features": 250,
                "max_designer_features": 500,
                "max_family_features": 500,
                "max_mechanic_features": 500,
                "max_category_features": 500,
                "create_category_features": True,
                "create_mechanic_features": True,
                "create_designer_features": True,
                "create_artist_features": True,
                "create_publisher_features": True,
                "create_family_features": True,
                "create_player_dummies": True,
                "include_base_numeric": True,
            },
            "data_splits": {
                "train_end_year": args.train_end_year,
                "tune_start_year": args.tune_start_year,
                "tune_end_year": args.tune_end_year,
                "test_start_year": args.test_start_year,
                "test_end_year": args.test_end_year,
            },
            "target": "rating",
            "target_type": "regression",
        },
    )

    # Learning curve generation removed
    logger.info("Skipping learning curve generation")

    log_experiment(
        experiment=experiment,
        pipeline=final_pipeline,
        train_metrics=train_metrics,
        tune_metrics=tune_metrics,
        test_metrics=test_metrics,
        train_df=train_df,
        tune_df=tune_df,
        test_df=test_df,
        train_X=train_X,
        tune_X=tune_X,
        test_X=test_X,
        train_y=train_y,
        tune_y=tune_y,
        test_y=test_y,
        best_params=best_params,
        args=args,
        model_type="regression",
        stratified_metrics=stratified_test_metrics,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
