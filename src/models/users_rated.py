"""Train/Tune/Test Users Rated Regression Model for Board Game User Rating Prediction"""

import logging
import argparse
from pathlib import Path
from typing import Optional

# Project imports
from src.models.experiments import (
    ExperimentTracker,
    log_experiment,
)

import numpy as np
import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Project imports
from src.utils.config import load_config
from src.data.loader import BGGDataLoader
from src.models.training import (
    load_data,
    create_data_splits,
    select_X_y,
    create_preprocessing_pipeline,
    tune_model,
    evaluate_model,
    configure_model,
)


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging for the training process."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def constrain_predictions(predictions: np.ndarray) -> np.ndarray:
    """Constrain predictions to be non-negative integers after exponentiating."""
    return np.maximum(np.round(np.exp(predictions)), 0).astype(int)


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train/Tune/Test Users Rated Regression Model"
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
        default=0,
        help="Minimum number of ratings threshold",
    )
    parser.add_argument("--output-dir", type=str, default="./models/experiments")
    parser.add_argument("--experiment", type=str, default="users_rated_regression")
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


def main():
    """Main script for training, tuning, and testing a users rated regression model."""
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

    # Get X, y splits using original target
    train_X, train_y = select_X_y(train_df, y_column="log_users_rated")
    tune_X, tune_y = select_X_y(tune_df, y_column="log_users_rated")
    test_X, test_y = select_X_y(test_df, y_column="log_users_rated")

    # Setup model and pipeline
    model, param_grid = configure_model(args.model)
    preprocessor = create_preprocessing_pipeline(
        model_type=args.preprocessor_type,
        model_name=args.model,
        preserve_columns=["year_published", "predicted_complexity"],
    )

    # # Wrap the model with TransformedTargetRegressor
    # transformed_model = TransformedTargetRegressor(
    #     regressor=model,
    #     func=np.log1p,  # log(1 + y)
    #     inverse_func=np.expm1,  # exp(y) - 1
    #     check_inverse=True
    # )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Adjust parameter grid for Pipeline
    param_grid = {"model__" + k: v for k, v in param_grid.items()}

    # Log experiment details
    logger.info(f"Training experiment: {args.experiment}")
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"Preprocessor Type: {args.preprocessor_type}")
    logger.info(f"Parameter Grid: {param_grid}")
    logger.info(f"Optimization metric: {args.metric}")
    logger.info(f"Feature dimensions: Train {train_X.shape}, Tune {tune_X.shape}")

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

    # Fit final model
    final_pipeline = clone(tuned_pipeline).fit(X_combined, y_combined)

    # Evaluate on test set (filtered)
    # Compute metrics on log scale
    test_metrics = evaluate_model(final_pipeline, test_X, test_y, "test")

    # Compute additional metrics on original scale
    test_predictions_log = final_pipeline.predict(test_X)
    test_predictions = np.exp(test_predictions_log)
    test_actual = np.exp(test_y)

    # Compute additional metrics on original scale
    test_metrics["mse_original_scale"] = mean_squared_error(
        test_actual, test_predictions
    )
    test_metrics["mae_original_scale"] = mean_absolute_error(
        test_actual, test_predictions
    )
    test_metrics["r2_original_scale"] = r2_score(test_actual, test_predictions)

    # Log experiment results
    tracker = ExperimentTracker("users_rated", args.output_dir)

    # Create experiment
    experiment = tracker.create_experiment(
        name=args.experiment,
        description=args.description,
        metadata={
            "train_end_year_exclusive": args.train_end_year,
            "tune_start_year": args.tune_start_year,
            "tune_end_year": args.tune_end_year,
            "test_start_year": args.test_start_year,
            "test_end_year": args.test_end_year,
            "model_type": "users_rated_regression",
            "target": "log_users_rated",
            "min_ratings": args.min_ratings,
        },
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
            "target": "log_users_rated",
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
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
