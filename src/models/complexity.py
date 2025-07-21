"""Train/Tune/Test Complexity Regression Model for Board Game Complexity Prediction"""

import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Project imports
from src.models.experiments import (
    ExperimentTracker,
    log_experiment,
    mean_absolute_percentage_error,
)

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, QuantileRegressor
from sklearn.base import clone
from sklearn.base import BaseEstimator, clone
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# Project imports
from src.data.config import load_config
from src.data.loader import BGGDataLoader
from src.features.preprocessor import create_bgg_preprocessor
from src.models.splitting import time_based_split
from src.models.training import (
    load_data,
    create_data_splits,
    select_X_y,
    create_preprocessing_pipeline,
    preprocess_data,
    tune_model,
    evaluate_model,
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
    """Constrain predictions to be between 1 and 5."""
    return np.clip(predictions, 1, 5)


def calculate_complexity_weights(
    complexities: np.ndarray,
    base: float = 10.0,  # Increased base for more dramatic weighting
    min_weight: float = 1.0,
    max_weight: float = 100.0,  # Added max weight to prevent extreme values
) -> np.ndarray:
    """
    Calculate exponential weights for complexity values.

    Args:
        complexities (array-like): Complexity values
        base (float): Base for exponential weighting. Higher values
                      create more dramatic weight increases
        min_weight (float): Minimum weight to apply
        max_weight (float): Maximum weight to prevent extreme values

    Returns:
        numpy array of weights normalized to have mean 1.0
    """
    logger = logging.getLogger(__name__)

    # Log input complexity distribution
    logger.info("Complexity Weight Calculation Diagnostic:")
    logger.info(
        f"  Complexity Range: min={complexities.min():.2f}, max={complexities.max():.2f}"
    )
    logger.info(f"  Complexity Mean: {complexities.mean():.2f}")
    logger.info(f"  Complexity Std Dev: {complexities.std():.2f}")

    # Normalize complexities to start from 0
    normalized_complexities = complexities - complexities.min()

    # Calculate exponential weights with more dramatic scaling
    weights = np.power(base, normalized_complexities)

    # Clip weights to prevent extreme values
    weights = np.clip(weights, min_weight, max_weight)

    # Normalize weights to have mean 1.0
    weights = weights / np.mean(weights)

    # Log weight distribution
    logger.info("  Weight Distribution:")
    logger.info(f"    Weight Range: min={weights.min():.2f}, max={weights.max():.2f}")
    logger.info(f"    Weight Mean: {weights.mean():.2f}")
    logger.info(f"    Weight Std Dev: {weights.std():.2f}")

    # Log a few example mappings
    logger.info("  Sample Complexity to Weight Mapping:")
    for complexity, weight in zip(complexities[:10], weights[:10]):
        logger.info(f"    Complexity {complexity:.2f} -> Weight {weight:.2f}")

    return weights


def configure_model(model_name: str) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Set up regression model and parameter grid."""
    model_MAPPING = {
        "linear": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "lightgbm": lgb.LGBMRegressor,
        "quantile": QuantileRegressor,
    }

    PARAM_GRIDS = {
        "linear": {},  # Linear Regression has no hyperparameters to tune
        "ridge": {
            "model__alpha": [0.0001, 0.0005, 0.01, 0.1, 1.0, 5],  # Expanded alpha range
            "model__solver": ["auto"],
            "model__fit_intercept": [True],
        },
        "lasso": {
            "model__alpha": [0.1, 1.0, 10.0],
            "model__selection": ["cyclic", "random"],
        },
        "lightgbm": {
            "model__n_estimators": [500, 1000],
            "model__learning_rate": [0.01, 0.05],
            "model__max_depth": [3, 7, 11],
            "model__num_leaves": [10, 20, 50],
            "model__min_child_samples": [20],
        },
        "quantile": {
            "model__quantile": [0.4, 0.5, 0.6],  # Different quantiles to explore
            "model__solver": ["highs-ds"],
            "model__alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0],  # Regularization strength
            "model__fit_intercept": [True],
        },
    }

    model = model_MAPPING[model_name]()
    param_grid = PARAM_GRIDS[model_name]

    return model, param_grid


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train/Tune/Test Complexity Regression Model"
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
    parser.add_argument("--min-weights", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="./models/experiments")
    parser.add_argument("--experiment", type=str, default="complexity_regression")
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
        "--model",
        type=str,
        default="ridge",
        choices=["linear", "ridge", "lasso", "lightgbm", "quantile"],
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
        help="Enable sample weights based on complexity values",
    )
    parser.add_argument(
        "--preprocessor-type",
        type=str,
        default="linear",
        choices=["linear", "tree"],
        help="Type of preprocessor to use",
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
    """Main script for training, tuning, and testing a complexity regression model."""
    # Parse arguments and setup logging
    args = parse_arguments()
    logger = setup_logging()

    # Load and split data
    # load full data
    df = load_data(
        local_data_path=args.local_data,
        min_weights=args.min_weights,
        end_train_year=args.test_end_year,
    )

    # filtered for training/evaluation
    logger.info(f"Training on games with at least {args.min_weights} weights")
    train_df, tune_df, test_df = create_data_splits(
        df,
        train_end_year=args.train_end_year,
        tune_start_year=args.tune_start_year,
        tune_end_year=args.tune_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )

    # Get X, y splits
    train_X, train_y = select_X_y(train_df, y_column="complexity")
    tune_X, tune_y = select_X_y(tune_df, y_column="complexity")
    test_X, test_y = select_X_y(test_df, y_column="complexity")

    # Setup model and pipeline
    model, param_grid = configure_model(args.model)
    preprocessor = create_preprocessing_pipeline(model_type=args.preprocessor_type)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Log experiment details
    logger.info(f"Training experiment: {args.experiment}")
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"Preprocessor Type: {args.preprocessor_type}")
    logger.info(f"Parameter Grid: {param_grid}")
    logger.info(f"Optimization metric: {args.metric}")
    logger.info(f"Feature dimensions: Train {train_X.shape}, Tune {tune_X.shape}")

    # Calculate sample weights for training data
    train_sample_weights = None
    if args.use_sample_weights:
        train_sample_weights = calculate_complexity_weights(train_y.values)
        logger.info("Training Sample Weights Diagnostic:")
        logger.info(
            f"  Weight Range: min={train_sample_weights.min():.2f}, max={train_sample_weights.max():.2f}"
        )
        logger.info(f"  Weight Mean: {train_sample_weights.mean():.2f}")

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
        sample_weights=train_sample_weights if args.use_sample_weights else None,
    )

    # Fit on train data with optional sample weights
    if args.use_sample_weights:
        train_pipeline = clone(tuned_pipeline).fit(
            train_X,
            train_y,
            model__sample_weight=(
                np.asarray(train_sample_weights)
                if train_sample_weights is not None
                else None
            ),
        )
    else:
        train_pipeline = clone(tuned_pipeline).fit(train_X, train_y)
    train_metrics = evaluate_model(train_pipeline, train_X, train_y, "training")

    # Evaluate tuning set
    tune_metrics = evaluate_model(train_pipeline, tune_X, tune_y, "tuning")

    # Fit final model on combined train+tune data
    logger.info("Fitting final model on combined training + validation data...")
    X_combined = pd.concat([train_X, tune_X])
    y_combined = pd.concat([train_y, tune_y])

    # Calculate sample weights for combined data
    combined_sample_weights = None
    if args.use_sample_weights:
        combined_sample_weights = calculate_complexity_weights(y_combined.values)
        logger.info("Combined Sample Weights Diagnostic:")
        logger.info(
            f"  Weight Range: min={combined_sample_weights.min():.2f}, max={combined_sample_weights.max():.2f}"
        )
        logger.info(f"  Weight Mean: {combined_sample_weights.mean():.2f}")

    # Fit final model with optional sample weights
    if args.use_sample_weights:
        final_pipeline = clone(tuned_pipeline).fit(
            X_combined,
            y_combined,
            model__sample_weight=(
                np.asarray(combined_sample_weights)
                if combined_sample_weights is not None
                else None
            ),
        )
    else:
        final_pipeline = clone(tuned_pipeline).fit(X_combined, y_combined)

    # Evaluate on test set (filtered)
    test_metrics = evaluate_model(final_pipeline, test_X, test_y, "test")

    # Log experiment results
    tracker = ExperimentTracker("complexity", args.output_dir)

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
            "model_type": "complexity_regression",
            "target": "complexity",
            "min_weights": args.min_weights,
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
            "target": "complexity",
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

    # Save complexity predictions for the entire dataset
    try:
        # Ensure predictions directory exists
        import os

        predictions_dir = os.path.join(args.output_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        # Predict complexity for the entire dataset
        df_pandas = df.to_pandas()
        complexity_predictions = final_pipeline.predict(df_pandas)

        # Create predictions DataFrame
        predictions_df = pl.DataFrame(
            {"game_id": df["game_id"], "predicted_complexity": complexity_predictions}
        )

        # Save predictions to parquet
        predictions_path = os.path.join(predictions_dir, f"{args.experiment}.parquet")
        predictions_df.write_parquet(predictions_path)
        logger.info(f"Complexity predictions saved to {predictions_path}")

    except Exception as e:
        logger.error(f"Failed to save complexity predictions: {e}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
