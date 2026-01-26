"""Unified training entry point for outcome models."""

import argparse
import logging
from typing import Optional, Type

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.models.outcomes.base import TrainableModel, TrainingConfig
from src.models.outcomes.data import (
    load_training_data,
    create_data_splits,
    select_X_y,
)
from src.models.training import (
    create_preprocessing_pipeline,
    tune_model,
    evaluate_model,
    calculate_sample_weights,
)
from src.models.experiments import ExperimentTracker, log_experiment
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


# Registry of available model classes
MODEL_REGISTRY = {}


def register_model(model_class: Type[TrainableModel]) -> Type[TrainableModel]:
    """Register a model class in the registry.

    Args:
        model_class: TrainableModel subclass to register.

    Returns:
        The same class (allows use as decorator).
    """
    MODEL_REGISTRY[model_class.model_type] = model_class
    return model_class


def get_model_class(model_type: str) -> Type[TrainableModel]:
    """Get model class from registry.

    Args:
        model_type: Model type name (e.g., 'hurdle', 'complexity').

    Returns:
        TrainableModel subclass.

    Raises:
        ValueError: If model type not found.
    """
    # Lazy import to populate registry
    if not MODEL_REGISTRY:
        _populate_registry()

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_type]


def _populate_registry():
    """Import and register all model classes."""
    from src.models.outcomes.hurdle import HurdleModel
    from src.models.outcomes.complexity import ComplexityModel
    from src.models.outcomes.rating import RatingModel
    from src.models.outcomes.users_rated import UsersRatedModel

    register_model(HurdleModel)
    register_model(ComplexityModel)
    register_model(RatingModel)
    register_model(UsersRatedModel)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train an outcome prediction model")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["hurdle", "complexity", "rating", "users_rated"],
        help="Model type to train",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Algorithm to use (default: from config)",
    )

    # Experiment tracking
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Experiment description",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/experiments",
        help="Output directory for experiments",
    )

    # Data options
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to local parquet file (instead of BigQuery)",
    )
    parser.add_argument(
        "--complexity-predictions",
        type=str,
        default=None,
        help="Path to complexity predictions parquet (for rating/users_rated)",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        default=False,
        help="Include text embeddings as features",
    )

    # Year splits
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--tune-start-year", type=int, default=2022)
    parser.add_argument("--tune-end-year", type=int, default=2023)
    parser.add_argument("--test-start-year", type=int, default=2024)
    parser.add_argument("--test-end-year", type=int, default=2025)

    # Training options
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric to optimize (default: rmse for regression, log_loss for classification)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--use-sample-weights",
        action="store_true",
        default=False,
        help="Use sample weights during training",
    )
    parser.add_argument(
        "--sample-weight-column",
        type=str,
        default=None,
        help="Column for sample weights",
    )
    parser.add_argument(
        "--preprocessor-type",
        type=str,
        default="auto",
        choices=["auto", "linear", "tree"],
        help="Preprocessor type",
    )

    args = parser.parse_args()

    # Validate year ranges
    if args.tune_start_year != args.train_end_year:
        raise ValueError(
            f"tune_start_year ({args.tune_start_year}) must equal "
            f"train_end_year ({args.train_end_year})"
        )

    if not (
        args.tune_start_year
        <= args.tune_end_year
        < args.test_start_year
        <= args.test_end_year
    ):
        raise ValueError(
            "Invalid year ranges. Must satisfy: "
            "tune_start <= tune_end < test_start <= test_end"
        )

    return args


def train_model(
    model_class: Type[TrainableModel],
    args: Optional[argparse.Namespace] = None,
) -> TrainableModel:
    """Train a model using the standardized pipeline.

    Args:
        model_class: TrainableModel subclass to instantiate and train.
        args: Command-line arguments. If None, will parse from command line.

    Returns:
        Trained model instance.
    """
    if args is None:
        args = parse_arguments()

    setup_logging()

    # Instantiate model
    model = model_class()

    # Determine algorithm
    algorithm = args.algorithm
    if algorithm is None:
        # Could load from config here
        algorithm = "ridge" if model.model_task == "regression" else "lightgbm"

    logger.info(f"Training {model.model_type} model with {algorithm}")

    # Load data
    df = load_training_data(
        data_config=model.data_config,
        end_year=args.test_end_year,
        use_embeddings=args.use_embeddings,
        complexity_predictions_path=args.complexity_predictions,
        local_data_path=args.local_data,
    )

    # Create splits
    train_df, tune_df, test_df = create_data_splits(
        df,
        train_end_year=args.train_end_year,
        tune_start_year=args.tune_start_year,
        tune_end_year=args.tune_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )

    # Extract X, y
    train_X, train_y = select_X_y(train_df, model.target_column)
    tune_X, tune_y = select_X_y(tune_df, model.target_column)
    test_X, test_y = select_X_y(test_df, model.target_column)

    logger.info(f"Train: {train_X.shape}, Tune: {tune_X.shape}, Test: {test_X.shape}")

    # Configure model and preprocessing
    estimator, param_grid = model.configure_model(algorithm)

    # Determine columns to preserve through preprocessing
    preserve_columns = ["year_published"]
    if model.data_config.requires_complexity_predictions:
        preserve_columns.append("predicted_complexity")

    preprocessor = create_preprocessing_pipeline(
        model_type=args.preprocessor_type,
        model_name=algorithm,
        preserve_columns=preserve_columns,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    # Calculate sample weights if requested
    sample_weights = None
    if args.use_sample_weights:
        weight_column = args.sample_weight_column
        if weight_column is None:
            # Default weight columns
            if model.model_type == "complexity":
                weight_column = "num_weights"
            else:
                weight_column = "users_rated"

        sample_weights = calculate_sample_weights(train_df, weight_column=weight_column)
        logger.info(f"Using sample weights from '{weight_column}'")

    # Determine metric
    metric = args.metric
    if metric is None:
        metric = "log_loss" if model.model_task == "classification" else "rmse"

    # Tune model
    logger.info(f"Tuning with metric: {metric}")
    tuned_pipeline, best_params = tune_model(
        pipeline=pipeline,
        train_X=train_X,
        train_y=train_y,
        tune_X=tune_X,
        tune_y=tune_y,
        param_grid=param_grid,
        metric=metric,
        patience=args.patience,
        sample_weights=sample_weights,
    )

    # Fit on train only for evaluation
    train_pipeline = clone(tuned_pipeline).fit(train_X, train_y)
    train_metrics = evaluate_model(train_pipeline, train_X, train_y, "training")

    # Model-specific post-tuning steps (e.g., threshold optimization for hurdle)
    if hasattr(model, "find_optimal_threshold") and model.model_task == "classification":
        tune_pred_proba = train_pipeline.predict_proba(tune_X)[:, 1]
        threshold_results = model.find_optimal_threshold(tune_y, tune_pred_proba)
        logger.info(f"Optimal threshold: {threshold_results['threshold']:.4f}")

    # Evaluate on tune set
    tune_metrics = evaluate_model(train_pipeline, tune_X, tune_y, "tuning")

    # Fit final model on train + tune
    logger.info("Fitting final model on combined train + tune data")
    X_combined = pd.concat([train_X, tune_X])
    y_combined = pd.concat([train_y, tune_y])

    if args.use_sample_weights:
        combined_weights = calculate_sample_weights(
            pd.concat([train_df.to_pandas(), tune_df.to_pandas()]),
            weight_column=weight_column,
        )
        final_pipeline = clone(tuned_pipeline).fit(
            X_combined,
            y_combined,
            model__sample_weight=np.asarray(combined_weights),
        )
    else:
        final_pipeline = clone(tuned_pipeline).fit(X_combined, y_combined)

    # Evaluate on test set
    test_metrics = evaluate_model(final_pipeline, test_X, test_y, "test")

    # Additional model-specific metrics
    test_pred = final_pipeline.predict(test_X)
    additional_metrics = model.compute_additional_metrics(
        test_y.values, test_pred, "test"
    )
    test_metrics.update(additional_metrics)

    # Store pipeline in model
    model.pipeline = final_pipeline

    # Log experiment
    tracker = ExperimentTracker(model.model_type, args.output_dir)

    experiment_metadata = {
        "model_type": model.model_type,
        "model_task": model.model_task,
        "algorithm": algorithm,
        "target_column": model.target_column,
        "train_end_year": args.train_end_year,
        "tune_end_year": args.tune_end_year,
        "test_end_year": args.test_end_year,
        "use_embeddings": args.use_embeddings,
    }

    if hasattr(model, "optimal_threshold"):
        experiment_metadata["optimal_threshold"] = model.optimal_threshold

    if args.use_sample_weights:
        experiment_metadata["sample_weights"] = {"column": weight_column}

    experiment = tracker.create_experiment(
        name=args.experiment,
        description=args.description,
        metadata=experiment_metadata,
    )

    log_experiment(
        experiment=experiment,
        pipeline=final_pipeline,
        train_metrics=train_metrics,
        tune_metrics=tune_metrics,
        test_metrics=test_metrics,
        best_params=best_params,
        args=args,
        train_df=train_df,
        tune_df=tune_df,
        test_df=test_df,
        train_X=train_X,
        tune_X=tune_X,
        test_X=test_X,
        train_y=train_y,
        tune_y=tune_y,
        test_y=test_y,
        model_type=model.model_task,
    )

    logger.info("Training complete!")

    return model


def parse_finalize_arguments() -> argparse.Namespace:
    """Parse command-line arguments for finalization."""
    parser = argparse.ArgumentParser(description="Finalize a trained model for production")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["hurdle", "complexity", "rating", "users_rated"],
        help="Model type to finalize",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name to finalize",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Optional specific experiment version",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year for training data (default: current year - 2)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional description for finalized model",
    )
    parser.add_argument(
        "--complexity-predictions",
        type=str,
        default=None,
        help="Path to complexity predictions parquet (for rating/users_rated)",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        default=None,
        help="Include text embeddings as features (default: read from experiment)",
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to local parquet file (instead of BigQuery)",
    )
    parser.add_argument(
        "--recent-year-threshold",
        type=int,
        default=2,
        help="Years to exclude from current year when end_year not specified",
    )

    return parser.parse_args()


def finalize_model(
    model_type: str,
    experiment_name: str,
    version: Optional[int] = None,
    end_year: Optional[int] = None,
    description: Optional[str] = None,
    complexity_predictions_path: Optional[str] = None,
    use_embeddings: Optional[bool] = None,
    local_data_path: Optional[str] = None,
    recent_year_threshold: int = 2,
) -> str:
    """Finalize a trained model for production use.

    Args:
        model_type: Type of model (hurdle, complexity, rating, users_rated).
        experiment_name: Name of experiment to finalize.
        version: Optional specific version.
        end_year: End year for training data.
        description: Optional description.
        complexity_predictions_path: Path to complexity predictions.
        use_embeddings: Whether to include embeddings.
        local_data_path: Optional local data path.
        recent_year_threshold: Years to exclude from current.

    Returns:
        Path to finalized model directory.
    """
    setup_logging()

    # Get model class and instantiate
    model_class = get_model_class(model_type)
    model = model_class()

    # Call finalize on the model instance
    finalized_dir = model.finalize(
        experiment_name=experiment_name,
        end_year=end_year,
        description=description,
        complexity_predictions_path=complexity_predictions_path,
        use_embeddings=use_embeddings,
        local_data_path=local_data_path,
        recent_year_threshold=recent_year_threshold,
        version=version,
    )

    return str(finalized_dir)


def main():
    """Main entry point for unified training."""
    args = parse_arguments()
    model_class = get_model_class(args.model)
    train_model(model_class, args)


def main_finalize():
    """Entry point for finalization."""
    args = parse_finalize_arguments()
    finalize_model(
        model_type=args.model,
        experiment_name=args.experiment,
        version=args.version,
        end_year=args.end_year,
        description=args.description,
        complexity_predictions_path=args.complexity_predictions,
        use_embeddings=args.use_embeddings,
        local_data_path=args.local_data,
        recent_year_threshold=args.recent_year_threshold,
    )


if __name__ == "__main__":
    main()
