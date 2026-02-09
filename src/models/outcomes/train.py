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
from src.utils.config import load_config


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
    from src.models.outcomes.geek_rating import GeekRatingModel

    register_model(HurdleModel)
    register_model(ComplexityModel)
    register_model(RatingModel)
    register_model(UsersRatedModel)
    register_model(GeekRatingModel)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train an outcome prediction model")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["hurdle", "complexity", "rating", "users_rated", "geek_rating"],
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
        default=None,
        help="Experiment name (default: from config)",
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

    # Year splits (all boundaries are inclusive)
    # Defaults are None - will be loaded from config.yaml
    parser.add_argument("--train-through", type=int, default=None)
    parser.add_argument("--tune-start", type=int, default=None)
    parser.add_argument("--tune-through", type=int, default=None)
    parser.add_argument("--test-start", type=int, default=None)
    parser.add_argument("--test-through", type=int, default=None)

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
    parser.add_argument(
        "--finalize",
        action="store_true",
        default=False,
        help="Finalize model for production after training",
    )

    args = parser.parse_args()

    # Load defaults from config if not provided
    config = load_config()
    model_config = config.models.get(args.model)

    if args.experiment is None:
        if model_config is None:
            raise ValueError(
                f"--experiment not provided and no config found for model '{args.model}'"
            )
        args.experiment = model_config.experiment_name

    if args.algorithm is None and model_config is not None:
        args.algorithm = model_config.type

    # Load use_embeddings from config if not explicitly set via CLI
    if not args.use_embeddings and model_config is not None:
        args.use_embeddings = model_config.use_embeddings

    # Load use_sample_weights from config if not explicitly set via CLI
    if not args.use_sample_weights and model_config is not None:
        args.use_sample_weights = model_config.use_sample_weights

    # Load min_ratings from config
    if model_config is not None:
        args.min_ratings = model_config.min_ratings

    # Load include_count_features from config (default False)
    args.include_count_features = getattr(model_config, "include_count_features", False) if model_config else False

    # Load algorithm_params from config
    args.algorithm_params = model_config.get_algorithm_params() if model_config else {}

    # Load geek_rating-specific config (mode, sub_model_experiments)
    if args.model == "geek_rating":
        args.mode = getattr(model_config, "mode", None) or "stacking"
        args.include_predictions = getattr(model_config, "include_predictions", True)

        # Build sub_model_experiments from other model configs
        args.sub_model_experiments = {
            "hurdle": getattr(config.models.get("hurdle"), "experiment_name", None),
            "complexity": getattr(config.models.get("complexity"), "experiment_name", None),
            "rating": getattr(config.models.get("rating"), "experiment_name", None),
            "users_rated": getattr(config.models.get("users_rated"), "experiment_name", None),
        }

    # Load complexity predictions path from config if not provided via CLI
    # Models like rating/users_rated need this to get complexity predictions
    # Path is: {predictions_dir}/{complexity.experiment_name}.parquet
    if args.complexity_predictions is None:
        complexity_config = config.models.get("complexity")
        if complexity_config:
            from pathlib import Path
            args.complexity_predictions = str(
                Path(config.predictions_dir) / f"{complexity_config.experiment_name}.parquet"
            )

    # Load year splits from config if not explicitly set via CLI
    years_config = config.years.training
    if args.train_through is None:
        args.train_through = years_config.train_through
    if args.tune_start is None:
        args.tune_start = years_config.tune_start
    if args.tune_through is None:
        args.tune_through = years_config.tune_through
    if args.test_start is None:
        args.test_start = years_config.test_start
    if args.test_through is None:
        args.test_through = years_config.test_through

    # Validate year ranges
    if args.tune_start <= args.train_through:
        raise ValueError(
            f"tune_start ({args.tune_start}) must be greater than "
            f"train_through ({args.train_through})"
        )

    if not (
        args.tune_start
        <= args.tune_through
        < args.test_start
        <= args.test_through
    ):
        raise ValueError(
            "Invalid year ranges. Must satisfy: "
            "tune_start <= tune_through < test_start <= test_through"
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

    # Instantiate model with any model-specific kwargs from args
    model_kwargs = {}
    if hasattr(args, "mode"):
        model_kwargs["mode"] = args.mode
    if hasattr(args, "min_ratings"):
        model_kwargs["min_ratings"] = args.min_ratings
    if hasattr(args, "include_predictions"):
        model_kwargs["include_predictions"] = args.include_predictions

    model = model_class(**model_kwargs)

    # Determine algorithm
    algorithm = args.algorithm
    if algorithm is None:
        # Could load from config here
        algorithm = "ridge" if model.model_task == "regression" else "lightgbm"

    logger.info(f"Training {model.model_type} model with {algorithm}")

    # Load data
    df = load_training_data(
        data_config=model.data_config,
        end_year=args.test_through,
        use_embeddings=args.use_embeddings,
        complexity_predictions_path=args.complexity_predictions,
        local_data_path=args.local_data,
    )

    # Create splits
    train_df, tune_df, test_df = create_data_splits(
        df,
        train_through=args.train_through,
        tune_start=args.tune_start,
        tune_through=args.tune_through,
        test_start=args.test_start,
        test_through=args.test_through,
    )

    # Extract X, y
    train_X, train_y = select_X_y(train_df, model.target_column)
    tune_X, tune_y = select_X_y(tune_df, model.target_column)
    test_X, test_y = select_X_y(test_df, model.target_column)

    # Store original tune_X before prepare_features (needed for refit filtering)
    tune_X_original = tune_X.copy()

    # Allow models to prepare features (e.g., run sub-models for stacking)
    train_X, train_y = model.prepare_features(train_X, train_y, "train", args)
    tune_X, tune_y = model.prepare_features(tune_X, tune_y, "tune", args)
    test_X, test_y = model.prepare_features(test_X, test_y, "test", args)

    # Filter polars DataFrames to match filtered X/y (for experiment logging)
    # This handles cases where prepare_features filters rows (e.g., geek_rating hurdle filter)
    if len(train_X) < len(train_df):
        train_df = train_df[train_X.index.tolist()]
    if len(tune_X) < len(tune_df):
        tune_df = tune_df[tune_X.index.tolist()]
    if len(test_X) < len(test_df):
        test_df = test_df[test_X.index.tolist()]

    logger.info(f"Train: {train_X.shape}, Tune: {tune_X.shape}, Test: {test_X.shape}")

    # Configure model and preprocessing
    algorithm_params = getattr(args, "algorithm_params", {})
    if algorithm_params:
        logger.info(f"Using algorithm params from config: {algorithm_params}")
    estimator, param_grid = model.configure_model(algorithm, algorithm_params)

    # Determine columns to preserve through preprocessing
    preserve_columns = ["year_published"]
    if model.data_config.requires_complexity_predictions:
        preserve_columns.append("predicted_complexity")

    # For geek_rating in direct mode, preserve sub-model predictions
    if args.model == "geek_rating" and getattr(args, "mode", None) == "direct":
        preserve_columns.append("predicted_complexity")
        if getattr(args, "include_predictions", True):
            preserve_columns.extend(["predicted_rating", "predicted_users_rated_log"])

    preprocessor = create_preprocessing_pipeline(
        model_type=args.preprocessor_type,
        model_name=algorithm,
        preserve_columns=preserve_columns,
        include_description_embeddings=args.use_embeddings,
        include_count_features=args.include_count_features,
    )

    # Allow models to customize pipeline construction
    pipeline = model.create_pipeline(estimator, preprocessor, algorithm, args)

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

    # For geek_rating, filter tune data for refit (same criteria as training)
    if hasattr(model, "filter_for_refit"):
        tune_X_refit, tune_y_refit = model.filter_for_refit(tune_X, tune_y, tune_X_original)
    else:
        tune_X_refit, tune_y_refit = tune_X, tune_y

    X_combined = pd.concat([train_X, tune_X_refit])
    y_combined = pd.concat([train_y, tune_y_refit])

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
        "algorithm_params": algorithm_params if algorithm_params else None,
        "target_column": model.target_column,
        "train_through": args.train_through,
        "tune_through": args.tune_through,
        "test_through": args.test_through,
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

    # Save coefficient estimates and plot for Bayesian models
    if model.supports_coefficient_uncertainty:
        logger.info("Saving coefficient estimates and plot for Bayesian model")
        exp_dir = experiment.exp_dir

        # Save coefficient estimates with uncertainty to CSV (replaces coefficients.csv)
        coef_path = exp_dir / "coefficients.csv"
        model.save_coefficient_estimates(coef_path)

        # Generate and save coefficient plot in plots subdirectory
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / "coefficient_plot.png"
        model.plot_top_coefficients(plot_path, top_n=100)

    # Generate residuals diagnostic plot for regression models
    if model.model_task == "regression":
        logger.info("Generating residuals diagnostic plot")
        exp_dir = experiment.exp_dir
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        residuals_path = plots_dir / "residuals_plot.png"
        model.plot_residuals(
            residuals_path,
            y_true=test_y.values,
            y_pred=test_pred,
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
        choices=["hurdle", "complexity", "rating", "users_rated", "geek_rating"],
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

    # Finalize if requested
    if args.finalize:
        logger.info("Finalizing model for production...")
        finalize_model(
            model_type=args.model,
            experiment_name=args.experiment,
            use_embeddings=args.use_embeddings,
            local_data_path=args.local_data,
            complexity_predictions_path=args.complexity_predictions,
        )


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
