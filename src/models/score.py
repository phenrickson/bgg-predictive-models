"""Score new data using the finalized models"""

import argparse
import json
import numpy as np
import os
import polars as pl
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from src.utils.logging import setup_logging
from src.models.experiments import ExperimentTracker
from src.models.outcomes.train import get_model_class
from src.models.outcomes.data import load_scoring_data as _load_scoring_data

logger = setup_logging()


def get_model_info(finalized_dir: Path) -> dict:
    """Extract metadata from finalized model directory.

    Args:
        finalized_dir: Path to finalized model directory

    Returns:
        Dictionary with model info including:
        - final_end_year: Training cutoff year
        - use_embeddings: Whether model was trained with embeddings
    """
    info_path = finalized_dir / "info.json"
    info = {}

    if info_path.exists():
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
        except Exception as e:
            logger.info(f"Warning: Error reading model info: {e}")

    return info


def get_finalized_model_info(
    experiment_name: str,
    model_type: str,
) -> dict:
    """Get metadata from the finalized model for an experiment.

    Args:
        experiment_name: Name of the experiment
        model_type: Type of model

    Returns:
        Dictionary with finalized model metadata.
    """
    tracker = ExperimentTracker(model_type)

    # Find the latest version of the experiment
    experiments = tracker.list_experiments()
    matching_experiments = [
        exp for exp in experiments if exp["name"] == experiment_name
    ]

    if not matching_experiments:
        raise ValueError(f"No experiments found matching {experiment_name}")

    latest_experiment = max(matching_experiments, key=lambda x: x["version"])

    # Load the experiment
    experiment = tracker.load_experiment(
        latest_experiment["name"], latest_experiment["version"]
    )

    # Get finalized model info
    finalized_dir = experiment.exp_dir / "finalized"
    info = get_model_info(finalized_dir)

    # Also include experiment metadata for use_embeddings
    info["use_embeddings"] = experiment.metadata.get("use_embeddings", False)

    return info


def extract_threshold(experiment_name: str, model_type: str) -> Optional[float]:
    """Extract threshold from the most recent version's metadata or model_info.json file.

    Args:
        experiment_name: Name of the experiment
        model_type: Type of model

    Returns:
        Threshold value if found, None otherwise
    """
    from src.models.experiments import ExperimentTracker

    # Get experiment tracker for the model type
    tracker = ExperimentTracker(model_type)

    # Find the latest version of the experiment
    experiments = tracker.list_experiments()
    matching_experiments = [
        exp for exp in experiments if exp["name"] == experiment_name
    ]

    if not matching_experiments:
        logger.info(f"No experiments found matching {experiment_name}")
        return None

    # Get the latest version
    latest_experiment = max(matching_experiments, key=lambda x: x["version"])

    # Load the experiment with the latest version
    experiment = tracker.load_experiment(
        latest_experiment["name"], latest_experiment["version"]
    )

    # First, check metadata.json for optimal_threshold
    metadata_path = experiment.exp_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                threshold = metadata.get("metadata", {}).get("optimal_threshold")

                if threshold is not None:
                    logger.info(f"Found threshold {threshold} in {metadata_path}")
                    return threshold
        except Exception as e:
            logger.info(f"Warning: Error reading {metadata_path}: {e}")

    # Then, look for model_info.json in the experiment directory
    model_info_path = experiment.exp_dir / "model_info.json"

    if model_info_path.exists():
        try:
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
                threshold = model_info.get("threshold")

                if threshold is not None:
                    logger.info(f"Found threshold {threshold} in {model_info_path}")
                    return threshold
        except Exception as e:
            logger.info(f"Warning: Error reading {model_info_path}: {e}")

    # If no threshold found
    logger.info("No threshold found in metadata.json or model_info.json")
    return None


def get_known_model_types() -> list:
    """Get list of known model types from the model registry.

    Returns:
        List of model type strings.
    """
    # Trigger registry population
    try:
        get_model_class("hurdle")
    except ValueError:
        pass

    from src.models.outcomes.train import MODEL_REGISTRY

    return list(MODEL_REGISTRY.keys())


def load_model(experiment_name: str, model_type: Optional[str] = None):
    """Load the finalized model and preprocessing pipeline.

    Attempts to load the experiment by extracting the model type from the experiment name.
    Supports experiments in different model type directories and more flexible path handling.

    Args:
        experiment_name: Name of the experiment to load
        model_type: Optional model type to restrict the search

    Returns:
        Finalized pipeline
    """
    # Determine model types to search
    if model_type:
        # If model_type is provided, only search in that type
        model_types = [model_type]
    else:
        # Otherwise, search in all known model types from registry
        model_types = get_known_model_types()


    # Try each model type until successful
    for current_model_type in model_types:
        try:
            tracker = ExperimentTracker(current_model_type)
            experiments = tracker.list_experiments()

            # Handle cases with or without version
            if "/" in experiment_name:
                # If version is specified (e.g., 'hurdle_model/v1')
                base_name, version_str = experiment_name.split("/")
                version = int(version_str[1:])
                experiment = tracker.load_experiment(base_name, version)
            else:
                # If no version specified, find matching experiments
                matching_experiments = [
                    exp for exp in experiments if exp["name"] == experiment_name
                ]

                if not matching_experiments:
                    continue

                # Sort and get the latest version
                latest_experiment = max(
                    matching_experiments, key=lambda x: x["version"]
                )

                experiment = tracker.load_experiment(
                    latest_experiment["name"], latest_experiment["version"]
                )

            # Explicitly look for finalized model
            finalized_path = experiment.exp_dir / "finalized" / "pipeline.pkl"

            if not finalized_path.exists():
                # Look for latest version's finalized model
                version_dirs = [
                    d
                    for d in experiment.exp_dir.iterdir()
                    if d.is_dir() and d.name.startswith("v")
                ]

                if version_dirs:
                    latest_version_dir = max(
                        version_dirs, key=lambda x: int(x.name[1:])
                    )
                    finalized_path = latest_version_dir / "finalized" / "pipeline.pkl"

            if finalized_path.exists():
                logger.info(f"Loaded {experiment_name} from {finalized_path}")
                import joblib

                return joblib.load(finalized_path)

            raise FileNotFoundError(f"No finalized model found for {experiment_name}")

        except (ValueError, FileNotFoundError, Exception):
            continue

    # If no model type works, raise an error
    raise ValueError(
        f"Could not load experiment '{experiment_name}' in any known model type"
    )


def load_scoring_data(
    data_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    model_type: str = "hurdle",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    complexity_predictions: Optional[pl.DataFrame] = None,
    use_embeddings: Optional[bool] = None,
) -> pl.DataFrame:
    """
    Load data for scoring based on provided parameters.

    Uses the centralized data loader from src.models.outcomes.data to ensure
    consistent feature loading between training and scoring.

    Args:
        data_path: Optional path to a CSV file for scoring
        experiment_name: Name of experiment to determine data loading parameters
        model_type: Type of model being used
        start_year: First year of data to include
        end_year: Last year of data to include
        complexity_predictions: Optional DataFrame with pre-computed complexity predictions
            (alternative to complexity_predictions_path)
        use_embeddings: Whether to load embeddings. If None, reads from finalized model info.

    Returns:
        Polars DataFrame with data to be scored
    """
    # If data path is provided, load directly from CSV
    if data_path:
        return pl.read_csv(data_path)

    # Get model class for data_config
    model_class = get_model_class(model_type)
    model = model_class()
    data_config = model.data_config

    # Get finalized model info to determine start_year and use_embeddings
    model_info = get_finalized_model_info(experiment_name, model_type)
    final_end_year = model_info.get("final_end_year")

    # Determine start year for scoring (year after training cutoff)
    if start_year is None:
        start_year = final_end_year + 1 if final_end_year else 0

    # Default end year to far future
    if end_year is None:
        end_year = datetime.now().year + 5

    # Determine whether to use embeddings from model info if not specified
    if use_embeddings is None:
        use_embeddings = model_info.get("use_embeddings", False)

    logger.info(f"Loading scoring data: years {start_year}-{end_year}")
    logger.info(f"Use embeddings: {use_embeddings}")

    # Handle complexity predictions - can be passed as DataFrame or path
    complexity_predictions_path = None
    if complexity_predictions is not None:
        # If passed as DataFrame, we'll join manually after loading
        pass
    elif data_config.requires_complexity_predictions:
        # Auto-load complexity predictions from config
        from src.utils.config import load_config
        config = load_config()
        complexity_config = config.models.get("complexity")
        if complexity_config and hasattr(complexity_config, "experiment_name"):
            # Look for predictions in the predictions directory
            predictions_dir = Path(config.predictions_dir)
            complexity_predictions_path = predictions_dir / f"{complexity_config.experiment_name}.parquet"
            if complexity_predictions_path.exists():
                logger.info(f"Auto-loading complexity predictions from {complexity_predictions_path}")
            else:
                logger.warning(
                    f"Complexity predictions not found at {complexity_predictions_path}. "
                    "Run score_complexity first or provide --complexity-predictions path."
                )
                complexity_predictions_path = None

    # Use centralized data loader
    df = _load_scoring_data(
        data_config=data_config,
        start_year=start_year,
        end_year=end_year,
        use_embeddings=use_embeddings,
        complexity_predictions_path=str(complexity_predictions_path) if complexity_predictions_path else None,
        local_data_path=None,
    )

    # If complexity predictions provided as DataFrame, join them manually
    if complexity_predictions is not None and data_config.requires_complexity_predictions:
        logger.info("Joining pre-computed complexity predictions from DataFrame")
        df = df.join(
            complexity_predictions.select(["game_id", "predicted_complexity"]),
            on="game_id",
            how="inner",
        )

        if len(df) == 0:
            raise ValueError(
                "No games remain after joining with complexity predictions"
            )

    return df


def predict_data(
    pipeline,
    df: pl.DataFrame,
    experiment_name: str,
    model_type: str = "hurdle",
    complexity_predictions: Optional[pl.DataFrame] = None,
    return_uncertainty: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """
    Predict data using the given pipeline and model type.

    Uses model classes from src.models.outcomes to handle model-specific
    prediction logic and post-processing.

    Args:
        pipeline: Trained model pipeline
        df: Input DataFrame to predict
        experiment_name: Name of experiment for threshold retrieval
        model_type: Type of model being used
        complexity_predictions: Unused, kept for backward compatibility
        return_uncertainty: If True, return uncertainty estimates for Bayesian models

    Returns:
        Tuple of (predicted_values, predicted_class, threshold, std_dev)
        std_dev is None for non-Bayesian models or when return_uncertainty=False
    """
    # Get the model class for this type
    model_class = get_model_class(model_type)
    model = model_class()
    model.pipeline = pipeline

    df_pandas = df.to_pandas()

    logger.info(f"{model_type.capitalize()} Model Prediction:")
    logger.info(f"Input DataFrame shape: {df.shape}")

    std_dev = None

    # Handle classification vs regression
    if model.model_task == "classification":
        # For hurdle model, get probabilities
        raw_predictions = model.predict_proba(df_pandas)[:, 1]

        # Get threshold from experiment metadata
        threshold = extract_threshold(experiment_name, model_type)
        threshold = threshold if threshold is not None else 0.5
        model.optimal_threshold = threshold

        logger.info(f"Using classification threshold: {threshold}")

        predicted_class = raw_predictions >= threshold
        predicted_values = raw_predictions
    else:
        # For regression models, check for uncertainty support
        if return_uncertainty and model.supports_uncertainty:
            result = model.predict_with_uncertainty(df_pandas)
            predicted_values = result.values
            std_dev = result.std

            logger.info(
                f"Prediction stats: min={predicted_values.min():.4f}, "
                f"max={predicted_values.max():.4f}, mean={predicted_values.mean():.4f}"
            )
            if std_dev is not None:
                logger.info(
                    f"Uncertainty stats: min_std={std_dev.min():.4f}, "
                    f"max_std={std_dev.max():.4f}, mean_std={std_dev.mean():.4f}"
                )
        else:
            # Standard prediction path
            raw_predictions = pipeline.predict(df_pandas)

            logger.info(
                f"Raw prediction stats: min={raw_predictions.min():.4f}, "
                f"max={raw_predictions.max():.4f}, mean={raw_predictions.mean():.4f}"
            )

            # Use model's post-processing (clipping, inverse transforms, etc.)
            predicted_values = model.post_process_predictions(raw_predictions)

            logger.info(
                f"Post-processed stats: min={predicted_values.min():.4f}, "
                f"max={predicted_values.max():.4f}, mean={predicted_values.mean():.4f}"
            )

        predicted_class = None
        threshold = None

    return predicted_values, predicted_class, threshold, std_dev


def compute_confidence_bounds(
    values: np.ndarray,
    std: np.ndarray,
    level: float,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute confidence interval bounds.

    Args:
        values: Point estimates.
        std: Standard deviations.
        level: Confidence level (e.g., 0.95 for 95% CI).
        lower_bound: Optional lower clipping bound.
        upper_bound: Optional upper clipping bound.

    Returns:
        Tuple of (lower, upper) bounds.
    """
    from scipy import stats

    z = stats.norm.ppf((1 + level) / 2)
    lower = values - z * std
    upper = values + z * std

    if lower_bound is not None:
        lower = np.maximum(lower, lower_bound)
    if upper_bound is not None:
        upper = np.minimum(upper, upper_bound)

    return lower, upper


def prepare_results(
    df: pl.DataFrame,
    predicted_values: np.ndarray,
    predicted_class: Optional[np.ndarray],
    model_type: str,
    threshold: Optional[float] = None,
    std_dev: Optional[np.ndarray] = None,
    confidence_levels: Optional[List[float]] = None,
) -> pl.DataFrame:
    """
    Prepare results DataFrame based on model type.

    Args:
        df: Original input DataFrame
        predicted_values: Predicted values
        predicted_class: Predicted classes (for classification models)
        model_type: Type of model being used
        threshold: Optional threshold used for classification
        std_dev: Optional standard deviation for uncertainty quantification
        confidence_levels: List of confidence levels for intervals (default: [0.80, 0.90, 0.95])

    Returns:
        Results DataFrame with predictions
    """
    if confidence_levels is None:
        confidence_levels = [0.80, 0.90, 0.95]

    # Model-specific bounds for clipping
    BOUNDS = {
        "complexity": (1, 5),
        "rating": (1, 10),
        "users_rated": (0, None),
    }

    # Select columns based on model type
    if model_type == "complexity":
        columns = [
            pl.Series("predicted_complexity", predicted_values),
            pl.Series("complexity", df.select("complexity").to_pandas().squeeze()),
        ]
        if std_dev is not None:
            columns.append(pl.Series("predicted_complexity_std", std_dev))
            lb, ub = BOUNDS["complexity"]
            for level in confidence_levels:
                level_pct = int(level * 100)
                lower, upper = compute_confidence_bounds(predicted_values, std_dev, level, lb, ub)
                columns.extend([
                    pl.Series(f"predicted_complexity_lower_{level_pct}", lower),
                    pl.Series(f"predicted_complexity_upper_{level_pct}", upper),
                ])
        results = df.select(["game_id", "name", "year_published"]).with_columns(columns)
    elif model_type == "rating":
        columns = [
            pl.Series("predicted_rating", predicted_values),
            pl.Series("rating", df.select("rating").to_pandas().squeeze()),
        ]
        if std_dev is not None:
            columns.append(pl.Series("predicted_rating_std", std_dev))
            lb, ub = BOUNDS["rating"]
            for level in confidence_levels:
                level_pct = int(level * 100)
                lower, upper = compute_confidence_bounds(predicted_values, std_dev, level, lb, ub)
                columns.extend([
                    pl.Series(f"predicted_rating_lower_{level_pct}", lower),
                    pl.Series(f"predicted_rating_upper_{level_pct}", upper),
                ])
        results = df.select(["game_id", "name", "year_published"]).with_columns(columns)
    elif model_type == "users_rated":
        columns = [
            pl.Series("predicted_users_rated", predicted_values),
            pl.Series(
                "users_rated", df.select("users_rated").to_pandas().squeeze()
            ),
        ]
        if std_dev is not None:
            columns.append(pl.Series("predicted_users_rated_std", std_dev))
            lb, ub = BOUNDS["users_rated"]
            for level in confidence_levels:
                level_pct = int(level * 100)
                lower, upper = compute_confidence_bounds(predicted_values, std_dev, level, lb, ub)
                columns.extend([
                    pl.Series(f"predicted_users_rated_lower_{level_pct}", lower),
                    pl.Series(f"predicted_users_rated_upper_{level_pct}", upper),
                ])
        results = df.select(["game_id", "name", "year_published"]).with_columns(columns)
    elif model_type == "hurdle":
        # Existing logic for hurdle model
        results = df.select(["game_id", "name", "year_published"]).with_columns(
            [
                pl.Series("predicted_prob", predicted_values),
                pl.Series("predicted_class", predicted_class),
                pl.Series("hurdle", df.select("hurdle").to_pandas().squeeze()),
                pl.Series(
                    "threshold",
                    (
                        [threshold] * len(df)
                        if threshold is not None
                        else [None] * len(df)
                    ),
                ),  # Add threshold used
            ]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return results


def save_and_display_results(
    results: pl.DataFrame,
    experiment_name: str,
    start_year: Optional[int],
    end_year: Optional[int],
    model_type: str,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Save and display prediction results.

    Args:
        results: Results DataFrame with predictions
        experiment_name: Name of experiment
        start_year: Start year of data
        end_year: End year of data
        model_type: Type of model being used
        output_path: Optional path to save predictions

    Returns:
        Results DataFrame
    """
    # Determine output path
    from pathlib import Path

    # Create base predictions directory for the model type
    base_predictions_dir = Path("data/predictions") / model_type
    base_predictions_dir.mkdir(parents=True, exist_ok=True)

    # Determine output path if not provided
    if output_path is None:
        # Create a filename that includes just the experiment name
        output_filename = f"{experiment_name}_predictions.parquet"
        output_path = base_predictions_dir / output_filename
    else:
        # Ensure output path is within the predictions directory
        output_path = Path(output_path)
        if not output_path.suffix == ".parquet":
            output_path = output_path.with_suffix(".parquet")

        # Ensure it's in the correct model type directory
        output_path = base_predictions_dir / output_path.name

    # Save results locally
    results.write_parquet(str(output_path))
    logger.info(f"Predictions for {experiment_name} saved to {output_path}")

    # For complexity model, also save to predictions_dir for downstream models
    if model_type == "complexity":
        from src.utils.config import load_config
        config = load_config()
        predictions_dir = Path(config.predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        downstream_path = predictions_dir / f"{experiment_name}.parquet"
        results.write_parquet(str(downstream_path))
        logger.info(f"Complexity predictions also saved to {downstream_path}")

    # Also save to GCS if bucket name is configured
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if bucket_name:
        try:
            from google.cloud import storage

            # Create GCS client and upload
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # Create GCS path
            gcs_blob_path = f"predictions/{model_type}/{output_path.name}"
            blob = bucket.blob(gcs_blob_path)

            # Upload the file
            blob.upload_from_filename(str(output_path))
            gcs_path = f"gs://{bucket_name}/{gcs_blob_path}"
            logger.info(f"Predictions also saved to GCS: {gcs_path}")

        except Exception as e:
            logger.warning(f"Failed to save to GCS: {e}")

    logger.info(f"Data loaded from year {start_year or 'beginning'} to {end_year}")

    # Display sample of results
    logger.info("\nSample predictions:")

    # Dynamically select sample columns based on model type
    if model_type == "complexity":
        sample_columns = [
            "game_id",
            "name",
            "year_published",
            "predicted_complexity",
            "complexity",
        ]
    elif model_type == "rating":
        sample_columns = [
            "game_id",
            "name",
            "year_published",
            "predicted_rating",
            "rating",
        ]
    elif model_type == "hurdle":
        sample_columns = [
            "game_id",
            "name",
            "year_published",
            "predicted_prob",
            "predicted_class",
            "hurdle",
            "threshold",
        ]
    elif model_type == "users_rated":
        sample_columns = [
            "game_id",
            "name",
            "year_published",
            "predicted_users_rated",
            "users_rated",
        ]
    else:
        # Fallback to default columns
        sample_columns = [
            "game_id",
            "name",
            "year_published",
            "predicted",
            "predicted_class",
        ]

    logger.info(results.select(sample_columns).head())

    return results


def score_data(
    experiment_name: str = None,
    data_path: str = None,
    start_year: int = None,
    end_year: int = None,
    min_ratings: int = 0,
    output_path: str = None,
    model_type: str = "hurdle",
    complexity_predictions: Optional[pl.DataFrame] = None,
    return_uncertainty: bool = False,
):
    """Score data using the finalized model.

    Args:
        experiment_name: Name of experiment with finalized model (optional, will use latest if not provided)
        data_path: Optional path to a CSV file for scoring (overrides query-based loading)
        start_year: First year of data to include (optional)
        end_year: Last year of data to include (optional)
        min_ratings: Minimum number of ratings to filter games (default 0)
        output_path: Path to save predictions (optional, will use experiment name if not provided)
        model_type: Type of model to use (default: hurdle)
        complexity_predictions: Optional pre-computed complexity predictions DataFrame
        return_uncertainty: If True, include uncertainty estimates for Bayesian models
    """
    # Handle case where model type is included in experiment name
    if experiment_name is not None and "/" in experiment_name:
        # If experiment name includes model type (e.g., 'rating/full-features')
        model_type, experiment_name = experiment_name.split("/")

    # Determine experiment name if not provided - read from config first
    if experiment_name is None:
        from src.utils.config import load_config

        config = load_config()
        model_config = config.models.get(model_type)

        if model_config and hasattr(model_config, "experiment_name") and model_config.experiment_name:
            experiment_name = model_config.experiment_name
            logger.info(f"Using experiment from config: {experiment_name}")
        else:
            # Fall back to latest experiment if not in config
            from src.models.experiments import ExperimentTracker

            tracker = ExperimentTracker(model_type)
            experiments = tracker.list_experiments()
            if not experiments:
                raise ValueError(f"No {model_type} model experiments found.")
            experiment = max(experiments, key=lambda x: x.get("version", 0))
            experiment_name = experiment["name"]
            logger.info(f"Using latest experiment (not found in config): {experiment_name}")

    # logger.info debug information about model type
    logger.info(f"Using model type: {model_type}")

    # Load pipeline
    pipeline = load_model(experiment_name)

    # Load data
    df = load_scoring_data(
        data_path=data_path,
        experiment_name=experiment_name,
        model_type=model_type,
        start_year=start_year,
        end_year=end_year,
        complexity_predictions=complexity_predictions,
    )

    # Predict data
    predicted_prob, predicted_class, threshold, std_dev = predict_data(
        pipeline,
        df,
        experiment_name,
        model_type=model_type,
        complexity_predictions=complexity_predictions,
        return_uncertainty=return_uncertainty,
    )

    # Prepare results
    results = prepare_results(
        df, predicted_prob, predicted_class, model_type, threshold, std_dev
    )

    # Save and display results
    return save_and_display_results(
        results, experiment_name, start_year, end_year, model_type, output_path
    )


def main():
    parser = argparse.ArgumentParser(description="Score new data using finalized model")
    parser.add_argument(
        "--data", help="Optional path to CSV file containing data to score"
    )
    parser.add_argument(
        "--experiment",
        help="Name of experiment with finalized model. Can include model type (e.g., 'rating/full-features')",
    )
    parser.add_argument(
        "--model",
        default="hurdle",
        help="Model type (hurdle, complexity, rating, users_rated)",
    )
    parser.add_argument("--start-year", type=int, help="First year of data to include")
    parser.add_argument("--end-year", type=int, help="Last year of data to include")
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Score all data regardless of year (ignores start-year/end-year defaults)",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=0,
        help="Minimum number of ratings to filter games",
    )
    parser.add_argument("--output", help="Path to save predictions")
    parser.add_argument(
        "--complexity-predictions",
        help="Path to parquet file with pre-computed complexity predictions",
    )
    parser.add_argument(
        "--with-uncertainty",
        action="store_true",
        default=False,
        help="Include uncertainty estimates in output (for Bayesian models)",
    )

    args = parser.parse_args()

    # If experiment includes model type, override model-type argument
    if args.experiment and "/" in args.experiment:
        model_type, experiment_name = args.experiment.split("/")
    else:
        model_type = args.model
        experiment_name = args.experiment

    # Load complexity predictions if provided
    complexity_predictions = None
    if args.complexity_predictions:
        complexity_predictions = pl.read_parquet(args.complexity_predictions)

    # Handle --all-years flag
    start_year = args.start_year
    end_year = args.end_year
    if args.all_years:
        # Use sentinel values that will be interpreted as "no filter"
        start_year = -9999  # Before any game
        end_year = 9999     # Far future

    score_data(
        data_path=args.data,
        experiment_name=experiment_name,
        start_year=start_year,
        end_year=end_year,
        min_ratings=args.min_ratings,
        output_path=args.output,
        model_type=model_type,
        complexity_predictions=complexity_predictions,
        return_uncertainty=args.with_uncertainty,
    )


if __name__ == "__main__":
    main()
