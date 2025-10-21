"""Finalize a model for production by fitting on full dataset."""

import argparse
import logging
from typing import Optional
from datetime import datetime

import numpy as np
import polars as pl

from src.models.experiments import ExperimentTracker
from src.models.experiments import Experiment
from src.utils.config import load_config
from src.data.loader import BGGDataLoader
from src.models.hurdle import setup_logging
from src.models.training import calculate_sample_weights  # Import the function


def extract_model_threshold(experiment: Experiment) -> Optional[float]:
    """
    Safely extract the model threshold from experiment metadata.

    Args:
        experiment: ExperimentTracker experiment object

    Returns:
        float: Threshold value if found, None otherwise
    """
    try:
        # Check multiple possible locations for threshold
        threshold_paths = [
            ("model_info", "threshold"),  # From log_experiment in hurdle.py
            ("threshold",),  # Direct metadata key
        ]

        for path in threshold_paths:
            current = experiment.metadata
            for key in path:
                current = current.get(key)
                if current is None:
                    break
            if current is not None:
                return float(current)

        return None
    except Exception:
        return None


def extract_min_weights(experiment: Experiment) -> Optional[float]:
    """
    Safely extract the min_weights from experiment metadata.

    Args:
        experiment: ExperimentTracker experiment object

    Returns:
        float: Min weights value if found, None otherwise
    """
    try:
        # Check multiple possible locations for min_weights
        min_weights_paths = [
            ("model_info", "min_weights"),  # From log_experiment
            ("min_weights",),  # Direct metadata key
        ]

        for path in min_weights_paths:
            current = experiment.metadata
            for key in path:
                current = current.get(key)
                if current is None:
                    break
            if current is not None:
                return float(current)

        return None
    except Exception:
        return None


def generate_model_description(
    base_description: Optional[str],
    final_end_year: int,
    threshold: Optional[float] = None,
) -> str:
    """
    Generate a model description, optionally including threshold information.

    Args:
        base_description: Optional user-provided description
        final_end_year: Year of model training
        threshold: Optional model threshold

    Returns:
        str: Descriptive string for the model
    """
    # Start with base description or create default
    description = (
        base_description or f"Production model trained on data through {final_end_year}"
    )

    # Append threshold if available
    if threshold is not None:
        description += f". Optimal classification threshold: {threshold:.4f}"

    return description


def extract_sample_weights(
    experiment: Experiment, df: Optional[pl.DataFrame] = None
) -> Optional[np.ndarray]:
    """
    Extract or calculate sample weights from experiment metadata.

    Args:
        experiment: Experiment object
        df: Optional DataFrame to calculate weights if method is specified

    Returns:
        numpy array of sample weights or None
    """
    logger = logging.getLogger(__name__)

    # Check for sample weights configuration
    sample_weights_info = experiment.metadata.get("sample_weights", {})
    weight_column = sample_weights_info.get("column")

    # If no column specified, return None
    if not weight_column:
        logger.warning("No sample weight column specified in metadata")
        return None

    # If DataFrame is provided and column exists, calculate weights
    if df is not None and weight_column in df.columns:
        logger.info(f"Calculating sample weights using column: {weight_column}")
        return calculate_sample_weights(df.to_pandas(), weight_column)

    logger.warning(f"Could not calculate sample weights for column: {weight_column}")
    return None


def load_data(
    config: dict,
    loader: BGGDataLoader,
    logger: logging.Logger,
    model_type: str = "hurdle",
    end_year: Optional[int] = None,
    min_ratings: Optional[float] = None,
    min_weights: Optional[float] = None,
    recent_year_threshold: int = 2,
    complexity_experiment: Optional[str] = None,
) -> tuple:
    """
    Load training data with specified parameters.

    Args:
        config: Configuration dictionary
        loader: BGGDataLoader instance
        model_type: Type of model being trained
        end_year: Year to filter data up to
        min_weights: Minimum weights filter
        recent_year_threshold: Number of years to consider "recent" for filtering
        logger: Logging instance
        complexity_experiment: Optional name of complexity experiment

    Returns:
        Tuple of (dataframe, final_end_year)
    """
    current_year = datetime.now().year

    # Dynamically calculate end year, excluding recent years
    if end_year is None:
        end_year = current_year - recent_year_threshold

    # Check if end_year is very recent
    years_from_current = current_year - end_year
    if years_from_current <= recent_year_threshold:
        logger.info(
            f"End year {end_year} is within {recent_year_threshold} years of current year"
        )
        logger.info(
            f"Will filter out the most recent {recent_year_threshold} years of data"
        )
        end_year = current_year - recent_year_threshold

    logger.info(f"Loading data through {end_year}")

    # Diagnostic logging for data loading parameters
    logger.info("Data Loading Parameters:")
    logger.info(f"  End Train Year: {end_year}")
    logger.info(f"  Minimum Ratings: {min_ratings}")
    logger.info(f"  Minimum Weights: {min_weights}")

    df = loader.load_training_data(
        end_train_year=end_year, min_ratings=min_ratings, min_weights=min_weights
    )

    # Load complexity predictions if experiment is specified
    if complexity_experiment:
        logger.info(
            f"Loading complexity predictions from experiment: {complexity_experiment}"
        )
        try:
            import polars as pl

            complexity_predictions_path = (
                f"models/experiments/predictions/{complexity_experiment}.parquet"
            )
            complexity_predictions = pl.read_parquet(complexity_predictions_path)

            # Join complexity predictions
            df = df.join(
                complexity_predictions.select(["game_id", "predicted_complexity"]),
                on="game_id",
                how="left",
            )

            logger.info(
                f"Joined complexity predictions. New DataFrame shape: {df.shape}"
            )
        except Exception as e:
            logger.warning(f"Could not load complexity predictions: {e}")

    # Detailed data diagnostics
    logger.info("Data Loading Diagnostics:")
    logger.info(f"  Total Rows: {len(df)}")
    logger.info(
        f"  Year Range: {df['year_published'].min()} - {df['year_published'].max()}"
    )
    # Sample row diagnostics
    # logger.info("\nSample Row Diagnostics:")
    # sample_row = df.head(1)
    # for col in sample_row.columns:
    #     logger.info(f"  {col}: {sample_row[col].to_pandas().squeeze()}")

    return df, end_year


def prepare_data(df, model_type="hurdle"):
    """
    Prepare data by splitting features and target.

    Args:
        df: Input dataframe
        model_type: Type of model ('hurdle', 'complexity', 'rating', etc.)

    Returns:
        Tuple of (X, y)
    """
    # Determine target column based on model type
    target_columns = {
        "hurdle": "hurdle",
        "complexity": "complexity",
        "rating": "rating",
        "users_rated": "log_users_rated",
    }

    # Get target column, default to 'hurdle' if not specified
    target_column = target_columns.get(model_type, "hurdle")

    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found for model type '{model_type}'"
        )

    # Get features and target (convert to pandas)
    X = df.drop(target_column).to_pandas()
    y = df.select(target_column).to_pandas().squeeze()

    return X, y


def get_model_parameters(
    experiment: Experiment, end_year: int, description: Optional[str] = None
) -> tuple:
    """
    Extract model parameters from experiment metadata.

    Args:
        experiment: Experiment object
        end_year: End year for training data
        description: Optional base description

    Returns:
        Tuple of (description, threshold, min_weights, min_ratings)
    """
    # Extract threshold if available
    threshold = extract_model_threshold(experiment)
    min_weights = extract_min_weights(experiment)

    # Extract min_ratings from experiment metadata
    min_ratings = experiment.metadata.get("min_ratings")
    if min_ratings is None:
        min_ratings = experiment.metadata.get("config", {}).get("min_ratings", 0)

    # Generate description with threshold, min_weights, and min_ratings info
    description_parts = [
        description or f"Production model trained on data through {end_year}"
    ]

    if threshold is not None:
        description_parts.append(f"Optimal classification threshold: {threshold:.4f}")

    if min_weights is not None:
        description_parts.append(f"Minimum weights filter: {min_weights}")

    if min_ratings is not None and min_ratings > 0:
        description_parts.append(f"Minimum ratings filter: {min_ratings}")

    description = ". ".join(description_parts)

    return description, threshold, min_weights, min_ratings


def finalize_model(
    model_type: str,
    experiment_name: str,
    version: Optional[int] = None,
    end_year: Optional[int] = None,
    description: Optional[str] = None,
    recent_year_threshold: int = 2,  # New parameter for filtering recent years
):
    """Finalize a model by fitting its pipeline on full dataset.

    Args:
        model_type: Type of model (e.g., 'hurdle', 'rating')
        experiment_name: Name of experiment to finalize
        version: Optional specific version to finalize
        end_year: Optional end year for training data
        description: Optional description of finalized model
        sample_weight_base: Optional base for complexity weight calculation
    """
    # Setup logging
    logger = setup_logging()

    # Load experiment
    tracker = ExperimentTracker(model_type)

    # Check if experiment name includes version/hash
    if "_v" in experiment_name:
        # Extract base name from full name
        experiment_name = experiment_name.split("_v")[0]

    experiment = tracker.load_experiment(experiment_name, version)

    # Load configuration and data loader
    config = load_config()
    loader = BGGDataLoader(config.get_bigquery_config())

    # Extract min_weights and min_ratings if available
    min_weights = extract_min_weights(experiment)

    # Extract min_ratings from experiment metadata
    min_ratings = experiment.metadata.get("min_ratings")
    if min_ratings is None:
        # Fallback to config or default
        min_ratings = experiment.metadata.get("config", {}).get("min_ratings", 0)

    logger.info(f"Using minimum ratings: {min_ratings}")

    # For models, extract complexity experiment name from metadata
    complexity_experiment = None
    if model_type in ["rating", "users_rated"]:
        # Try to extract complexity experiment from metadata
        complexity_experiment = experiment.metadata.get(
            "complexity_experiment"
        ) or experiment.metadata.get("config", {}).get("complexity_experiment")

        if not complexity_experiment:
            raise ValueError(
                "No complexity experiment found in metadata for rating model. "
                "Ensure the rating model was trained with a --complexity-experiment argument."
            )

        logger.info(f"Using complexity experiment: {complexity_experiment}")

    # Load data
    df, final_end_year = load_data(
        config=config,
        loader=loader,
        logger=logger,
        model_type=model_type,
        end_year=end_year,
        min_ratings=min_ratings,
        min_weights=min_weights,
        recent_year_threshold=recent_year_threshold,
        complexity_experiment=complexity_experiment,
    )

    # Prepare data with model-specific target column
    X, y = prepare_data(df, model_type)

    # Get model parameters
    description, threshold, min_weights, min_ratings = get_model_parameters(
        experiment=experiment, end_year=final_end_year, description=description
    )

    # Determine sample weights
    logger.info("Experiment Metadata:")
    for key, value in experiment.metadata.items():
        logger.info(f"  {key}: {value}")

    sample_weights = extract_sample_weights(experiment, df)

    # Log comprehensive sample weight diagnostics
    logger.info(f"Raw sample weights: {sample_weights}")
    logger.info(f"Type of sample weights: {type(sample_weights)}")

    if sample_weights is not None:
        try:
            # Explicitly convert to numpy array
            sample_weights_array = np.asarray(sample_weights)

            logger.info("Sample Weights Diagnostic:")
            logger.info(f"  Total sample weights: {len(sample_weights_array)}")
            logger.info(
                f"  Weight range: {sample_weights_array.min():.4f} - {sample_weights_array.max():.4f}"
            )
            logger.info(f"  Mean weight: {np.mean(sample_weights_array):.4f}")
            logger.info(f"  Median weight: {np.median(sample_weights_array):.4f}")
            logger.info(f"  Standard deviation: {np.std(sample_weights_array):.4f}")

        except Exception as e:
            logger.warning(f"Error processing sample weights: {e}")
            logger.warning(f"Raw sample weights: {sample_weights}")

    # Finalize model
    logger.info("Fitting pipeline on full dataset...")
    finalized_dir = experiment.finalize_model(
        X=X,
        y=y,
        description=description,
        final_end_year=final_end_year,
        sample_weight=(
            np.asarray(sample_weights) if sample_weights is not None else None
        ),  # Pass sample weights if available
    )

    logger.info(f"Model finalized and saved to {finalized_dir}")
    logger.info(f"Final end year for model training: {final_end_year}")
    return finalized_dir


def main():
    parser = argparse.ArgumentParser(description="Finalize model for production")
    parser.add_argument(
        "--model-type", type=str, default="hurdle", help="Type of model to finalize"
    )
    parser.add_argument(
        "--experiment", type=str, required=True, help="Name of experiment to finalize"
    )
    parser.add_argument(
        "--version", type=int, help="Optional specific version to finalize"
    )
    parser.add_argument(
        "--end-year", type=int, help="Optional end year for training data"
    )
    parser.add_argument(
        "--description", type=str, help="Optional description of finalized model"
    )
    parser.add_argument(
        "--recent-year-threshold",
        type=int,
        default=2,
        help="Number of years to consider 'recent' for filtering",
    )

    args = parser.parse_args()
    finalize_model(
        model_type=args.model_type,
        experiment_name=args.experiment,
        version=args.version,
        end_year=args.end_year,
        description=args.description,
        recent_year_threshold=args.recent_year_threshold,
    )


if __name__ == "__main__":
    main()
