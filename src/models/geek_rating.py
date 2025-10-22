"""Functions for predicting geek ratings by combining multiple model predictions."""

import numpy as np
import pandas as pd
import polars as pl
import os
import argparse
from typing import Dict, Optional, Union

from src.data.loader import BGGDataLoader
from src.utils.config import load_config
from src.models.score import load_model
from src.utils.logging import setup_logging
from src.models.experiments import ExperimentTracker

logger = setup_logging()


def load_all_models(
    hurdle_experiment: str,
    complexity_experiment: str,
    rating_experiment: str,
    users_rated_experiment: str,
) -> Dict:
    """
    Load all required models from experiments.

    Args:
        hurdle_experiment: Experiment name for hurdle model
        complexity_experiment: Experiment name for complexity model
        rating_experiment: Experiment name for rating model
        users_rated_experiment: Experiment name for users_rated model

    Returns:
        Dict of loaded models
    """
    logger.info("Loading models from experiments:")
    logger.info(f"Hurdle: {hurdle_experiment}")
    logger.info(f"Complexity: {complexity_experiment}")
    logger.info(f"Rating: {rating_experiment}")
    logger.info(f"Users Rated: {users_rated_experiment}")

    return {
        "hurdle": load_model(hurdle_experiment, model_type="hurdle"),
        "complexity": load_model(complexity_experiment, model_type="complexity"),
        "rating": load_model(rating_experiment, model_type="rating"),
        "users_rated": load_model(users_rated_experiment, model_type="users_rated"),
    }


def predict_game(
    features: pd.DataFrame,
    models: Dict,
    threshold: Optional[float] = None,
    hurdle_experiment: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate all predictions for games.

    Args:
        features: DataFrame with game features
        models: Dict of loaded models
        threshold: Optional threshold for hurdle model classification
        hurdle_experiment: Optional experiment name to load optimal threshold

    Returns:
        DataFrame with predictions for each game
    """
    logger.info("Predicting game characteristics")

    # Dynamically load optimal threshold if experiment is provided
    if hurdle_experiment is not None:
        from src.models.experiments import ExperimentTracker

        tracker = ExperimentTracker(model_type="hurdle")
        experiment = tracker.load_experiment(hurdle_experiment)

        # Retrieve optimal threshold from experiment metadata
        threshold = experiment.metadata.get("optimal_threshold", 0.5)
        logger.info(f"Using optimal threshold from experiment: {threshold}")

    # Use default threshold if not provided
    if threshold is None:
        threshold = 0.5

    # Step 1: Predict if games will get enough ratings
    will_rate_proba = models["hurdle"].predict_proba(features)[:, 1]

    # Prepare results DataFrame
    results = pd.DataFrame(index=features.index)

    # Preserve identifying fields
    results["game_id"] = features["game_id"]
    results["name"] = features["name"]
    results["year_published"] = features["year_published"]

    results["predicted_hurdle_prob"] = will_rate_proba

    # Identify games likely to receive ratings
    likely_games_mask = will_rate_proba >= threshold

    # Predict for likely games
    if likely_games_mask.any():
        # Subset features for likely games
        likely_features = features[likely_games_mask]

        # Predict complexity
        results.loc[likely_games_mask, "predicted_complexity"] = models[
            "complexity"
        ].predict(likely_features)

        # Add predicted complexity as feature
        likely_features_with_complexity = likely_features.copy()
        likely_features_with_complexity["predicted_complexity"] = results.loc[
            likely_games_mask, "predicted_complexity"
        ]

        # Predict rating and users
        results.loc[likely_games_mask, "predicted_rating"] = models["rating"].predict(
            likely_features_with_complexity
        )
        results.loc[likely_games_mask, "predicted_users_rated"] = models[
            "users_rated"
        ].predict(likely_features_with_complexity)

        # Ensure users is at least minimum threshold
        results.loc[likely_games_mask, "predicted_users_rated"] = np.maximum(
            np.round(
                np.expm1(results.loc[likely_games_mask, "predicted_users_rated"]) / 50
            )
            * 50,
            25,
        )

    # Fill in default values for unlikely games
    results.loc[~likely_games_mask, "predicted_complexity"] = 1.0
    results.loc[~likely_games_mask, "predicted_rating"] = 5.5
    results.loc[~likely_games_mask, "predicted_users_rated"] = 25

    logger.info(
        f"Predictions: {len(likely_games_mask)} likely games out of {len(features)} total"
    )

    return results


def calculate_geek_rating(
    predictions: Union[pd.DataFrame, Dict[str, float]],
    prior_rating: float = 5.5,
    prior_weight: float = 2000,
) -> Union[pd.Series, float]:
    """
    Calculate geek rating using Bayesian average.

    Args:
        predictions: DataFrame with 'rating' and 'users_rated' columns,
                    or Dict with 'rating' and 'users_rated' keys
        prior_rating: Prior mean rating
        prior_weight: Weight given to prior

    Returns:
        Series of geek ratings if input is DataFrame,
        or single geek rating float if input is Dict
    """
    # Try original column names first, then predicted column names
    try:
        rating = predictions["rating"]
        users = predictions["users_rated"]
    except (KeyError, TypeError):
        rating = predictions["predicted_rating"]
        users = predictions["predicted_users_rated"]

    return ((rating * users) + (prior_rating * prior_weight)) / (users + prior_weight)


def predict_geek_rating(
    features: pd.DataFrame,
    models: Optional[Dict] = None,
    experiments: Optional[Dict[str, str]] = None,
    hurdle_experiment: Optional[str] = None,
    complexity_experiment: Optional[str] = None,
    rating_experiment: Optional[str] = None,
    users_rated_experiment: Optional[str] = None,
    threshold: Optional[float] = None,
    prior_rating: float = 5.5,
    prior_weight: float = 2000,
) -> pd.DataFrame:
    """
    Predict geek ratings for board games using a multi-stage predictive model.

    This function provides a flexible approach to predicting geek ratings by combining
    multiple machine learning models. It supports various model loading strategies and
    allows fine-tuning of the prediction process.

    The prediction process involves:
    1. Hurdle model: Predicting likelihood of game receiving ratings
    2. Complexity model: Estimating game complexity
    3. Rating model: Predicting game rating
    4. Users Rated model: Estimating number of users who will rate the game
    5. Bayesian average calculation: Computing final geek rating

    Model Loading Strategies:
    - Directly provide pre-loaded models
    - Specify experiment names to load models dynamically
    - Use a combination of pre-loaded models and experiment names

    Args:
        features (pd.DataFrame): Input DataFrame containing game features.
            Must include columns like 'game_id', 'name', 'year_published', etc.

        models (Optional[Dict]): Pre-loaded dictionary of machine learning models.
            Keys should match: 'hurdle', 'complexity', 'rating', 'users_rated'.
            Example: {
                'hurdle': hurdle_model,
                'complexity': complexity_model,
                'rating': rating_model,
                'users_rated': users_rated_model
            }

        experiments (Optional[Dict[str, str]]): Dictionary mapping model types to
            experiment names for dynamic model loading.
            Example: {
                'hurdle_experiment': 'test-hurdle',
                'complexity_experiment': 'test-complexity',
                'rating_experiment': 'test-rating',
                'users_rated_experiment': 'test-users_rated'
            }

        hurdle_experiment (Optional[str]): Specific experiment name for hurdle model.
        complexity_experiment (Optional[str]): Specific experiment name for complexity model.
        rating_experiment (Optional[str]): Specific experiment name for rating model.
        users_rated_experiment (Optional[str]): Specific experiment name for users_rated model.

        threshold (Optional[float]): Classification threshold for hurdle model.
            Games with prediction probability above this are considered likely to be rated.
            Defaults to 0.5 if not specified or cannot be retrieved from experiment.

        prior_rating (float, optional): Prior mean rating for Bayesian average calculation.
            Defaults to 5.5, representing a neutral game rating.

        prior_weight (float, optional): Weight given to prior rating in Bayesian average.
            Higher values give more importance to the prior rating.
            Defaults to 2000, providing significant prior influence.

    Returns:
        pd.DataFrame: Predictions including:
            - game_id: Original game identifier
            - name: Game name
            - year_published: Year game was published
            - will_rate: Probability of game receiving ratings
            - complexity: Predicted game complexity
            - rating: Predicted game rating
            - users_rated: Predicted number of users who will rate the game
            - geek_rating: Final Bayesian average geek rating

    Raises:
        ValueError: If no valid method is provided to load models.

    Examples:
        # Example 1: Using pre-loaded models
        predictions = predict_geek_rating(
            features=game_features,
            models={
                'hurdle': hurdle_model,
                'complexity': complexity_model,
                'rating': rating_model,
                'users_rated': users_rated_model
            }
        )

        # Example 2: Using experiment names
        predictions = predict_geek_rating(
            features=game_features,
            hurdle_experiment='test-hurdle',
            complexity_experiment='test-complexity',
            rating_experiment='test-rating',
            users_rated_experiment='test-users_rated'
        )

        # Example 3: Customizing Bayesian average
        predictions = predict_geek_rating(
            features=game_features,
            models=models,
            threshold=0.6,
            prior_rating=6.0,
            prior_weight=1500
        )
    """
    logger.info("Starting geek rating prediction")

    # Prioritize loading models in this order:
    # 1. Directly provided models
    # 2. Experiments dictionary
    # 3. Individual experiment arguments
    if models is None:
        if experiments is not None:
            # Use experiments dictionary if provided
            models = load_all_models(**experiments)
        elif any(
            [
                hurdle_experiment,
                complexity_experiment,
                rating_experiment,
                users_rated_experiment,
            ]
        ):
            # Use individual experiment arguments
            models = load_all_models(
                hurdle_experiment=hurdle_experiment or "test-hurdle",
                complexity_experiment=complexity_experiment or "test-complexity",
                rating_experiment=rating_experiment or "test-rating",
                users_rated_experiment=users_rated_experiment or "test-users_rated",
            )
        else:
            raise ValueError(
                "No valid method provided to load models. "
                "Provide either 'models', 'experiments', or individual experiment names."
            )

    # Get predictions
    predictions = predict_game(
        features, models, threshold=threshold, hurdle_experiment=hurdle_experiment
    )

    # Calculate geek rating
    predictions["prediction"] = calculate_geek_rating(
        predictions, prior_rating=prior_rating, prior_weight=prior_weight
    )

    logger.info(f"Final Predictions: {predictions}")

    return predictions


# # Debugging function to help diagnose import issues
# def debug_import():
#     """Print debug information about module import."""
#     print("Geek Rating Module Imported Successfully")
#     print(f"Module Name: {__name__}")
#     print(f"Module File: {__file__}")

# # Call debug function when module is imported
# debug_import()


def main():
    """
    Predict geek ratings using command-line specified experiments.

    Allows flexible loading of models via command-line arguments and
    tracks the experiment using ExperimentTracker.
    """
    # Load config to get default paths
    config = load_config()
    complexity_predictions_path = config.models["complexity"].predictions_path

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Predict geek ratings for board games")

    # Add argument for local complexity predictions path
    parser.add_argument(
        "--local-complexity-path",
        type=str,
        default=complexity_predictions_path,
        help="Path to local complexity predictions parquet file",
    )

    # Add arguments for each model experiment
    parser.add_argument(
        "--hurdle",
        default="test-hurdle",
        help="Experiment name for hurdle model (default: test-hurdle)",
    )
    parser.add_argument(
        "--complexity",
        default="test-complexity",
        help="Experiment name for complexity model (default: test-complexity)",
    )
    parser.add_argument(
        "--rating",
        default="test-rating",
        help="Experiment name for rating model (default: test-rating)",
    )
    parser.add_argument(
        "--users-rated",
        default="test-users_rated",
        help="Experiment name for users rated model (default: test-users_rated)",
    )

    # Optional parameters for prediction
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold for hurdle model",
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

    # Output file argument
    parser.add_argument(
        "--output",
        default="geek_ratings.parquet",
        help="Output filename for predictions",
    )

    # Experiment name argument
    parser.add_argument(
        "--experiment",
        default="geek_rating_prediction",
        help="Name of the experiment for tracking (default: geek_rating_prediction)",
    )

    # Add output directory argument
    parser.add_argument(
        "--output-dir",
        default="./data/predictions",
        help="Base directory for output files",
    )

    # Add year filtering arguments
    parser.add_argument(
        "--start-year", type=int, help="Start year for filtering games (inclusive)"
    )
    parser.add_argument(
        "--end-year", type=int, help="End year for filtering games (exclusive)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create experiment tracker
    tracker = ExperimentTracker(model_type="geek_rating")

    # Function to get experiment years
    def get_experiment_years(experiment_name, model_type):
        try:
            # Create tracker with correct model type
            model_tracker = ExperimentTracker(model_type=model_type)
            experiment = model_tracker.load_experiment(experiment_name)
            # Assuming the years are stored in the metadata
            return experiment.metadata.get("data_years", {})
        except Exception as e:
            logger.warning(
                f"Could not retrieve years for experiment {experiment_name}: {e}"
            )
            return {}

    # Map experiment names to their model types
    experiment_types = {
        args.hurdle: "hurdle",
        args.complexity: "complexity",
        args.rating: "rating",
        args.users_rated: "users_rated",
    }

    # Collect years from individual experiments
    experiment_years = {
        model_type: get_experiment_years(exp_name, model_type)
        for exp_name, model_type in experiment_types.items()
    }

    # Create experiment with additional metadata
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
                "threshold": args.threshold or 0.5,
                "prior_rating": args.prior_rating,
                "prior_weight": args.prior_weight,
            },
            "experiment_years": experiment_years,
        },
    )

    # Load configuration and data
    config = load_config()
    bigquery_config = config.get_bigquery_config()
    loader = BGGDataLoader(bigquery_config)

    # Load complexity predictions from local file
    logger.info(f"Loading complexity predictions from {args.local_complexity_path}")
    try:
        complexity_df = pl.read_parquet(args.local_complexity_path)
        logger.info(f"Loaded {len(complexity_df)} complexity predictions")
    except Exception as e:
        logger.error(f"Failed to load complexity predictions: {e}")
        raise

    # Construct WHERE clause for year filtering
    where_clauses = []
    if args.start_year is not None:
        where_clauses.append(f"year_published >= {args.start_year}")
    if args.end_year is not None:
        where_clauses.append(f"year_published <= {args.end_year}")

    # Load data with optional year filtering
    where_clause = " AND ".join(where_clauses) if where_clauses else ""
    df = loader.load_data(where_clause=where_clause, preprocessor=None)

    # Join with complexity predictions
    df = df.join(complexity_df, on="game_id", how="inner")
    logger.info(f"After joining with complexity predictions: {len(df)} games")

    logger.info(
        f"Filtered to {len(df)} games between years {args.start_year or 'min'} and {args.end_year or 'max'}"
    )

    # Convert to pandas for prediction
    df_pandas = df.to_pandas()

    # Predict geek ratings using command-line specified experiments
    predictions = predict_geek_rating(
        features=df_pandas,
        hurdle_experiment=args.hurdle,
        complexity_experiment=args.complexity,
        rating_experiment=args.rating,
        users_rated_experiment=args.users_rated,
        threshold=args.threshold,
        prior_rating=args.prior_rating,
        prior_weight=args.prior_weight,
    )

    # Convert to Polars for saving
    results = pl.from_pandas(predictions)

    # Ensure predictions directory exists with full path
    predictions_dir = os.path.join(
        os.path.abspath(args.output_dir), tracker.model_type, args.experiment
    )
    os.makedirs(predictions_dir, exist_ok=True)

    # Construct output filename
    output_filename = "predictions.parquet"
    output_path = os.path.join(predictions_dir, output_filename)

    # Log detailed file saving information
    logger.info(f"Saving predictions to: {output_path}")

    # Verify directory exists and is writable
    if not os.access(predictions_dir, os.W_OK):
        logger.error(f"Cannot write to directory: {predictions_dir}")
        raise PermissionError(f"No write permission for directory: {predictions_dir}")

    # Save predictions
    results.write_parquet(output_path)

    # Verify file was created
    if not os.path.exists(output_path):
        logger.error(f"Failed to save predictions to: {output_path}")
        raise IOError(f"Could not save predictions file: {output_path}")

    # Determine actual values
    if "geek_rating" in df.columns:
        # Use existing geek_rating as actual values
        # Set to NaN if geek_rating is zero (representing missingness)
        actuals = np.where(
            df["geek_rating"].to_numpy() == 0, np.nan, df["geek_rating"].to_numpy()
        )
    else:
        # Fallback to NaN if no geek_rating column exists
        actuals = np.full(len(results), np.nan)

    # Log predictions to the experiment
    experiment.log_predictions(
        predictions=results["prediction"].to_numpy(),
        actuals=actuals,
        df=results,
        dataset="test",  # Changed to "test" to match the filename
    )

    logger.info("Experiment tracked:")
    logger.info(f"  Name: {args.experiment}")
    logger.info("  Experiments used:")
    logger.info(f"    Hurdle: {args.hurdle}")
    logger.info(f"    Complexity: {args.complexity}")
    logger.info(f"    Rating: {args.rating}")
    logger.info(f"    Users Rated: {args.users_rated}")


# Allow script to be run directly
if __name__ == "__main__":
    main()
