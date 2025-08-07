import argparse
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from src.data.loader import BGGDataLoader
from src.data.config import load_config
from src.models.geek_rating import calculate_geek_rating
from src.models.score import load_model
from src.models.experiments import ExperimentTracker


def construct_year_filter(
    start_year: Optional[int] = None, end_year: Optional[int] = None
) -> str:
    """
    Construct SQL WHERE clause for year filtering.

    Args:
        start_year: Minimum year (inclusive)
        end_year: Maximum year (exclusive)

    Returns:
        SQL WHERE clause as a string
    """
    where_clauses = []
    if start_year is not None:
        where_clauses.append(f"year_published >= {start_year}")
    if end_year is not None:
        where_clauses.append(f"year_published < {end_year}")

    return " AND ".join(where_clauses) if where_clauses else ""


def load_game_data(
    start_year: Optional[int] = None, end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Load game data with optional year filtering.

    Args:
        start_year: Minimum year (inclusive)
        end_year: Maximum year (exclusive)

    Returns:
        Pandas DataFrame of game features
    """
    config = load_config()
    loader = BGGDataLoader(config)

    where_clause = construct_year_filter(start_year, end_year)
    df = loader.load_data(where_clause=where_clause, preprocessor=None)

    return df.to_pandas()


def predict_hurdle_probabilities(
    hurdle_model: Any, features: pd.DataFrame, threshold: float = 0.5
) -> pd.Series:
    """
    Predict hurdle probabilities for games.

    Args:
        hurdle_model: Trained hurdle model
        features: Game features DataFrame
        threshold: Probability threshold for rating likelihood

    Returns:
        Series of hurdle probabilities
    """
    predicted_hurdle_prob = hurdle_model.predict_proba(features)[:, 1]
    return pd.Series(predicted_hurdle_prob, name="predicted_hurdle_prob")


def predict_game_characteristics(
    features: pd.DataFrame,
    complexity_model: Any,
    rating_model: Any,
    users_rated_model: Any,
    likely_games_mask: pd.Series,  # Kept for compatibility but not used
) -> pd.DataFrame:
    """
    Predict game complexity, rating, and users rated for all games.

    Args:
        features: Game features DataFrame
        complexity_model: Trained complexity model
        rating_model: Trained rating model
        users_rated_model: Trained users rated model
        likely_games_mask: Boolean mask of games likely to be rated (kept for compatibility)

    Returns:
        DataFrame with predicted characteristics
    """
    results = pd.DataFrame(index=features.index)

    # Predict complexity for all games
    results["predicted_complexity"] = complexity_model.predict(features)

    # Add predicted complexity to features
    features_with_complexity = features.copy()
    features_with_complexity["predicted_complexity"] = results["predicted_complexity"]

    # Predict rating and users rated for all games
    results["predicted_rating"] = rating_model.predict(features_with_complexity)
    results["predicted_users_rated"] = users_rated_model.predict(
        features_with_complexity
    )

    # Ensure users is at least minimum threshold
    results["predicted_users_rated"] = np.maximum(
        np.round(np.expm1(results["predicted_users_rated"]) / 50) * 50,
        25,
    )

    return results


def predict_games(
    hurdle_model: Any,
    complexity_model: Any,
    rating_model: Any,
    users_rated_model: Any,
    hurdle_experiment: str,
    complexity_experiment: str,
    rating_experiment: str,
    users_rated_experiment: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    threshold: float = 0.5,
    prior_rating: float = 5.5,
    prior_weight: float = 2000,
) -> pd.DataFrame:
    """
    Predict game characteristics and geek ratings.

    Args:
        hurdle_model: Trained hurdle model
        complexity_model: Trained complexity model
        rating_model: Trained rating model
        users_rated_model: Trained users rated model
        hurdle_experiment: Experiment name for hurdle model
        complexity_experiment: Experiment name for complexity model
        rating_experiment: Experiment name for rating model
        users_rated_experiment: Experiment name for users rated model
        start_year: Minimum year (inclusive)
        end_year: Maximum year (exclusive)
        threshold: Probability threshold for rating likelihood
        prior_rating: Prior mean rating for Bayesian average
        prior_weight: Weight given to prior rating

    Returns:
        DataFrame with game predictions
    """
    # Initialize experiment trackers
    hurdle_tracker = ExperimentTracker(model_type="hurdle")
    complexity_tracker = ExperimentTracker(model_type="complexity")
    rating_tracker = ExperimentTracker(model_type="rating")
    users_rated_tracker = ExperimentTracker(model_type="users_rated")

    # Load experiment details
    hurdle_experiment_details = hurdle_tracker.load_experiment(hurdle_experiment)
    complexity_experiment_details = complexity_tracker.load_experiment(
        complexity_experiment
    )
    rating_experiment_details = rating_tracker.load_experiment(rating_experiment)
    users_rated_experiment_details = users_rated_tracker.load_experiment(
        users_rated_experiment
    )

    # Load game data
    df_pandas = load_game_data(start_year, end_year)

    # Predict hurdle probabilities
    predicted_hurdle_prob = predict_hurdle_probabilities(
        hurdle_model, df_pandas, threshold
    )

    # Prepare results DataFrame
    results = pd.DataFrame(
        {
            "game_id": df_pandas["game_id"],
            "name": df_pandas["name"],
            "year_published": df_pandas["year_published"],
            "predicted_hurdle_prob": predicted_hurdle_prob,
        }
    )

    # Identify games likely to receive ratings
    likely_games_mask = predicted_hurdle_prob >= threshold

    # Predict game characteristics
    characteristics = predict_game_characteristics(
        df_pandas, complexity_model, rating_model, users_rated_model, likely_games_mask
    )

    # Combine results
    results = pd.concat([results, characteristics], axis=1)

    # Calculate predicted geek rating
    results["predicted_geek_rating"] = calculate_geek_rating(
        results, prior_rating=prior_rating, prior_weight=prior_weight
    )

    # Add model experiment identifiers and metadata
    results["hurdle_experiment"] = hurdle_experiment
    results["complexity_experiment"] = complexity_experiment
    results["rating_experiment"] = rating_experiment
    results["users_rated_experiment"] = users_rated_experiment

    # Add timestamp of scoring
    results["score_ts"] = datetime.now(timezone.utc).isoformat()

    return results


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Predict game characteristics and geek ratings"
    )

    # Model arguments
    parser.add_argument(
        "--hurdle",
        default="linear-hurdle",
        help="Experiment name for hurdle model",
    )
    parser.add_argument(
        "--complexity",
        default="catboost-complexity",
        help="Experiment name for complexity model",
    )
    parser.add_argument(
        "--rating",
        default="catboost-rating",
        help="Experiment name for rating model",
    )
    parser.add_argument(
        "--users-rated",
        default="lightgbm-users_rated",
        help="Experiment name for users rated model",
    )

    # Year filtering arguments
    parser.add_argument(
        "--start-year",
        type=int,
        default=2024,
        help="Start year for predictions (inclusive)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2029,
        help="End year for predictions (exclusive)",
    )

    # Prediction parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for rating likelihood",
    )
    parser.add_argument(
        "--prior-rating",
        type=float,
        default=5.5,
        help="Prior mean rating for Bayesian average",
    )
    parser.add_argument(
        "--prior-weight", type=float, default=2000, help="Weight given to prior rating"
    )

    # Output arguments
    parser.add_argument(
        "--output",
        default="data/predictions/game_predictions.parquet",
        help="Output filename for predictions",
    )

    # Parse arguments
    args = parser.parse_args()

    # Load models
    hurdle_model = load_model(args.hurdle, model_type="hurdle")
    complexity_model = load_model(args.complexity, model_type="complexity")
    rating_model = load_model(args.rating, model_type="rating")
    users_rated_model = load_model(args.users_rated, model_type="users_rated")

    # Predict games
    predictions = predict_games(
        hurdle_model=hurdle_model,
        complexity_model=complexity_model,
        rating_model=rating_model,
        users_rated_model=users_rated_model,
        hurdle_experiment=args.hurdle,
        complexity_experiment=args.complexity,
        rating_experiment=args.rating,
        users_rated_experiment=args.users_rated,
        start_year=args.start_year,
        end_year=args.end_year,
        threshold=args.threshold,
        prior_rating=args.prior_rating,
        prior_weight=args.prior_weight,
    )

    # Save predictions
    predictions.to_parquet(args.output, index=False)
    print(predictions)


if __name__ == "__main__":
    main()
