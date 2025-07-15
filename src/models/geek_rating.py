"""Functions for predicting geek ratings by combining multiple model predictions."""

import logging
import numpy as np
import pandas as pd
import polars as pl
import os
from typing import Dict, Optional, Union

from src.data.loader import BGGDataLoader
from src.data.config import load_config
from src.models.score import load_model, load_scoring_data, save_and_display_results
from src.models.experiments import ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_all_models(
    hurdle_experiment: str,
    complexity_experiment: str,
    rating_experiment: str,
    users_rated_experiment: str
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
    logger.info(f"Loading models from experiments:")
    logger.info(f"Hurdle: {hurdle_experiment}")
    logger.info(f"Complexity: {complexity_experiment}")
    logger.info(f"Rating: {rating_experiment}")
    logger.info(f"Users Rated: {users_rated_experiment}")
    
    return {
        'hurdle': load_model(hurdle_experiment, model_type='hurdle'),
        'complexity': load_model(complexity_experiment, model_type='complexity'),
        'rating': load_model(rating_experiment, model_type='rating'),
        'users_rated': load_model(users_rated_experiment, model_type='users_rated')
    }

def predict_game(
    features: pd.DataFrame,
    models: Dict,
    threshold: Optional[float] = None,
    hurdle_experiment: Optional[str] = None
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
        tracker = ExperimentTracker(model_type='hurdle')
        experiment = tracker.load_experiment(hurdle_experiment)
        
        # Retrieve optimal threshold from experiment metadata
        threshold = experiment.metadata.get('optimal_threshold', 0.5)
        logger.info(f"Using optimal threshold from experiment: {threshold}")
    
    # Use default threshold if not provided
    if threshold is None:
        threshold = 0.5
    
    # Step 1: Predict if games will get enough ratings
    will_rate_proba = models['hurdle'].predict_proba(features)[:, 1]
    
    # Prepare results DataFrame
    results = pd.DataFrame(index=features.index)
    
    # Preserve identifying fields
    results['game_id'] = features['game_id']
    results['name'] = features['name']
    results['year_published'] = features['year_published']
    
    results['will_rate'] = will_rate_proba
    
    # Identify games likely to receive ratings
    likely_games_mask = will_rate_proba >= threshold
    
    # Predict for likely games
    if likely_games_mask.any():
        # Subset features for likely games
        likely_features = features[likely_games_mask]
        
        # Predict complexity
        results.loc[likely_games_mask, 'complexity'] = models['complexity'].predict(likely_features)
        
        # Add predicted complexity as feature
        likely_features_with_complexity = likely_features.copy()
        likely_features_with_complexity['predicted_complexity'] = results.loc[likely_games_mask, 'complexity']
        
        # Predict rating and users
        results.loc[likely_games_mask, 'rating'] = models['rating'].predict(likely_features_with_complexity)
        results.loc[likely_games_mask, 'users_rated'] = models['users_rated'].predict(likely_features_with_complexity)
        
        # Ensure users is at least minimum threshold
        results.loc[likely_games_mask, 'users_rated'] = np.maximum(
            np.round(np.expm1(results.loc[likely_games_mask, 'users_rated']) / 50) * 50, 
            25
        )
    
    # Fill in default values for unlikely games
    results.loc[~likely_games_mask, 'complexity'] = 1.0
    results.loc[~likely_games_mask, 'rating'] = 5.5
    results.loc[~likely_games_mask, 'users_rated'] = 25
    
    logger.info(f"Predictions: {len(likely_games_mask)} likely games out of {len(features)} total")
    
    return results

def calculate_geek_rating(
    predictions: Union[pd.DataFrame, Dict[str, float]],
    prior_rating: float = 5.5,
    prior_weight: float = 2000
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
    try:
        # DataFrame case - access columns
        rating = predictions['rating']
        users = predictions['users_rated']
    except (TypeError, AttributeError):
        # Dictionary case - access keys
        rating = predictions['rating']
        users = predictions['users_rated']
        
    return ((rating * users) + (prior_rating * prior_weight)) / (users + prior_weight)


def predict_geek_rating(
    features: pd.DataFrame,
    models: Optional[Dict] = None,
    experiments: Optional[Dict[str, str]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Main function to predict geek rating for a game.
    
    Args:
        features: DataFrame with game features
        models: Optional pre-loaded models
        experiments: Optional experiment names to load models
        **kwargs: Additional arguments for calculate_geek_rating
    
    Returns:
        Dict with all predictions including geek rating
    
    Raises:
        ValueError: If neither models nor experiments are provided
    """
    logger.info("Starting geek rating prediction")
    
    # Load models if not provided
    if models is None:
        if experiments is None:
            raise ValueError("Either models or experiments must be provided")
        models = load_all_models(**experiments)
    
    # Get predictions
    predictions = predict_game(features, models)
    
    # Calculate geek rating
    geek_rating = calculate_geek_rating(predictions, **kwargs)
    predictions['geek_rating'] = geek_rating
    
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
    Predict geek ratings for demonstration.
    """
    config = load_config()
    loader = BGGDataLoader(config)
    
    # Load all games with non-null year_published
    df = loader.load_data(
        preprocessor=None
    )
    
    # Convert to pandas for prediction
    df_pandas = df.to_pandas()
    
    # Load models
    models = load_all_models(
        hurdle_experiment='test-hurdle',
        complexity_experiment='test-complexity',
        rating_experiment='test-rating',
        users_rated_experiment='test-users_rated'
    )
    
    # Predict geek ratings for entire DataFrame
    predictions = predict_game(
        df_pandas, 
        models, 
        hurdle_experiment='test-hurdle'
    )
    
    # Calculate geek ratings
    predictions['geek_rating'] = calculate_geek_rating(predictions)
    
    # Convert to Polars for saving
    results = pl.from_pandas(predictions)
    
    # Ensure predictions directory exists
    os.makedirs('data/predictions', exist_ok=True)
    
    output_file = 'geek_ratings.parquet'
    
    # Save predictions to CSV
    results.write_parquet(f'data/predictions/{output_file}')
    
    logger.info(f"Predictions saved to {output_file}")

# Allow script to be run directly
if __name__ == '__main__':
    main()
