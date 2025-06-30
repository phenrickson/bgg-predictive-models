"""Script for making predictions using trained BGG models."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import pandas as pd

from ..data.config import get_config_from_env
from ..data.loader import BGGDataLoader
from ..models.pipeline import BGGPipeline


def setup_logging(log_file: Optional[Path] = None) -> None:
    """Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def load_pipeline(model_dir: Union[str, Path]) -> BGGPipeline:
    """Load trained pipeline from disk.
    
    Args:
        model_dir: Directory containing saved model
        
    Returns:
        Loaded pipeline
        
    Raises:
        FileNotFoundError: If model files not found
    """
    model_dir = Path(model_dir)
    model_path = model_dir / "pipeline.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    return joblib.load(model_path)


def make_predictions(
    pipeline: BGGPipeline,
    game_ids: Optional[List[int]] = None,
    output_file: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate predictions for games.
    
    Args:
        pipeline: Trained prediction pipeline
        game_ids: Optional list of specific game IDs to predict
        output_file: Optional path to save predictions
        
    Returns:
        DataFrame with predictions
    """
    logger = logging.getLogger(__name__)
    
    # Load feature data
    logger.info("Loading game data...")
    config = get_config_from_env()
    loader = BGGDataLoader(config)
    features = loader.load_prediction_data(game_ids=game_ids)
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = pipeline.predict(features)
    
    # Calculate bayesaverage
    logger.info("Calculating bayesaverage...")
    bayesavg = pipeline.predict_bayesaverage(features)
    
    # Combine predictions into DataFrame
    results = pd.DataFrame({
        "game_id": features.index,
        "hurdle_prob": predictions["hurdle"],
        "predicted_complexity": predictions["complexity"],
        "predicted_rating": predictions["rating"],
        "predicted_users_rated": predictions["users_rated"],
        "predicted_bayesaverage": bayesavg,
    })
    
    # Save predictions if output file provided
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving predictions to {output_file}")
        results.to_csv(output_file, index=False)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Make predictions using trained models")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--game-ids",
        type=int,
        nargs="*",
        help="Optional list of game IDs to predict",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save predictions CSV",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(Path(args.log_file) if args.log_file else None)
    
    # Load pipeline
    pipeline = load_pipeline(args.model_dir)
    
    # Make predictions
    predictions = make_predictions(
        pipeline=pipeline,
        game_ids=args.game_ids,
        output_file=args.output_file,
    )
    
    # Print preview if not saving to file
    if not args.output_file:
        print("\nPrediction Preview:")
        print(predictions.head())
