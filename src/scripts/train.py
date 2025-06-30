"""Example script for training the BGG predictive models pipeline."""
import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

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


def train_pipeline(
    end_train_year: int = 2021,
    valid_years: int = 2,
    min_ratings: int = 25,
    random_state: int = 42,
    model_params: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
) -> BGGPipeline:
    """Train full prediction pipeline.
    
    Args:
        end_train_year: Last year to include in training
        valid_years: Number of years to use for validation
        min_ratings: Minimum number of ratings threshold
        random_state: Random seed
        model_params: Optional model-specific parameters
        output_dir: Optional directory to save models
        
    Returns:
        Trained pipeline
    """
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info("Loading data from warehouse...")
    config = get_config_from_env()
    loader = BGGDataLoader(config)
    features, targets = loader.load_training_data(
        end_train_year=end_train_year,
        min_ratings=min_ratings,
    )
    
    # Split into train/validation
    logger.info("Splitting data...")
    X_train, X_val, y_train_dict, y_val_dict = {}, {}, {}, {}
    
    for target_name, y in targets.items():
        # Remove rows with missing targets
        mask = y.notna()
        X = features[mask]
        y = y[mask]
        
        # Split while preserving time-based order
        X_train[target_name], X_val[target_name], y_train_dict[target_name], y_val_dict[target_name] = (
            train_test_split(X, y, test_size=0.2, shuffle=False)
        )
    
    # Initialize and train pipeline
    logger.info("Training models...")
    pipeline = BGGPipeline(
        valid_years=valid_years,
        min_ratings=min_ratings,
        random_state=random_state,
        model_params=model_params,
    )
    
    pipeline.fit(
        X=X_train["hurdle"],  # Use same features for all models
        y_hurdle=y_train_dict["hurdle"],
        y_complexity=y_train_dict["complexity"],
        y_rating=y_train_dict["rating"],
        y_users_rated=y_train_dict["users_rated"],
    )
    
    # Evaluate on validation set
    logger.info("Evaluating models...")
    metrics = pipeline.evaluate(
        X=X_val["hurdle"],
        y_hurdle=y_val_dict["hurdle"],
        y_complexity=y_val_dict["complexity"],
        y_rating=y_val_dict["rating"],
        y_users_rated=y_val_dict["users_rated"],
    )
    
    # Log metrics
    for model_name, model_metrics in metrics.items():
        logger.info(f"{model_name} metrics:")
        for metric_name, value in model_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
    
    # Save pipeline if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / "pipeline.joblib"
        logger.info(f"Saving pipeline to {model_path}")
        joblib.dump(pipeline, model_path)
        
        # Save feature names
        feature_names = pd.DataFrame({
            "feature": features.columns.tolist(),
        })
        feature_names.to_csv(output_dir / "feature_names.csv", index=False)
    
    return pipeline


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BGG predictive models")
    parser.add_argument(
        "--end-train-year",
        type=int,
        default=2021,
        help="Last year to include in training",
    )
    parser.add_argument(
        "--valid-years",
        type=int,
        default=2,
        help="Number of years to use for validation",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=25,
        help="Minimum number of ratings threshold",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(Path(args.log_file) if args.log_file else None)
    
    # Train pipeline
    train_pipeline(
        end_train_year=args.end_train_year,
        valid_years=args.valid_years,
        min_ratings=args.min_ratings,
        random_state=args.random_state,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
