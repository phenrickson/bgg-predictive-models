"""
Comprehensive pipeline for Hurdle Model: Training, Finalization, and Scoring

This script provides a complete workflow for:
1. Training a hurdle model
2. Finalizing the model
3. Scoring the model on new data
"""

import argparse
import logging
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Project imports
import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data.config import load_config
from src.data.loader import BGGDataLoader
from src.models.experiments import ExperimentTracker
from src.models.hurdle import (
    create_preprocessing_pipeline,
    preprocess_data,
    tune_hyperparameters,
    evaluate_model
)
from src.models.finalize_model import finalize_experiment
from src.models.score import score_data

def setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)

def train_hurdle_model(
    train_end_year: int,
    tune_start_year: int,
    tune_end_year: int,
    test_start_year: int,
    test_end_year: int,
    experiment_name: str = "hurdle_model",
    output_dir: str = "./models/experiments"
):
    """
    Train a hurdle model with specified data splits.
    
    Args:
        train_end_year: Exclusive end year for training data
        tune_start_year: Inclusive start year for tuning data
        tune_end_year: Inclusive end year for tuning data
        test_start_year: Inclusive start year for test data
        test_end_year: Inclusive end year for test data
        experiment_name: Name of the experiment
        output_dir: Directory to save experiment results
    
    Returns:
        Trained experiment object
    """
    logger = setup_logging()
    
    # Load data
    config = load_config()
    loader = BGGDataLoader(config)
    df = loader.load_training_data(end_train_year=test_end_year + 1, min_ratings=0)
    
    logger.info(f"Loaded {len(df)} total rows")
    logger.info(f"Year range: {df['year_published'].min()} - {df['year_published'].max()}")
    
    # Create data splits
    train_df = df.filter(pl.col("year_published") < train_end_year)
    tune_df = df.filter(
        (pl.col("year_published") >= tune_start_year) & 
        (pl.col("year_published") <= tune_end_year)
    )
    test_df = df.filter(
        (pl.col("year_published") >= test_start_year) & 
        (pl.col("year_published") <= test_end_year)
    )
    
    logger.info(f"Training data: {len(train_df)} rows (years < {train_end_year})")
    logger.info(f"Tuning data: {len(tune_df)} rows (years {tune_start_year}-{tune_end_year})")
    logger.info(f"Test data: {len(test_df)} rows (years {test_start_year}-{test_end_year})")
    
    # Create preprocessing pipeline
    preprocessing_pipeline = create_preprocessing_pipeline()
    
    # Preprocess data
    X_train = preprocess_data(train_df, preprocessing_pipeline, fit=True, dataset_name="training")
    X_tune = preprocess_data(tune_df, preprocessing_pipeline, fit=False, dataset_name="tuning")
    X_test = preprocess_data(test_df, preprocessing_pipeline, fit=False, dataset_name="test")
    
    # Get targets
    y_train = train_df.select("hurdle").to_pandas().squeeze()
    y_tune = tune_df.select("hurdle").to_pandas().squeeze()
    y_test = test_df.select("hurdle").to_pandas().squeeze()
    
    # Tune hyperparameters
    best_params = tune_hyperparameters(X_train, y_train, X_tune, y_tune)
    
    # Train final model
    final_model = LogisticRegression(**best_params)
    final_model.fit(X_train, y_train)
    
    # Create complete pipeline
    full_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('model', final_model)
    ])
    
    # Evaluate model
    train_metrics = evaluate_model(final_model, X_train, y_train, "training")
    tune_metrics = evaluate_model(final_model, X_tune, y_tune, "tuning")
    test_metrics = evaluate_model(final_model, X_test, y_test, "test")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker("hurdle", output_dir)
    
    # Create experiment
    experiment = tracker.create_experiment(
        name=experiment_name,
        metadata={
            'train_end_year_exclusive': train_end_year,
            'tune_start_year': tune_start_year,
            'tune_end_year': tune_end_year,
            'test_start_year': test_start_year,
            'test_end_year': test_end_year,
            'model_type': 'hurdle',
            'target': 'users_rated >= 25'
        }
    )
    
    # Log metrics and pipeline
    experiment.log_metrics(train_metrics, "train")
    experiment.log_metrics(tune_metrics, "tune")
    experiment.log_metrics(test_metrics, "test")
    experiment.save_pipeline(full_pipeline)
    
    return experiment

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Hurdle Model Training and Scoring Pipeline")
    
    # Training parameters
    parser.add_argument("--train-end-year", type=int, default=2022, 
                        help="End year for training (exclusive)")
    parser.add_argument("--tune-start-year", type=int, default=2022,
                        help="Start year for tuning (inclusive)")
    parser.add_argument("--tune-end-year", type=int, default=2023,
                        help="End year for tuning (inclusive)")
    parser.add_argument("--test-start-year", type=int, default=2024,
                        help="Start year for testing (inclusive)")
    parser.add_argument("--test-end-year", type=int, default=2025,
                        help="End year for testing (inclusive)")
    
    # Experiment and output parameters
    parser.add_argument("--experiment", type=str, default="hurdle_model",
                        help="Name of the experiment")
    parser.add_argument("--output-dir", type=str, default="./models/experiments",
                        help="Directory to save experiment results")
    
    # Finalization parameters
    parser.add_argument("--finalize", action="store_true",
                        help="Finalize the model after training")
    
    # Scoring parameters
    parser.add_argument("--score", action="store_true",
                        help="Score the model after training and optional finalization")
    parser.add_argument("--data-path", type=str,
                        help="Optional path to CSV file for scoring")
    parser.add_argument("--output-path", type=str,
                        help="Optional path to save predictions")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Train model
    experiment = train_hurdle_model(
        train_end_year=args.train_end_year,
        tune_start_year=args.tune_start_year,
        tune_end_year=args.tune_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
        experiment_name=args.experiment,
        output_dir=args.output_dir
    )
    
    # Optional: Finalize model
    if args.finalize:
        logger.info("Finalizing model...")
        finalize_experiment(
            model_type="hurdle",
            experiment_name=args.experiment,
            version=None  # Use latest version
        )
    
    # Optional: Score model
    if args.score:
        logger.info("Scoring model...")
        score_data(
            experiment_name=args.experiment,
            data_path=args.data_path,
            output_path=args.output_path
        )
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
