"""Finalize a model for production by fitting on full dataset."""
import argparse
import logging
from typing import Optional
from datetime import datetime

from src.models.experiments import ExperimentTracker
from src.models.experiments import Experiment
from src.data.config import load_config
from src.data.loader import BGGDataLoader
from src.models.hurdle import setup_logging

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
            ('model_info', 'threshold'),  # From log_experiment in hurdle.py
            ('threshold',),               # Direct metadata key
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

def generate_model_description(
    base_description: Optional[str], 
    final_end_year: int, 
    threshold: Optional[float] = None
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
    description = base_description or f"Production model trained on data through {final_end_year}"
    
    # Append threshold if available
    if threshold is not None:
        description += f". Optimal classification threshold: {threshold:.4f}"
    
    return description

def finalize_model(
    model_type: str,
    experiment_name: str,
    version: Optional[int] = None,
    end_year: Optional[int] = None,
    description: Optional[str] = None
):
    """Finalize a model by fitting its pipeline on full dataset.
    
    Args:
        model_type: Type of model (e.g., 'hurdle', 'rating')
        experiment_name: Name of experiment to finalize
        version: Optional specific version to finalize
        end_year: Optional end year for training data
        description: Optional description of finalized model
    """
    logger = setup_logging()
    
    # Load experiment
    tracker = ExperimentTracker(model_type)
    
    # Check if experiment name includes version/hash
    if '_v' in experiment_name:
        # Extract base name from full name
        experiment_name = experiment_name.split('_v')[0]
    
    experiment = tracker.load_experiment(experiment_name, version)
    
    # Load full dataset
    config = load_config()
    loader = BGGDataLoader(config)
    if end_year is None:
        # Dynamically calculate end year, excluding last two years
        current_year = datetime.now().year
        end_year = current_year - 2
        
        # Check if the experiment's test set overlaps with the last two years
        test_end_year = experiment.metadata.get('test_end_year', 0)
        if test_end_year > current_year - 2:
            logger.warning(f"Test set extends into recent years (up to {test_end_year}). "
                           f"Automatically filtering to exclude games published in the last two years (before {end_year}).")
    
    logger.info(f"Loading data through {end_year}")
    df = loader.load_training_data(end_train_year=end_year + 1, min_ratings=0)
    
    # Get features and target (convert to pandas)
    X = df.drop("hurdle").to_pandas()
    y = df.select("hurdle").to_pandas().squeeze()
    
    # Determine final end year (accounting for potential filtering)
    current_year = datetime.now().year
    final_end_year = end_year if end_year is not None else current_year - 2
    
    # Extract threshold if available
    threshold = extract_model_threshold(experiment)
    
    # Generate description with threshold info if available
    description = generate_model_description(
        base_description=description,
        final_end_year=final_end_year,
        threshold=threshold
    )
    
    # Finalize model
    logger.info("Fitting pipeline on full dataset...")
    finalized_dir = experiment.finalize_model(
        X=X,
        y=y,
        description=description,
        final_end_year=final_end_year
    )
    
    logger.info(f"Model finalized and saved to {finalized_dir}")
    logger.info(f"Final end year for model training: {final_end_year}")
    return finalized_dir

def main():
    parser = argparse.ArgumentParser(description="Finalize model for production")
    parser.add_argument("--model-type", type=str, default="hurdle",
                       help="Type of model to finalize")
    parser.add_argument("--experiment", type=str, required=True,
                       help="Name of experiment to finalize")
    parser.add_argument("--version", type=int,
                       help="Optional specific version to finalize")
    parser.add_argument("--end-year", type=int,
                       help="Optional end year for training data")
    parser.add_argument("--description", type=str,
                       help="Optional description of finalized model")
    
    args = parser.parse_args()
    finalize_model(
        model_type=args.model_type,
        experiment_name=args.experiment,
        version=args.version,
        end_year=args.end_year,
        description=args.description
    )

if __name__ == "__main__":
    main()
