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
            ('model_info', 'min_weights'),  # From log_experiment
            ('min_weights',),               # Direct metadata key
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

def load_data(
    config: dict, 
    loader: BGGDataLoader, 
    end_year: Optional[int], 
    min_ratings: Optional[float],
    min_weights: Optional[float],
    logger: logging.Logger
) -> tuple:
    """
    Load training data with specified parameters.
    
    Args:
        config: Configuration dictionary
        loader: BGGDataLoader instance
        end_year: Year to filter data up to
        min_weights: Minimum weights filter
        logger: Logging instance
    
    Returns:
        Tuple of (dataframe, final_end_year)
    """
    current_year = datetime.now().year
    
    # Dynamically calculate end year, excluding last two years
    if end_year is None:
        end_year = current_year - 2
    
    logger.info(f"Loading data through {end_year}")
    
    # Diagnostic logging for data loading parameters
    logger.info("Data Loading Parameters:")
    logger.info(f"  End Train Year: {end_year + 1}")
    logger.info(f"  Minimum Ratings: {min_ratings}")
    logger.info(f"  Minimum Weights: {min_weights}")
    
    df = loader.load_training_data(
        end_train_year=end_year + 1, 
        min_ratings=min_ratings,
        min_weights=min_weights
    )
    
    # Detailed data diagnostics
    logger.info("Data Loading Diagnostics:")
    logger.info(f"  Total Rows: {len(df)}")
    logger.info(f"  Year Range: {df['year_published'].min()} - {df['year_published'].max()}")
    logger.info(f"  Complexity Range: {df['complexity'].min():.2f} - {df['complexity'].max():.2f}")
    logger.info(f"  Complexity Mean: {df['complexity'].mean():.2f}")
    logger.info(f"  Complexity Median: {df['complexity'].median():.2f}")
    
    # Sample row diagnostics
    logger.info("\nSample Row Diagnostics:")
    sample_row = df.head(1)
    for col in sample_row.columns:
        logger.info(f"  {col}: {sample_row[col].to_pandas().squeeze()}")
    
    return df, end_year

def prepare_data(df, model_type='hurdle'):
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
        'hurdle': 'hurdle',
        'complexity': 'complexity',
        'rating': 'rating',
        'users_rated': 'log_users_rated'
    }
    
    # Get target column, default to 'hurdle' if not specified
    target_column = target_columns.get(model_type, 'hurdle')
    
    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found for model type '{model_type}'")
    
    # Get features and target (convert to pandas)
    X = df.drop(target_column).to_pandas()
    y = df.select(target_column).to_pandas().squeeze()
    
    return X, y

def get_model_parameters(
    experiment: Experiment, 
    end_year: int, 
    description: Optional[str] = None
) -> tuple:
    """
    Extract model parameters from experiment metadata.
    
    Args:
        experiment: Experiment object
        end_year: End year for training data
        description: Optional base description
    
    Returns:
        Tuple of (description, threshold, min_weights)
    """
    # Extract threshold if available
    threshold = extract_model_threshold(experiment)
    min_weights = extract_min_weights(experiment)
    
    # Generate description with threshold and min_weights info
    description_parts = [
        description or f"Production model trained on data through {end_year}"
    ]
    
    if threshold is not None:
        description_parts.append(f"Optimal classification threshold: {threshold:.4f}")
    
    if min_weights is not None:
        description_parts.append(f"Minimum weights filter: {min_weights}")
    
    description = ". ".join(description_parts)
    
    return description, threshold, min_weights

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
    # Setup logging
    logger = setup_logging()
    
    # Load experiment
    tracker = ExperimentTracker(model_type)
    
    # Check if experiment name includes version/hash
    if '_v' in experiment_name:
        # Extract base name from full name
        experiment_name = experiment_name.split('_v')[0]
    
    experiment = tracker.load_experiment(experiment_name, version)
    
    # Load configuration and data loader
    config = load_config()
    loader = BGGDataLoader(config)
    
    # Extract min_weights if available
    min_weights = extract_min_weights(experiment)
    
    # Load data
    df, final_end_year = load_data(
        config=config, 
        loader=loader, 
        end_year=end_year, 
        min_ratings = 0,
        min_weights=min_weights,
        logger=logger
    )
    
    # Prepare data with model-specific target column
    X, y = prepare_data(df, model_type)
    
    # Get model parameters
    description, threshold, min_weights = get_model_parameters(
        experiment=experiment, 
        end_year=final_end_year, 
        description=description
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
