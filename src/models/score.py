"""Score new data using the finalized hurdle model."""
import argparse
import json
import polars as pl
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from src.models.experiments import ExperimentTracker

def get_model_info(finalized_dir: Path) -> Tuple[Optional[float], Optional[int]]:
    """Extract model information from finalized model directory.
    
    Args:
        finalized_dir: Path to finalized model directory
        
    Returns:
        Tuple containing:
        - threshold: Classification threshold if found, None otherwise
        - final_end_year: Final training end year if found, None otherwise
    """
    info_path = finalized_dir / "info.json"
    threshold = None
    final_end_year = None
    
    if info_path.exists():
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
                # Check both possible locations for threshold
                threshold = info.get('model_info', {}).get('threshold')
                if threshold is None:
                    threshold = info.get('threshold')
                final_end_year = info.get('final_end_year')
        except Exception as e:
            print(f"Warning: Error reading model info: {e}")
    
    return threshold, final_end_year

def load_model(experiment_name: str):
    """Load the finalized model and preprocessing pipeline.
    
    Attempts to load the experiment by extracting the model type from the experiment name.
    Supports experiments in different model type directories.
    
    Returns:
        Finalized pipeline
    """
    # Determine model type from experiment name
    model_types = ['hurdle', 'rating', 'complexity', 'users_rated']
    
    # Print diagnostic information
    print(f"Attempting to load experiment: {experiment_name}")
    print(f"Searching in model types: {model_types}")
    
    # Try each model type until successful
    for model_type in model_types:
        try:
            print(f"Trying model type: {model_type}")
            tracker = ExperimentTracker(model_type)
            
            # Print available experiments for this model type
            experiments = tracker.list_experiments()
            print(f"Available experiments for {model_type}: {[exp['full_name'] for exp in experiments]}")
            
            # Handle cases with or without version
            if '/' in experiment_name:
                # If version is specified (e.g., 'hurdle_model/v1')
                base_name, version_str = experiment_name.split('/')
                version = int(version_str[1:])
                experiment = tracker.load_experiment(base_name, version)
            else:
                # If no version specified, load latest
                experiment = tracker.load_experiment(experiment_name)
            
            # Load the finalized model directly
            print(f"Successfully loaded experiment: {experiment.name}")
            return experiment.load_finalized_model()
        except (ValueError, Exception) as e:
            print(f"Failed to load in {model_type} model type: {e}")
            continue
    
    # If no model type works, raise an error
    raise ValueError(f"Could not load experiment '{experiment_name}' in any known model type")

def score_data(
    experiment_name: str = None, 
    data_path: str = None,
    start_year: int = None,
    end_year: int = None, 
    min_ratings: int = 0, 
    output_path: str = None
):
    """Score data using the finalized model.
    
    Args:
        experiment_name: Name of experiment with finalized model (optional, will use latest if not provided)
        data_path: Optional path to a CSV file for scoring (overrides query-based loading)
        start_year: First year of data to include (optional)
        end_year: Last year of data to include (optional)
        min_ratings: Minimum number of ratings to filter games (default 0)
        output_path: Path to save predictions (optional, will use experiment name if not provided)
    """
    from src.data.loader import BGGDataLoader
    from src.data.config import load_config
    from src.models.experiments import ExperimentTracker

    # Load configuration and data loader
    config = load_config()
    loader = BGGDataLoader(config)
    
    # Determine model type
    model_type = "hurdle"  # Default model type
    if experiment_name is not None and '/' in experiment_name:
        # If experiment name includes model type (e.g., 'rating/full-features')
        model_type, experiment_name = experiment_name.split('/')

    # Determine experiment name
    if experiment_name is None:
        # Use the latest model experiment if not specified
        tracker = ExperimentTracker(model_type)
        experiments = tracker.list_experiments()
        if not experiments:
            raise ValueError(f"No {model_type} model experiments found.")
        experiment = max(experiments, key=lambda x: x.metadata.get('timestamp', 0))
        experiment_name = experiment.name
    
    # Load pipeline
    pipeline = load_model(experiment_name)
    
    # Load data
    if data_path:
        # Load from CSV if provided
        df = pl.read_csv(data_path)
    else:
        # Retrieve the finalized model metadata
        tracker = ExperimentTracker("hurdle")
        experiment = tracker.load_experiment(experiment_name)
        
        # Get model info
        finalized_dir = experiment.exp_dir / "finalized"
        _, final_end_year = get_model_info(finalized_dir)
        
        # Determine start and end years for scoring
        if start_year is None:
            start_year = final_end_year + 1 if final_end_year else 0
        
        if end_year is None:
            # Default to 5 years greater than the current year
            end_year = datetime.now().year + 5
        
        # Construct where clause for year filtering
        where_clause = [f"year_published >= {start_year}"]
        
        # Add end year filtering if specified
        if end_year is not None:
            where_clause.append(f"year_published <= {end_year}")
        
        # Combine where clauses
        where_str = " AND ".join(where_clause)
        
        # Load data with optional filtering
        df = loader.load_data(
            where_clause=where_str,
            preprocessor=None
        )
    
    # Get model threshold from finalized model info
    tracker = ExperimentTracker("hurdle")
    experiment = tracker.load_experiment(experiment_name)
    finalized_dir = experiment.exp_dir / "finalized"
    threshold, _ = get_model_info(finalized_dir)
    
    # Use default threshold of 0.5 if none found
    threshold = threshold if threshold is not None else 0.5
    print(f"Using classification threshold: {threshold}")
    
    # Preprocess and predict in one step
    predictions = pipeline.predict_proba(df.to_pandas())[:, 1]
    
    # Select only the required columns and add predictions
    results = df.select([
        "game_id", 
        "name", 
        "year_published"
    ]).with_columns([
        pl.Series("predicted_prob", predictions),
        pl.Series("predicted_class", predictions >= threshold),
        pl.Series("hurdle", df.select("hurdle").to_pandas().squeeze()),
        pl.Series("threshold", [threshold] * len(df))  # Add threshold used
    ])
    
    # Determine output path if not provided
    if output_path is None:
        output_path = f"data/predictions/{experiment_name}_predictions.parquet"
    elif not output_path.endswith('.parquet'):
        output_path = output_path.rsplit('.', 1)[0] + '.parquet'
    
    # Save results
    results.write_parquet(output_path)
    print(f"Predictions for {experiment_name} saved to {output_path}")
    print(f"Data loaded from year {start_year or 'beginning'} to {end_year}")
    
    # Display sample of results
    print("\nSample predictions:")
    sample_columns = [
        "game_id", "name", "year_published", 
        "predicted_prob", "predicted_class", 
        "hurdle",  # Actual outcome
        "threshold"  # Classification threshold used
    ]
    print(results.select(sample_columns).head())

    return results

def main():
    parser = argparse.ArgumentParser(description="Score new data using finalized model")
    parser.add_argument("--data", 
                       help="Optional path to CSV file containing data to score")
    parser.add_argument("--experiment", 
                       help="Name of experiment with finalized model. Can include model type (e.g., 'rating/full-features')")
    parser.add_argument("--model-type", default="hurdle",
                       help="Model type directory to search for experiments (default: hurdle)")
    parser.add_argument("--start-year", type=int,
                       help="First year of data to include")
    parser.add_argument("--end-year", type=int,
                       help="Last year of data to include")
    parser.add_argument("--min-ratings", type=int, default=0,
                       help="Minimum number of ratings to filter games")
    parser.add_argument("--output", 
                       help="Path to save predictions")
    
    args = parser.parse_args()
    
    # If experiment includes model type, override model-type argument
    if args.experiment and '/' in args.experiment:
        model_type, experiment_name = args.experiment.split('/')
    else:
        model_type = args.model_type
        experiment_name = args.experiment
    
    score_data(
        data_path=args.data,
        experiment_name=experiment_name,
        start_year=args.start_year,
        end_year=args.end_year,
        min_ratings=args.min_ratings,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
