"""Score new data using the finalized hurdle model."""
import argparse
import polars as pl
from src.models.experiments import ExperimentTracker

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
    from datetime import datetime

    # Load configuration and data loader
    config = load_config()
    loader = BGGDataLoader(config)
    
    # Determine experiment name
    if experiment_name is None:
        # Use the latest hurdle model experiment if not specified
        tracker = ExperimentTracker("hurdle")
        experiments = tracker.list_experiments()
        if not experiments:
            raise ValueError("No hurdle model experiments found.")
        experiment = max(experiments, key=lambda x: x.metadata.get('timestamp', 0))
        experiment_name = experiment.name
    
    # Load pipeline
    pipeline = load_model(experiment_name)
    
    # Load data
    if data_path:
        # Load from CSV if provided
        df = pl.read_csv(data_path)
    else:
        # Determine start and end years for scoring
        if start_year is None or end_year is None:
            # Retrieve the model training date from experiment metadata
            tracker = ExperimentTracker("hurdle")
            experiment = tracker.load_experiment(experiment_name)
            
            # Use the training end year and validation/test window
            train_end_year = experiment.metadata.get('train_end_year', 2022)  # Default to 2022 if not found
            valid_window = experiment.metadata.get('valid_window', 2)
            test_window = experiment.metadata.get('test_window', 2)
            
            # Set start year to the last year of validation
            start_year = train_end_year + valid_window
            
            # Set end year to a future year if not specified
            end_year = datetime.now().year + 5  # 5 years into the future
        
        # Construct where clause for year filtering
        where_clause = [f"year_published > {start_year}"]
        
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
    
    # Preprocess and predict in one step
    predictions = pipeline.predict_proba(df.to_pandas())[:, 1]
    
    # Select only the required columns and add predictions
    results = df.select([
        "game_id", 
        "name", 
        "year_published"
    ]).with_columns([
        pl.Series("predicted_prob", predictions),
        pl.Series("predicted_class", predictions >= 0.5),
        pl.Series("hurdle", df.select("hurdle").to_pandas().squeeze())
    ])
    
    # Determine output path if not provided
    if output_path is None:
        output_path = f"data/predictions/{experiment_name}_predictions.parquet"
    elif not output_path.endswith('.parquet'):
        output_path = output_path.rsplit('.', 1)[0] + '.parquet'
    
    # Dynamically determine end year if not specified
    if end_year is None:
        # Use the current year or a sufficiently far future year
        from datetime import datetime
        end_year = datetime.now().year
    
    # Save results
    results.write_parquet(output_path)
    print(f"Predictions for {experiment_name} saved to {output_path}")
    print(f"Data loaded from year {start_year or 'beginning'} to {end_year}")
    
    # Display sample of results
    print("\nSample predictions:")
    sample_columns = [
        "game_id", "name", "year_published", 
        "predicted_prob", "predicted_class", 
        "hurdle"  # Actual outcome
    ]
    print(results.select(sample_columns).head())

    return results

def main():
    parser = argparse.ArgumentParser(description="Score new data using finalized model")
    parser.add_argument("--data", 
                       help="Optional path to CSV file containing data to score")
    parser.add_argument("--experiment", 
                       help="Name of experiment with finalized model")
    parser.add_argument("--start-year", type=int,
                       help="First year of data to include")
    parser.add_argument("--end-year", type=int,
                       help="Last year of data to include")
    parser.add_argument("--min-ratings", type=int, default=0,
                       help="Minimum number of ratings to filter games")
    parser.add_argument("--output", 
                       help="Path to save predictions")
    
    args = parser.parse_args()
    score_data(
        data_path=args.data,
        experiment_name=args.experiment,
        start_year=args.start_year,
        end_year=args.end_year,
        min_ratings=args.min_ratings,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
