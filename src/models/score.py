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
    
    # Try each model type until successful
    for model_type in model_types:
        try:
            tracker = ExperimentTracker(model_type)
            experiment = tracker.load_experiment(experiment_name)
            
            # Load the finalized model directly
            return experiment.load_finalized_model()
        except (ValueError, Exception):
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
        # Construct where clause for year filtering
        where_clause = []
        if start_year is not None:
            where_clause.append(f"year_published >= {start_year}")
        if end_year is not None:
            where_clause.append(f"year_published <= {end_year}")
        
        # Combine where clauses
        where_str = " AND ".join(where_clause) if where_clause else None
        
        # Load data with optional filtering
        df = loader.load_data(
            where_clause=where_str,
            preprocessor=None
        )
    
    # Preprocess and predict in one step
    predictions = pipeline.predict_proba(df.to_pandas())[:, 1]
    
    # Add predictions to the dataframe
    results = df.with_columns([
        pl.Series("predicted_prob", predictions),
        pl.Series("predicted_class", predictions >= 0.5)
    ])
    
    # Determine output path if not provided
    if output_path is None:
        output_path = f"data/predictions/{experiment_name}_predictions.csv"
    
    # Save results
    results.write_parquet(output_path)
    print(f"Predictions for {experiment_name} saved to {output_path}")
    print(f"Data loaded from year {start_year or 'beginning'} to {end_year or 'present'}")
    
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
