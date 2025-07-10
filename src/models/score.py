"""Score new data using the finalized hurdle model."""
import argparse
import json
import numpy as np
import polars as pl
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from src.models.experiments import ExperimentTracker

def get_model_info(finalized_dir: Path) -> Optional[int]:
    """Extract final end year from finalized model directory.
    
    Args:
        finalized_dir: Path to finalized model directory
        
    Returns:
        Final training end year if found, None otherwise
    """
    info_path = finalized_dir / "info.json"
    final_end_year = None
    
    if info_path.exists():
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
                final_end_year = info.get('final_end_year')
        except Exception as e:
            print(f"Warning: Error reading model info: {e}")
    
    return final_end_year

def extract_threshold(
    experiment_name: str, 
    model_type: str
) -> Optional[float]:
    """Extract threshold from the most recent version's metadata or model_info.json file.
    
    Args:
        experiment_name: Name of the experiment
        model_type: Type of model
        
    Returns:
        Threshold value if found, None otherwise
    """
    from src.models.experiments import ExperimentTracker
    
    # Get experiment tracker for the model type
    tracker = ExperimentTracker(model_type)
    
    # Find the latest version of the experiment
    experiments = tracker.list_experiments()
    matching_experiments = [
        exp for exp in experiments 
        if exp['name'] == experiment_name
    ]
    
    if not matching_experiments:
        print(f"No experiments found matching {experiment_name}")
        return None
    
    # Get the latest version
    latest_experiment = max(
        matching_experiments, 
        key=lambda x: x['version']
    )
    
    # Load the experiment with the latest version
    experiment = tracker.load_experiment(
        latest_experiment['name'], 
        latest_experiment['version']
    )
    
    # First, check metadata.json for optimal_threshold
    metadata_path = experiment.exp_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                threshold = metadata.get('metadata', {}).get('optimal_threshold')
                
                if threshold is not None:
                    print(f"Found threshold {threshold} in {metadata_path}")
                    return threshold
        except Exception as e:
            print(f"Warning: Error reading {metadata_path}: {e}")
    
    # Then, look for model_info.json in the experiment directory
    model_info_path = experiment.exp_dir / "model_info.json"
    
    if model_info_path.exists():
        try:
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                threshold = model_info.get('threshold')
                
                if threshold is not None:
                    print(f"Found threshold {threshold} in {model_info_path}")
                    return threshold
        except Exception as e:
            print(f"Warning: Error reading {model_info_path}: {e}")
    
    # If no threshold found
    print("No threshold found in metadata.json or model_info.json")
    return None

def load_model(experiment_name: str, model_type: Optional[str] = None):
    """Load the finalized model and preprocessing pipeline.
    
    Attempts to load the experiment by extracting the model type from the experiment name.
    Supports experiments in different model type directories and more flexible path handling.
    
    Args:
        experiment_name: Name of the experiment to load
        model_type: Optional model type to restrict the search
    
    Returns:
        Finalized pipeline
    """
    # Determine model types to search
    if model_type:
        # If model_type is provided, only search in that type
        model_types = [model_type]
    else:
        # Otherwise, search in all known model types
        model_types = ['hurdle', 'rating', 'complexity', 'users_rated']
    
    # Print diagnostic information
    print(f"Attempting to load experiment: {experiment_name}")
    print(f"Searching in model types: {model_types}")
    
    # Try each model type until successful
    for current_model_type in model_types:
        try:
            print(f"Trying model type: {current_model_type}")
            tracker = ExperimentTracker(current_model_type)
            
            # Print available experiments for this model type
            experiments = tracker.list_experiments()
            print(f"Available experiments for {current_model_type}: {[exp['full_name'] for exp in experiments]}")
            
            # Handle cases with or without version
            if '/' in experiment_name:
                # If version is specified (e.g., 'hurdle_model/v1')
                base_name, version_str = experiment_name.split('/')
                version = int(version_str[1:])
                experiment = tracker.load_experiment(base_name, version)
            else:
                # If no version specified, find matching experiments
                matching_experiments = [
                    exp for exp in experiments 
                    if exp['name'] == experiment_name
                ]
                
                if not matching_experiments:
                    print(f"No experiments found matching base name: {experiment_name}")
                    continue
                
                # Sort and get the latest version
                latest_experiment = max(
                    matching_experiments, 
                    key=lambda x: x['version']
                )
                
                print(f"Auto-selecting latest version: {latest_experiment['full_name']}")
                experiment = tracker.load_experiment(
                    latest_experiment['name'], 
                    latest_experiment['version']
                )
            
            # Print experiment directory for debugging
            print(f"Experiment directory: {experiment.exp_dir}")
            
            # Explicitly look for finalized model
            finalized_path = experiment.exp_dir / "finalized" / "pipeline.pkl"
            print(f"Checking finalized model path: {finalized_path}")
            print(f"Path exists: {finalized_path.exists()}")
            
            if not finalized_path.exists():
                # Look for latest version's finalized model
                version_dirs = [
                    d for d in experiment.exp_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('v')
                ]
                print(f"Version directories found: {[d.name for d in version_dirs]}")
                
                if version_dirs:
                    latest_version_dir = max(
                        version_dirs,
                        key=lambda x: int(x.name[1:])
                    )
                    finalized_path = latest_version_dir / "finalized" / "pipeline.pkl"
                    print(f"Checking alternative finalized model path: {finalized_path}")
                    print(f"Alternative path exists: {finalized_path.exists()}")
            
            if finalized_path.exists():
                print(f"Successfully loaded finalized model from: {finalized_path}")
                import joblib
                return joblib.load(finalized_path)
            
            raise FileNotFoundError(f"No finalized model found for {experiment_name}")
        
        except (ValueError, FileNotFoundError, Exception) as e:
            print(f"Failed to load in {current_model_type} model type: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # If no model type works, raise an error
    raise ValueError(f"Could not load experiment '{experiment_name}' in any known model type")

def load_scoring_data(
    data_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    model_type: str = "hurdle",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pl.DataFrame:
    """
    Load data for scoring based on provided parameters.
    
    Args:
        data_path: Optional path to a CSV file for scoring
        experiment_name: Name of experiment to determine data loading parameters
        model_type: Type of model being used
        start_year: First year of data to include
        end_year: Last year of data to include
    
    Returns:
        Polars DataFrame with data to be scored
    """
    from src.data.loader import BGGDataLoader
    from src.data.config import load_config
    from src.models.experiments import ExperimentTracker

    # Load configuration and data loader
    config = load_config()
    loader = BGGDataLoader(config)
    
    # If data path is provided, load directly from CSV
    if data_path:
        return pl.read_csv(data_path)
    
    # Retrieve the finalized model metadata
    tracker = ExperimentTracker(model_type)
    
    # Find the latest version of the experiment
    experiments = tracker.list_experiments()
    matching_experiments = [
        exp for exp in experiments 
        if exp['name'] == experiment_name
    ]
    if not matching_experiments:
        raise ValueError(f"No experiments found matching {experiment_name}")
    
    latest_experiment = max(
        matching_experiments, 
        key=lambda x: x['version']
    )
    
    # Load the experiment with the latest version
    experiment = tracker.load_experiment(
        latest_experiment['name'], 
        latest_experiment['version']
    )
    
    # Get model info
    finalized_dir = experiment.exp_dir / "finalized"
    final_end_year = get_model_info(finalized_dir)
    
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
    return loader.load_data(
        where_clause=where_str,
        preprocessor=None
    )

def predict_data(
    pipeline,
    df: pl.DataFrame,
    experiment_name: str,
    model_type: str = "hurdle"
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Predict data using the given pipeline and model type.
    
    Args:
        pipeline: Trained model pipeline
        df: Input DataFrame to predict
        experiment_name: Name of experiment for threshold retrieval
        model_type: Type of model being used
    
    Returns:
        Tuple of (predicted_prob, predicted_class, threshold)
    """
    import numpy as np
    from src.models.experiments import ExperimentTracker

    # Predict based on model type
    if model_type == "hurdle":
        # Use predict_proba for hurdle model
        predictions = pipeline.predict_proba(df.to_pandas())[:, 1]
        
        # Try to extract threshold from the experiment
        threshold = extract_threshold(experiment_name, model_type)
        
        # Use default threshold of 0.5 if none found
        threshold = threshold if threshold is not None else 0.5
        print(f"Using classification threshold: {threshold}")
        
        predicted_class = predictions >= threshold
        predicted_prob = predictions
    elif model_type == "complexity":
        # Diagnostic logging for complexity model
        print("Complexity Model Prediction Diagnostics:")

        # Convert to pandas for prediction
        df_pandas = df.to_pandas()
        
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Input columns: {df.columns}")
        
        # Print first few rows of input data
        print("First few rows of input data:")
        print(df_pandas.head())
        
        # Predict and log details
        predictions = pipeline.predict(df_pandas)
        print(f"Raw predictions: {predictions}")
        print(f"Prediction stats: min={predictions.min()}, max={predictions.max()}, mean={predictions.mean()}")
        
        # Constrain predictions to 1-5 range
        predicted_class = np.clip(predictions, 1, 5)
        print(f"Constrained predictions: {predicted_class}")
        
        predicted_prob = predicted_class  # For complexity, prob is the same as prediction
        threshold = None  # No threshold for complexity
    elif model_type == "users_rated":
        # Regression outcome for users_rated
        predictions = pipeline.predict(df.to_pandas())
        predicted_class = predictions
        predicted_prob = predictions
        threshold = None  # No threshold for regression
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return predicted_prob, predicted_class, threshold

def prepare_results(
    df: pl.DataFrame, 
    predicted_prob: np.ndarray, 
    predicted_class: np.ndarray, 
    model_type: str,
    threshold: Optional[float] = None
) -> pl.DataFrame:
    """
    Prepare results DataFrame based on model type.
    
    Args:
        df: Original input DataFrame
        predicted_prob: Predicted probabilities
        predicted_class: Predicted classes
        model_type: Type of model being used
        threshold: Optional threshold used for classification
    
    Returns:
        Results DataFrame with predictions
    """
    # Select columns based on model type
    if model_type == "complexity":
        results = df.select([
            "game_id", 
            "name", 
            "year_published"
        ]).with_columns([
            pl.Series("predicted_complexity", predicted_class),
            pl.Series("complexity", df.select("complexity").to_pandas().squeeze())
        ])
    else:
        # Existing logic for other model types
        results = df.select([
            "game_id", 
            "name", 
            "year_published"
        ]).with_columns([
            pl.Series("predicted_prob", predicted_prob),
            pl.Series("predicted_class", predicted_class),
            pl.Series("hurdle", df.select("hurdle").to_pandas().squeeze()),
            pl.Series("threshold", [threshold] * len(df) if threshold is not None else [None] * len(df))  # Add threshold used
        ])
    
    return results

def save_and_display_results(
    results: pl.DataFrame, 
    experiment_name: str, 
    start_year: Optional[int], 
    end_year: Optional[int], 
    model_type: str,
    output_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Save and display prediction results.
    
    Args:
        results: Results DataFrame with predictions
        experiment_name: Name of experiment
        start_year: Start year of data
        end_year: End year of data
        model_type: Type of model being used
        output_path: Optional path to save predictions
    
    Returns:
        Results DataFrame
    """
    # Determine output path
    from pathlib import Path
    
    # Create base predictions directory for the model type
    base_predictions_dir = Path("data/predictions") / model_type
    base_predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output path if not provided
    if output_path is None:
        # Create a filename that includes just the experiment name
        output_filename = f"{experiment_name}_predictions.parquet"
        output_path = base_predictions_dir / output_filename
    else:
        # Ensure output path is within the predictions directory
        output_path = Path(output_path)
        if not output_path.suffix == '.parquet':
            output_path = output_path.with_suffix('.parquet')
        
        # Ensure it's in the correct model type directory
        output_path = base_predictions_dir / output_path.name
    
    # Save results
    results.write_parquet(str(output_path))
    print(f"Predictions for {experiment_name} saved to {output_path}")
    print(f"Data loaded from year {start_year or 'beginning'} to {end_year}")
    
    # Display sample of results
    print("\nSample predictions:")
    
    # Dynamically select sample columns based on model type
    if model_type == "complexity":
        sample_columns = [
            "game_id", "name", "year_published", 
            "predicted_complexity", "complexity"
        ]
    elif model_type == "hurdle":
        sample_columns = [
            "game_id", "name", "year_published", 
            "predicted_prob", "predicted_class", 
            "hurdle", "threshold"
        ]
    elif model_type == "users_rated":
        sample_columns = [
            "game_id", "name", "year_published", 
            "predicted_class"  # For regression, predicted_class is the prediction
        ]
    else:
        # Fallback to default columns
        sample_columns = [
            "game_id", "name", "year_published", 
            "predicted_prob", "predicted_class"
        ]
    
    print(results.select(sample_columns).head())

    return results

def score_data(
    experiment_name: str = None, 
    data_path: str = None,
    start_year: int = None,
    end_year: int = None, 
    min_ratings: int = 0, 
    output_path: str = None,
    model_type: str = "hurdle"
):
    """Score data using the finalized model.
    
    Args:
        experiment_name: Name of experiment with finalized model (optional, will use latest if not provided)
        data_path: Optional path to a CSV file for scoring (overrides query-based loading)
        start_year: First year of data to include (optional)
        end_year: Last year of data to include (optional)
        min_ratings: Minimum number of ratings to filter games (default 0)
        output_path: Path to save predictions (optional, will use experiment name if not provided)
        model_type: Type of model to use (default: hurdle)
    """
    # Handle case where model type is included in experiment name
    if experiment_name is not None and '/' in experiment_name:
        # If experiment name includes model type (e.g., 'rating/full-features')
        model_type, experiment_name = experiment_name.split('/')

    # Determine experiment name if not provided
    if experiment_name is None:
        from src.models.experiments import ExperimentTracker
        tracker = ExperimentTracker(model_type)
        experiments = tracker.list_experiments()
        if not experiments:
            raise ValueError(f"No {model_type} model experiments found.")
        experiment = max(experiments, key=lambda x: x.get('version', 0))
        experiment_name = experiment['name']
    
    # Print debug information about model type
    print(f"Using model type: {model_type}")

    # Load pipeline
    pipeline = load_model(experiment_name)
    
    # Load data
    df = load_scoring_data(
        data_path=data_path,
        experiment_name=experiment_name,
        model_type=model_type,
        start_year=start_year,
        end_year=end_year
    )
    
    # Predict data
    predicted_prob, predicted_class, threshold = predict_data(
        pipeline, 
        df, 
        experiment_name,
        model_type=model_type
    )
    
    # Prepare results
    results = prepare_results(
        df, 
        predicted_prob, 
        predicted_class, 
        model_type,
        threshold
    )
    
    # Save and display results
    return save_and_display_results(
        results, 
        experiment_name, 
        start_year, 
        end_year, 
        model_type,
        output_path
    )

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
        output_path=args.output,
        model_type=model_type
    )

if __name__ == "__main__":
    main()
