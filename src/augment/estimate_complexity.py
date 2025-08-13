"""Estimate complexity for board games using a trained complexity model."""
import argparse
import numpy as np
import polars as pl
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.score import load_model, load_scoring_data, predict_data
from src.models.experiments import ExperimentTracker

def prepare_complexity_results(
    df: pl.DataFrame, 
    predicted_complexity: np.ndarray, 
    experiment_name: str
) -> pl.DataFrame:
    """
    Prepare results DataFrame for complexity predictions.
    
    Args:
        df: Original input DataFrame
        predicted_complexity: Predicted complexity values
        experiment_name: Name of the experiment used for predictions
    
    Returns:
        Results DataFrame with predictions and metadata
    """
    # Current timestamp for load_ts
    load_ts = datetime.now()
    
    # Get experiment details for score_ts
    tracker = ExperimentTracker('complexity')
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
    
    # Use experiment creation time as score_ts
    score_ts = datetime.fromtimestamp(latest_experiment.get('created_at', datetime.now().timestamp()))
    
    # Prepare results DataFrame
    results = df.select([
        "game_id", 
        "name", 
        "year_published"
    ]).with_columns([
        pl.Series("predicted_complexity", predicted_complexity),
        pl.Series("model_id", [experiment_name] * len(df)),
        pl.Series("score_ts", [score_ts] * len(df)),
        pl.Series("load_ts", [load_ts] * len(df))
    ])
    
    # Optionally include original complexity if available
    if "complexity" in df.columns:
        results = results.with_columns(
            pl.Series("original_complexity", df.select("complexity").to_pandas().squeeze())
        )
    
    return results

def generate_complexity_predictions(
    experiment_name: Optional[str] = None, 
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    output_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Generate complexity predictions for all games.
    
    Args:
        experiment_name: Name of complexity experiment (optional)
        start_year: First year of data to include (optional)
        end_year: Last year of data to include (optional)
        output_path: Path to save predictions (optional)
    
    Returns:
        DataFrame with complexity predictions
    """
    # Determine experiment name if not provided
    if experiment_name is None:
        tracker = ExperimentTracker('complexity')
        experiments = tracker.list_experiments()
        if not experiments:
            raise ValueError("No complexity model experiments found.")
        experiment = max(experiments, key=lambda x: x.get('version', 0))
        experiment_name = experiment['name']
    
    print(f"Using complexity experiment: {experiment_name}")

    # Load pipeline
    pipeline = load_model(experiment_name, model_type='complexity')
    
    # Load data
    from src.data.loader import BGGDataLoader
    from src.data.config import load_config
    
    config = load_config()
    loader = BGGDataLoader(config)
    
    # Load all games with non-null year_published
    df = loader.load_data(
        where_clause="year_published IS NOT NULL",
        preprocessor=None
    )
    
    # Predict data
    _, predicted_complexity, _ = predict_data(
        pipeline, 
        df, 
        experiment_name,
        model_type='complexity'
    )
    
    # Prepare results
    results = prepare_complexity_results(
        df, 
        predicted_complexity, 
        experiment_name
    )
    
    # Determine output path
    if output_path is None:
        base_predictions_dir = Path("data/estimates")
        base_predictions_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_predictions_dir / f"{experiment_name}_complexity_predictions.parquet"
    
    # Save results
    results.write_parquet(str(output_path))
    print(f"Complexity predictions saved to {output_path}")
    
    # Display sample of results
    print("\nSample predictions:")
    print(results.head())
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate Complexity Predictions")
    parser.add_argument("--experiment", 
                        help="Name of complexity experiment (default: latest)")
    parser.add_argument("--start-year", type=int,
                        help="First year of data to include")
    parser.add_argument("--end-year", type=int,
                        help="Last year of data to include")
    parser.add_argument("--output", 
                        help="Path to save predictions parquet file")
    
    args = parser.parse_args()
    
    generate_complexity_predictions(
        experiment_name=args.experiment,
        start_year=args.start_year,
        end_year=args.end_year,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
