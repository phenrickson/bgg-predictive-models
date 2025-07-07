import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_complexity_predictions_plot(predictions_path: str, experiment_dir: str):
    """
    Create a scatter plot of actual vs. predicted complexity and save detailed predictions.
    
    Args:
        predictions_path (str): Path to the predictions Parquet file
        experiment_dir (str): Base directory for the experiment outputs
    """
    # Ensure output directories exist
    output_dir = Path(experiment_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read predictions
    predictions_df = pl.read_parquet(predictions_path)
    
    # Convert to pandas for easier manipulation
    df = predictions_df.to_pandas()
    
    # Calculate prediction metrics
    df['prediction_error'] = np.abs(df['true_complexity'] - df['predicted_complexity'])
    df['prediction_error_percentage'] = np.abs(df['prediction_error'] / df['true_complexity']) * 100
    
    # Save detailed predictions
    detailed_predictions_path = output_dir / 'complexity_predictions_detailed.parquet'
    pl.from_pandas(df).write_parquet(detailed_predictions_path)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Scatter plot with color gradient based on year
    scatter = plt.scatter(
        df['true_complexity'], 
        df['predicted_complexity'], 
        c=df['year_published'], 
        cmap='viridis', 
        alpha=0.7,
        edgecolors='black', 
        linewidth=0.5
    )
    plt.colorbar(scatter, label='Year Published')
    
    # Perfect prediction line
    min_val = min(df['true_complexity'].min(), df['predicted_complexity'].min())
    max_val = max(df['true_complexity'].max(), df['predicted_complexity'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Calculate R² and RMSE
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(df['true_complexity'], df['predicted_complexity'])
    rmse = np.sqrt(mean_squared_error(df['true_complexity'], df['predicted_complexity']))
    
    # Annotations
    plt.title(f'Complexity Predictions\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=14)
    plt.xlabel('Actual Complexity', fontsize=12)
    plt.ylabel('Predicted Complexity', fontsize=12)
    plt.legend()
    
    # Tight layout and save
    plt.tight_layout()
    plot_path = output_dir / 'complexity_predictions_scatter.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Scatter plot saved to {plot_path}")
    print(f"Detailed predictions saved to {detailed_predictions_path}")

def main():
    # Find the most recent predictions file
    import glob
    import os
    from pathlib import Path
    
    # Look for predictions in the default experiments directory
    experiments_dir = Path('./models/experiments')
    
    # Find all complexity predictions files recursively
    predictions_files = list(experiments_dir.rglob('*_complexity_predictions.parquet'))
    
    if not predictions_files:
        raise FileNotFoundError("No complexity predictions file found in model experiments directory")
    
    # Get the most recently modified file
    latest_predictions = max(predictions_files, key=os.path.getmtime)
    
    # Determine experiment directory (the parent of the predictions file)
    experiment_dir = latest_predictions.parent
    
    create_complexity_predictions_plot(str(latest_predictions), str(experiment_dir))

if __name__ == "__main__":
    main()
