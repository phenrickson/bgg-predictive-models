"""Script to view model predictions stored in parquet files."""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_analyze_predictions(experiment_path: str, dataset: str = "test"):
    """Load and analyze predictions from an experiment.
    
    Args:
        experiment_path: Path to experiment directory
        dataset: Dataset to analyze (e.g., 'test', 'tune')
    """
    predictions_file = Path(experiment_path) / f"{dataset}_predictions.parquet"
    if not predictions_file.exists():
        raise ValueError(f"No predictions found at {predictions_file}")
    
    # Load predictions
    df = pl.read_parquet(predictions_file)
    
    # Basic information
    print("\nDataset Information:")
    print(f"Number of predictions: {len(df)}")
    print("\nColumns available:")
    print(df.columns)
    
    # Get prediction and actual columns
    predictions = df.get_column("prediction")
    actuals = df.get_column("actual")
    
    # # Calculate error metrics
    # mae = (predictions - actuals).abs().mean()
    # mse = ((predictions - actuals) ** 2).mean()
    # rmse = mse.sqrt()
    
    # print("\nError Metrics:")
    # print(f"MAE:  {mae:.4f}")
    # print(f"RMSE: {rmse:.4f}")
    
    # Summary statistics
    print("\nPrediction Summary:")
    summary = df.select([
        pl.col("prediction").mean().alias("mean_pred"),
        pl.col("prediction").std().alias("std_pred"),
        pl.col("prediction").min().alias("min_pred"),
        pl.col("prediction").max().alias("max_pred"),
        pl.col("actual").mean().alias("mean_actual"),
        pl.col("actual").std().alias("std_actual"),
        pl.col("actual").min().alias("min_actual"),
        pl.col("actual").max().alias("max_actual"),
    ])
    
    # for col in summary.columns:
    #     print(f"{col}: {summary[0][col]:.4f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Scatter plot with improved visualization
    plt.subplot(131)
    scatter = plt.scatter(actuals, predictions, 
                         alpha=0.5,
                         c=abs(predictions - actuals),  # Color by prediction error
                         cmap='viridis',
                         s=50,  # Point size
                         edgecolors='white',
                         linewidth=0.5)
    plt.colorbar(scatter, label='Absolute Error')
    
    # Add perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', label='Perfect Prediction')
    
    # Add trend line
    z = np.polyfit(actuals, predictions, 1)
    p = np.poly1d(z)
    plt.plot(actuals, p(actuals), 'b-', alpha=0.8,
             label=f'Trend Line (y={z[0]:.2f}x+{z[1]:.2f})')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actuals\nColored by Prediction Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution of predictions
    plt.subplot(132)
    plt.hist(predictions, bins=50, alpha=0.5, label='Predictions')
    plt.hist(actuals, bins=50, alpha=0.5, label='Actuals')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Distribution of Values')
    plt.legend()
    
    # # Error distribution
    # errors = predictions - actuals
    # plt.subplot(133)
    # plt.hist(errors, bins=50)
    # plt.xlabel('Prediction Error')
    # plt.ylabel('Count')
    # plt.title('Error Distribution')
    
    plt.tight_layout()
    plt.savefig('figures/prediction_analysis.png')
    print("\nSaved visualizations to 'prediction_analysis.png'")
    
    # # Show worst predictions
    # print("\nWorst Predictions:")
    # df = df.with_columns(pl.col("prediction") - pl.col("actual").alias("error").abs())
    # worst_predictions = df.sort("error", descending=True).head(10)
    # print(worst_predictions)

if __name__ == "__main__":
    # Example usage
    experiment_path = "models/experiments/complexity/lightgbm-complexity/v1"
    load_and_analyze_predictions(experiment_path, "test")
