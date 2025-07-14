from __future__ import annotations

import typing as t
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def plot_predictions_scatter(
    predictions_df: pl.DataFrame, 
    color: str | None = 'year_published', 
    ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a scatter plot of predicted vs actual values.

    Args:
        predictions_df: DataFrame containing prediction and actual values
        color: Column to use for color mapping (optional)
        ax: Matplotlib axes to plot on (optional)

    Returns:
        Matplotlib figure and axes
    """
    if 'prediction' not in predictions_df.columns or 'actual' not in predictions_df.columns:
        raise ValueError("DataFrame must contain 'prediction' and 'actual' columns")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    # Prepare scatter plot data
    scatter_kwargs = {
        'alpha': 0.7,
        'edgecolors': 'black',
        'linewidth': 0.5
    }

    if color is not None and color in predictions_df.columns:
        scatter_kwargs.update({
            'c': predictions_df[color],
            'cmap': 'viridis'
        })
        scatter = ax.scatter(
            predictions_df['prediction'],
            predictions_df['actual'],
            **scatter_kwargs
        )
        fig.colorbar(scatter, ax=ax, label=color)
    else:
        scatter = ax.scatter(
            predictions_df['prediction'],
            predictions_df['actual'],
            **scatter_kwargs
        )

    # Perfect prediction line
    min_val = min(predictions_df['prediction'].min(), predictions_df['actual'].min())
    max_val = max(predictions_df['prediction'].max(), predictions_df['actual'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    # Calculate performance metrics
    r2 = r2_score(predictions_df['actual'], predictions_df['prediction'])
    rmse = np.sqrt(mean_squared_error(predictions_df['prediction'], predictions_df['actual']))

    # Styling
    ax.set_title(f'Predictions\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.legend()

    return fig, ax

def plot_predictions_distribution(
    predictions_df: pl.DataFrame, 
    bins: int = 50, 
    title: str = "Distribution of Values", 
    ax: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a histogram comparing prediction and actual value distributions.

    Args:
        predictions_df: DataFrame containing prediction and actual values
        bins: Number of histogram bins
        title: Plot title
        ax: Matplotlib axes to plot on (optional)

    Returns:
        Matplotlib figure and axes
    """
    if 'prediction' not in predictions_df.columns or 'actual' not in predictions_df.columns:
        raise ValueError("DataFrame must contain 'prediction' and 'actual' columns")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.hist(predictions_df['prediction'], bins=bins, alpha=0.5, label='Predictions')
    ax.hist(predictions_df['actual'], bins=bins, alpha=0.5, label='Actuals')

    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

    return fig, ax

def plot_regression_diagnostics(
    predictions_df: pl.DataFrame, 
    color: str | None = None, 
    style: str = 'seaborn-v0_8-darkgrid'
) -> tuple[plt.Figure, t.Sequence[plt.Axes]]:
    """
    Create comprehensive regression diagnostic plots.

    Args:
        predictions_df: DataFrame containing prediction and actual values
        color: Column to use for color mapping (optional)
        style: Matplotlib style to use

    Returns:
        Matplotlib figure and axes
    """
    plt.style.use(style)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot on the left
    plot_predictions_scatter(predictions_df, color, ax=axes[0])

    # Distribution plot on the right
    plot_predictions_distribution(predictions_df, ax=axes[1])

    fig.tight_layout()
    return fig, axes

def main():
    """
    Load predictions and generate diagnostic plots.
    """
    predictions_df = pl.read_parquet('models/experiments/complexity/test-complexity/v1/test_predictions.parquet')
    fig, _ = plot_regression_diagnostics(predictions_df)
    
    # Save the figure
    output_path = 'figures/regression_diagnostics.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
