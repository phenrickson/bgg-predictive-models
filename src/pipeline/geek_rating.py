"""Geek rating prediction entry point.

Usage:
    uv run -m src.pipeline.geek_rating \
        --hurdle lightgbm-hurdle \
        --complexity catboost-complexity \
        --rating catboost-rating \
        --users-rated ridge-users_rated \
        --start-year 2024 \
        --end-year 2026
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.models.experiments import ExperimentTracker
from src.models.outcomes.geek_rating import GeekRatingModel
from src.models.outcomes.base import DataConfig
from src.models.outcomes.data import load_data

logger = setup_logging()


def create_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    metrics: dict,
) -> None:
    """Create regression diagnostic plots.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        output_dir: Directory to save plots.
        metrics: Dictionary of computed metrics for annotation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Predicted vs Actual scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.3, s=10, c="steelblue")

    # Add diagonal line (perfect predictions)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")

    ax1.set_xlabel("Actual Geek Rating")
    ax1.set_ylabel("Predicted Geek Rating")
    ax1.set_title("Predicted vs Actual Geek Rating")
    ax1.legend()

    # Add metrics annotation
    textstr = f"RMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\nR²: {metrics['r2']:.4f}"
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Residuals vs Predicted
    ax2 = axes[0, 1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.3, s=10, c="steelblue")
    ax2.axhline(y=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Predicted Geek Rating")
    ax2.set_ylabel("Residual (Actual - Predicted)")
    ax2.set_title("Residuals vs Predicted")

    # 3. Histogram of residuals
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax3.axvline(x=0, color="r", linestyle="--", lw=2)
    ax3.set_xlabel("Residual (Actual - Predicted)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Distribution of Residuals")

    # Add residual stats
    res_mean = residuals.mean()
    res_std = residuals.std()
    textstr = f"Mean: {res_mean:.4f}\nStd: {res_std:.4f}"
    ax3.text(0.95, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Distribution comparison (actual vs predicted)
    ax4 = axes[1, 1]
    ax4.hist(y_true, bins=50, alpha=0.5, label="Actual", color="green", edgecolor="black")
    ax4.hist(y_pred, bins=50, alpha=0.5, label="Predicted", color="steelblue", edgecolor="black")
    ax4.set_xlabel("Geek Rating")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution: Actual vs Predicted")
    ax4.legend()

    plt.tight_layout()

    # Save figure
    plot_path = output_dir / "regression_diagnostics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved diagnostic plots to: {plot_path}")

    # Create additional plot: Predicted vs Actual by rating bucket
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Bin actuals and compute mean predicted for each bin
    bins = np.arange(3.5, 9.0, 0.25)
    bin_indices = np.digitize(y_true, bins)

    bin_means_actual = []
    bin_means_pred = []
    bin_counts = []

    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means_actual.append((bins[i-1] + bins[i]) / 2)
            bin_means_pred.append(y_pred[mask].mean())
            bin_counts.append(mask.sum())

    bin_means_actual = np.array(bin_means_actual)
    bin_means_pred = np.array(bin_means_pred)

    # Plot with size proportional to count
    sizes = np.array(bin_counts) / max(bin_counts) * 200 + 20
    ax.scatter(bin_means_actual, bin_means_pred, s=sizes, alpha=0.7, c="steelblue", edgecolors="black")

    # Perfect prediction line
    ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], "r--", lw=2, label="Perfect prediction")

    ax.set_xlabel("Actual Geek Rating (binned)")
    ax.set_ylabel("Mean Predicted Geek Rating")
    ax.set_title("Calibration Plot: Mean Predicted vs Actual (by rating bucket)")
    ax.legend()

    # Save calibration plot
    calib_path = output_dir / "calibration_plot.png"
    plt.savefig(calib_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved calibration plot to: {calib_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    config = load_config()

    # Get default experiment names from config
    hurdle_default = getattr(config.models.get("hurdle"), "experiment_name", None)
    complexity_default = getattr(config.models.get("complexity"), "experiment_name", None)
    rating_default = getattr(config.models.get("rating"), "experiment_name", None)
    users_rated_default = getattr(config.models.get("users_rated"), "experiment_name", None)

    # Construct default complexity predictions path from config
    predictions_dir = config.models.get("predictions_dir", "./models/experiments/predictions")
    complexity_predictions_default = None
    if complexity_default:
        complexity_predictions_default = f"{predictions_dir}/{complexity_default}.parquet"

    # Get default scoring years from config
    score_start_year = config.years.score.start
    score_end_year = config.years.score.end

    parser = argparse.ArgumentParser(description="Predict geek ratings for board games")

    # Model experiment arguments - now with config defaults
    parser.add_argument(
        "--hurdle",
        default=hurdle_default,
        help=f"Experiment name for hurdle model (default from config: {hurdle_default})",
    )
    parser.add_argument(
        "--complexity",
        default=complexity_default,
        help=f"Experiment name for complexity model (default from config: {complexity_default})",
    )
    parser.add_argument(
        "--rating",
        default=rating_default,
        help=f"Experiment name for rating model (default from config: {rating_default})",
    )
    parser.add_argument(
        "--users-rated",
        default=users_rated_default,
        help=f"Experiment name for users rated model (default from config: {users_rated_default})",
    )

    # Data arguments
    parser.add_argument(
        "--local-complexity-path",
        type=str,
        default=complexity_predictions_default,
        help="Path to local complexity predictions parquet file",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=score_start_year,
        help=f"Start year for filtering games (default from config: {score_start_year})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=score_end_year,
        help=f"End year for filtering games (default from config: {score_end_year})",
    )

    # Prediction parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold for hurdle model (default: from experiment)",
    )
    parser.add_argument(
        "--prior-rating",
        type=float,
        default=5.5,
        help="Prior mean rating for Bayesian average (default: 5.5)",
    )
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=2000,
        help="Weight given to prior rating (default: 2000)",
    )

    # Output arguments
    parser.add_argument(
        "--experiment",
        default="geek_rating_prediction",
        help="Name of the experiment for tracking",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/predictions",
        help="Base directory for output files",
    )

    return parser.parse_args()


def main():
    """Main entry point for geek rating prediction."""
    args = parse_arguments()

    # Create experiment tracker
    tracker = ExperimentTracker(model_type="geek_rating")

    # Create experiment with metadata
    experiment = tracker.create_experiment(
        name=args.experiment,
        description="Geek rating predictions using multiple model experiments",
        metadata={
            "model_experiments": {
                "hurdle": args.hurdle,
                "complexity": args.complexity,
                "rating": args.rating,
                "users_rated": args.users_rated,
            },
            "prediction_parameters": {
                "threshold": args.threshold,
                "prior_rating": args.prior_rating,
                "prior_weight": args.prior_weight,
            },
        },
    )

    # Load configuration
    config = load_config()

    # Create data config for geek rating prediction
    # This requires complexity predictions and supports embeddings
    # min_ratings=0 includes all games (we're scoring, not training)
    data_config = DataConfig(
        min_ratings=0,
        requires_complexity_predictions=True,
        supports_embeddings=True,
    )

    # Load data using centralized loader (includes embeddings and complexity predictions)
    logger.info(f"Loading data with embeddings and complexity predictions")
    df = load_data(
        data_config=data_config,
        start_year=args.start_year,
        end_year=args.end_year,
        use_embeddings=True,
        complexity_predictions_path=args.local_complexity_path,
        apply_filters=False,  # Don't filter - predict on all games
    )

    logger.info(
        f"Loaded {len(df)} games between years "
        f"{args.start_year or 'min'} and {args.end_year or 'max'}"
    )

    # Convert to pandas for prediction
    df_pandas = df.to_pandas()

    # Create and load model
    model = GeekRatingModel.from_experiments(
        hurdle_experiment=args.hurdle,
        complexity_experiment=args.complexity,
        rating_experiment=args.rating,
        users_rated_experiment=args.users_rated,
        prior_rating=args.prior_rating,
        prior_weight=args.prior_weight,
        hurdle_threshold=args.threshold,
    )

    # Generate predictions
    predictions = model.predict(df_pandas)

    # Rename for consistency with old code
    predictions = predictions.rename(columns={"predicted_geek_rating": "prediction"})

    # Convert to Polars for saving
    results = pl.from_pandas(predictions)

    # Ensure predictions directory exists
    predictions_dir = Path(args.output_dir) / tracker.model_type / args.experiment
    predictions_dir.mkdir(parents=True, exist_ok=True)

    output_path = predictions_dir / "predictions.parquet"
    logger.info(f"Saving predictions to: {output_path}")

    results.write_parquet(str(output_path))

    # Get actual values for metrics computation
    if "geek_rating" in df.columns:
        actuals = df["geek_rating"].to_numpy()
    else:
        actuals = np.full(len(results), np.nan)

    # Add actuals to results for saving
    results = results.with_columns(pl.Series("geek_rating", actuals))

    # Log predictions to experiment
    experiment.log_predictions(
        predictions=results["prediction"].to_numpy(),
        actuals=actuals,
        df=results,
        dataset="test",
    )

    # Compute metrics for games with valid geek ratings (non-zero, non-nan)
    pred_values = results["prediction"].to_numpy()
    valid_mask = (actuals > 0) & ~np.isnan(actuals)
    n_valid = valid_mask.sum()

    if n_valid > 0:
        y_true = actuals[valid_mask]
        y_pred = pred_values[valid_mask]

        metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "n_samples": int(n_valid),
            "n_total": int(len(results)),
        }

        # Log metrics to experiment
        experiment.log_metrics(metrics, "test")

        # Display metrics
        logger.info("\nGeek Rating Prediction Metrics:")
        logger.info(f"  Samples with actuals: {n_valid:,} / {len(results):,}")
        logger.info(f"  RMSE:  {metrics['rmse']:.4f}")
        logger.info(f"  MAE:   {metrics['mae']:.4f}")
        logger.info(f"  R²:    {metrics['r2']:.4f}")

        # Show prediction distribution
        logger.info("\nPrediction Summary:")
        logger.info(f"  Min:   {y_pred.min():.4f}")
        logger.info(f"  Max:   {y_pred.max():.4f}")
        logger.info(f"  Mean:  {y_pred.mean():.4f}")
        logger.info(f"  Std:   {y_pred.std():.4f}")

        # Show actual distribution for comparison
        logger.info("\nActual Geek Rating Summary:")
        logger.info(f"  Min:   {y_true.min():.4f}")
        logger.info(f"  Max:   {y_true.max():.4f}")
        logger.info(f"  Mean:  {y_true.mean():.4f}")
        logger.info(f"  Std:   {y_true.std():.4f}")

        # Create diagnostic plots
        plots_dir = experiment.exp_dir / "plots"
        create_diagnostic_plots(y_true, y_pred, plots_dir, metrics)
    else:
        logger.info("\nNo games with valid geek ratings found - skipping metrics computation")

    logger.info("\nExperiment tracked:")
    logger.info(f"  Name: {args.experiment}")
    logger.info("  Experiments used:")
    logger.info(f"    Hurdle: {args.hurdle}")
    logger.info(f"    Complexity: {args.complexity}")
    logger.info(f"    Rating: {args.rating}")
    logger.info(f"    Users Rated: {args.users_rated}")


if __name__ == "__main__":
    main()
