#!/usr/bin/env python3
"""
Evaluate simulation-based predictions vs point predictions.

This script loads trained models from evaluate.py experiments and compares:
1. Point predictions: rating/users_rated conditioned on complexity_point
2. Simulation: rating/users_rated conditioned on complexity_samples (uncertainty propagated)

Usage:
    uv run python evaluate_simulation.py --year 2023
    uv run python evaluate_simulation.py --start-year 2023 --end-year 2024
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import polars as pl
import joblib
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.models.outcomes.simulation import (
    simulate_batch,
    precompute_cholesky,
    compute_simulation_metrics,
)
from src.models.outcomes.data import load_data
from src.models.outcomes.base import DataConfig


logger = logging.getLogger(__name__)


def create_scatter_plots(
    predictions_df: pl.DataFrame,
    test_year: int,
    output_dir: Path,
) -> None:
    """Create scatter plots for predicted vs actual and simulation vs point.

    Args:
        predictions_df: DataFrame with predictions and actuals.
        test_year: The test year for labeling.
        output_dir: Directory to save plots.
    """
    df = predictions_df.to_pandas()
    outcomes = ["complexity", "rating", "users_rated", "geek_rating"]

    # Create figure with 2 rows: predicted vs actual, simulation vs point
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Predicted vs Actual (using point predictions)
    for i, outcome in enumerate(outcomes):
        ax = axes[0, i]
        actual = df[f"{outcome}_actual"].copy()
        point = df[f"{outcome}_point"].copy()

        # Filter out invalid actuals (complexity=0 means missing)
        if outcome == "complexity":
            valid_mask = actual > 0
            actual = actual[valid_mask]
            point = point[valid_mask]

        # users_rated is already in log scale from simulation.py
        if outcome == "users_rated":
            ax.set_xlabel("Predicted (log scale)")
            ax.set_ylabel("Actual (log scale)")
        else:
            ax.set_xlabel("Predicted (Point)")
            ax.set_ylabel("Actual")

        ax.scatter(point, actual, alpha=0.3, s=10)

        # Add diagonal line
        min_val = min(actual.min(), point.min())
        max_val = max(actual.max(), point.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1, label="y=x")

        ax.set_title(f"{outcome.replace('_', ' ').title()}")

        # Add correlation
        corr = actual.corr(point)
        ax.text(
            0.05, 0.95, f"r = {corr:.3f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Row 2: Simulation median vs Point prediction
    for i, outcome in enumerate(outcomes):
        ax = axes[1, i]
        point = df[f"{outcome}_point"]
        sim_median = df[f"{outcome}_median"]

        # users_rated is already in log scale from simulation.py
        ax.scatter(point, sim_median, alpha=0.3, s=10)

        # Add diagonal line
        min_val = min(point.min(), sim_median.min())
        max_val = max(point.max(), sim_median.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1, label="y=x")

        ax.set_xlabel("Point Prediction" + (" (log scale)" if outcome == "users_rated" else ""))
        ax.set_ylabel("Simulation Median" + (" (log scale)" if outcome == "users_rated" else ""))
        ax.set_title(f"{outcome.replace('_', ' ').title()}")

        # Add correlation
        corr = point.corr(sim_median)
        ax.text(
            0.05, 0.95, f"r = {corr:.3f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Add row labels
    fig.text(0.5, 0.98, "Predicted vs Actual", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.48, "Simulation Median vs Point Prediction", ha="center", fontsize=12, fontweight="bold")

    plt.suptitle(f"Simulation Evaluation - {test_year}", fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    plot_path = output_dir / f"scatter_plots_{test_year}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved scatter plots to {plot_path}")


def create_top_games_plot(
    predictions_df: pl.DataFrame,
    test_year: int,
    output_dir: Path,
    top_n: int = 100,
) -> None:
    """Create forest plot showing top N games with intervals for all outcomes.

    Shows game names on y-axis with horizontal intervals for complexity, rating,
    users_rated (log scale), and geek_rating. Filled dots for predictions,
    open circles for actuals.

    Args:
        predictions_df: DataFrame with predictions and actuals.
        test_year: The test year for labeling.
        output_dir: Directory to save plots.
        top_n: Number of top games to show (default 100).
    """
    df = predictions_df.to_pandas()

    # Sort by predicted geek_rating (point prediction) descending
    df_sorted = df.sort_values("geek_rating_point", ascending=False).head(top_n)
    df_sorted = df_sorted.reset_index(drop=True)

    # Create game labels (truncate long names)
    df_sorted["label"] = df_sorted.apply(
        lambda row: f"{row['name'][:30]}..." if len(str(row['name'])) > 30 else row['name'],
        axis=1
    )

    outcomes = ["complexity", "rating", "users_rated", "geek_rating"]
    n_games = len(df_sorted)

    # Create figure with 4 subplots side by side
    fig, axes = plt.subplots(1, 4, figsize=(20, max(12, n_games * 0.15)), sharey=True)

    y_positions = np.arange(n_games)

    for i, outcome in enumerate(outcomes):
        ax = axes[i]

        lower_90 = df_sorted[f"{outcome}_lower_90"]
        upper_90 = df_sorted[f"{outcome}_upper_90"]
        lower_50 = df_sorted[f"{outcome}_lower_50"]
        upper_50 = df_sorted[f"{outcome}_upper_50"]
        median = df_sorted[f"{outcome}_median"]
        actual = df_sorted[f"{outcome}_actual"]

        # Plot 90% intervals as thin lines
        for j in range(n_games):
            ax.plot(
                [lower_90.iloc[j], upper_90.iloc[j]],
                [y_positions[j], y_positions[j]],
                color="steelblue",
                linewidth=1,
                alpha=0.6,
            )

        # Plot 50% intervals as thicker lines
        for j in range(n_games):
            ax.plot(
                [lower_50.iloc[j], upper_50.iloc[j]],
                [y_positions[j], y_positions[j]],
                color="steelblue",
                linewidth=3,
                alpha=0.8,
            )

        # Plot median (prediction) as filled dots
        ax.scatter(
            median,
            y_positions,
            color="steelblue",
            s=25,
            zorder=5,
            label="Predicted",
        )

        # Plot actual as open circles
        ax.scatter(
            actual,
            y_positions,
            facecolors="none",
            edgecolors="red",
            s=40,
            linewidths=1.5,
            zorder=6,
            label="Actual",
        )

        ax.set_xlabel(outcome.replace("_", " ").title(), fontsize=11)
        ax.set_title(outcome.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

        # Add legend only to first subplot
        if i == 0:
            ax.legend(loc="lower left", fontsize=8)

    # Set y-axis labels on first subplot only
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(df_sorted["label"], fontsize=7)
    axes[0].set_ylabel("Game", fontsize=11)

    # Invert y-axis so rank 1 is at top
    axes[0].invert_yaxis()

    plt.suptitle(
        f"Top {top_n} Games by Predicted Geek Rating - {test_year}\n"
        "(line = 90% interval, thick = 50% interval, dot = predicted, circle = actual)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    # Save
    plot_path = output_dir / f"top_{top_n}_games_{test_year}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved top {top_n} games plot to {plot_path}")


def load_pipeline(
    model_type: str,
    experiment_name: str,
    base_dir: str = "models/experiments",
) -> object:
    """Load a trained pipeline from an experiment.

    Args:
        model_type: Type of model (complexity, rating, users_rated, geek_rating).
        experiment_name: Name of the experiment.
        base_dir: Base directory for experiments.

    Returns:
        Loaded sklearn pipeline.
    """
    exp_dir = Path(base_dir) / model_type / experiment_name

    # Find latest version
    version_dirs = [
        d for d in exp_dir.iterdir()
        if d.is_dir() and d.name.startswith("v")
    ]
    if not version_dirs:
        raise FileNotFoundError(f"No versions found in {exp_dir}")

    latest_version = max(version_dirs, key=lambda x: int(x.name[1:]))

    # Try pipeline.pkl first, then model.joblib
    pipeline_path = latest_version / "pipeline.pkl"
    if not pipeline_path.exists():
        pipeline_path = latest_version / "model.joblib"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"No pipeline found in {latest_version}")

    return joblib.load(pipeline_path)


def load_pipelines_from_config(
    config,
    base_dir: str = "models/experiments",
) -> Dict[str, Any]:
    """Load all pipelines needed for simulation from config.

    Args:
        config: Loaded Config object.
        base_dir: Base directory for experiments.

    Returns:
        Dictionary with pipelines keyed by model type.
    """
    pipelines = {}

    # Load required models (complexity, rating, users_rated)
    for model_type in ["complexity", "rating", "users_rated"]:
        if model_type not in config.models:
            raise ValueError(f"Missing {model_type} in config.models")
        exp_name = config.models[model_type].experiment_name
        pipelines[model_type] = load_pipeline(model_type, exp_name, base_dir)
        logger.info(f"  Loaded {model_type}: {exp_name}")

    # Load geek_rating pipeline if mode requires it
    geek_rating_mode = "bayesian"
    if config.simulation:
        geek_rating_mode = config.simulation.geek_rating_mode

    if geek_rating_mode in ["stacking", "direct"]:
        if "geek_rating" not in config.models:
            raise ValueError(
                f"geek_rating_mode={geek_rating_mode} requires geek_rating in config.models"
            )
        exp_name = config.models["geek_rating"].experiment_name
        pipelines["geek_rating"] = load_pipeline("geek_rating", exp_name, base_dir)
        logger.info(f"  Loaded geek_rating: {exp_name}")

    return pipelines


def check_bayesian_model(pipeline) -> bool:
    """Check if a pipeline contains a Bayesian model that supports simulation."""
    model = pipeline.named_steps.get("model")
    if model is None:
        return False
    return hasattr(model, "coef_") and hasattr(model, "sigma_")


def evaluate_year(
    test_year: int,
    config=None,
    base_dir: str = "models/experiments",
    n_samples: int = 500,
    save_predictions: bool = False,
    output_dir: Optional[str] = None,
    geek_rating_mode: str = "bayesian",
    experiment_names: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Evaluate simulation vs point prediction for a single test year.

    Args:
        test_year: The test year to evaluate.
        config: Loaded Config object. If provided and experiment_names is None,
            loads models from config experiment names.
        base_dir: Base directory for experiments.
        n_samples: Number of posterior samples.
        save_predictions: Whether to save game-level predictions.
        output_dir: Directory to save predictions (defaults to base_dir/simulation).
        geek_rating_mode: How to compute geek rating: "bayesian", "stacking", or "direct".
        experiment_names: Optional dict mapping model type to experiment name.
            If provided, loads these experiments directly.

    Returns:
        Dictionary of evaluation results.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating simulation for test year {test_year}")
    logger.info(f"geek_rating_mode: {geek_rating_mode}")
    logger.info(f"{'='*60}")

    # Load trained models
    pipelines = {}

    if experiment_names is not None:
        # Load explicitly named experiments
        for model_type in ["complexity", "rating", "users_rated"]:
            if model_type not in experiment_names:
                logger.error(f"  Missing {model_type} in experiment_names")
                return {"test_year": test_year, "error": f"Missing {model_type}"}
            exp_name = experiment_names[model_type]
            try:
                pipeline = load_pipeline(model_type, exp_name, base_dir)
                pipelines[model_type] = pipeline
                logger.info(f"  Loaded {model_type}: {exp_name}")
            except FileNotFoundError as e:
                logger.error(f"  Could not load {model_type}: {e}")
                return {"test_year": test_year, "error": str(e)}

        # Load geek_rating if provided
        if "geek_rating" in experiment_names:
            exp_name = experiment_names["geek_rating"]
            try:
                pipeline = load_pipeline("geek_rating", exp_name, base_dir)
                pipelines["geek_rating"] = pipeline
                logger.info(f"  Loaded geek_rating: {exp_name}")
            except FileNotFoundError as e:
                logger.warning(f"  Could not load geek_rating: {e}")

    elif config is not None:
        # Load from config experiment names
        try:
            pipelines = load_pipelines_from_config(config, base_dir)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"  Could not load pipelines from config: {e}")
            return {"test_year": test_year, "error": str(e)}
    else:
        logger.error("  Must provide either experiment_names or config")
        return {"test_year": test_year, "error": "No experiments specified"}

    # Check if models support Bayesian simulation
    for model_type, pipeline in pipelines.items():
        if not check_bayesian_model(pipeline):
            logger.warning(
                f"  {model_type} model does not support Bayesian simulation"
            )
            return {"test_year": test_year, "error": f"{model_type} not Bayesian"}

    # Load test data
    if config is None:
        config = load_config()

    data_config = DataConfig(
        min_ratings=0,
        requires_complexity_predictions=False,
        supports_embeddings=True,
    )

    df = load_data(
        data_config=data_config,
        start_year=test_year,
        end_year=test_year,
        use_embeddings=True,
        apply_filters=False,
    )

    df_pandas = df.to_pandas()

    # Filter to games with valid outcomes for evaluation
    valid_mask = (
        ~df_pandas["rating"].isna()
        & ~df_pandas["users_rated"].isna()
        & (df_pandas["users_rated"] >= 0)
    )
    df_valid = df_pandas[valid_mask].reset_index(drop=True)
    n_games = len(df_valid)

    logger.info(f"  Total games: {len(df_pandas)}")
    logger.info(f"  Valid games (with ratings): {n_games}")

    if n_games == 0:
        logger.warning(f"  No valid games for evaluation")
        return {"test_year": test_year, "error": "No valid games"}

    # Get scoring params
    scoring_params = config.scoring.parameters
    prior_rating = scoring_params.get("prior_rating", 5.5)
    prior_weight = scoring_params.get("prior_weight", 2000)

    # Get random state from config or default
    random_state = 42
    if config.simulation:
        random_state = config.simulation.random_state

    # Get geek_rating pipeline if needed
    geek_rating_pipeline = pipelines.get("geek_rating")

    # Pre-compute Cholesky
    logger.info(f"  Pre-computing Cholesky decompositions...")
    cholesky_cache = precompute_cholesky(
        pipelines["complexity"],
        pipelines["rating"],
        pipelines["users_rated"],
        geek_rating_pipeline=geek_rating_pipeline,
    )

    # Run simulation
    logger.info(f"  Running simulation ({n_samples} samples, {n_games} games)...")
    results = simulate_batch(
        df_valid,
        pipelines["complexity"],
        pipelines["rating"],
        pipelines["users_rated"],
        n_samples=n_samples,
        prior_rating=prior_rating,
        prior_weight=prior_weight,
        random_state=random_state,
        cholesky_cache=cholesky_cache,
        geek_rating_mode=geek_rating_mode,
        geek_rating_pipeline=geek_rating_pipeline,
    )

    # Compute metrics
    metrics = compute_simulation_metrics(results)

    # Save game-level predictions if requested
    if save_predictions:
        predictions_dir = Path(output_dir) if output_dir else Path("models/simulation")
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Build predictions dataframe
        predictions_data = []
        for r in results:
            s = r.summary()
            predictions_data.append({
                "game_id": r.game_id,
                "name": r.game_name,
                # Actuals - users_rated in LOG SCALE (matches model training)
                "complexity_actual": r.actual_complexity,
                "rating_actual": r.actual_rating,
                "users_rated_actual": s["users_rated"]["actual"],  # log scale
                "geek_rating_actual": r.actual_geek_rating,
                # Point predictions - users_rated in LOG SCALE
                "complexity_point": r.complexity_point,
                "rating_point": r.rating_point,
                "users_rated_point": s["users_rated"]["point"],  # log scale
                "geek_rating_point": r.geek_rating_point,
                # Simulation median - users_rated in LOG SCALE
                "complexity_median": s["complexity"]["median"],
                "rating_median": s["rating"]["median"],
                "users_rated_median": s["users_rated"]["median"],  # log scale
                "geek_rating_median": s["geek_rating"]["median"],
                # 90% intervals
                "complexity_lower_90": s["complexity"]["interval_90"][0],
                "complexity_upper_90": s["complexity"]["interval_90"][1],
                "rating_lower_90": s["rating"]["interval_90"][0],
                "rating_upper_90": s["rating"]["interval_90"][1],
                "users_rated_lower_90": s["users_rated"]["interval_90"][0],
                "users_rated_upper_90": s["users_rated"]["interval_90"][1],
                "geek_rating_lower_90": s["geek_rating"]["interval_90"][0],
                "geek_rating_upper_90": s["geek_rating"]["interval_90"][1],
                # 50% intervals
                "complexity_lower_50": s["complexity"]["interval_50"][0],
                "complexity_upper_50": s["complexity"]["interval_50"][1],
                "rating_lower_50": s["rating"]["interval_50"][0],
                "rating_upper_50": s["rating"]["interval_50"][1],
                "users_rated_lower_50": s["users_rated"]["interval_50"][0],
                "users_rated_upper_50": s["users_rated"]["interval_50"][1],
                "geek_rating_lower_50": s["geek_rating"]["interval_50"][0],
                "geek_rating_upper_50": s["geek_rating"]["interval_50"][1],
            })

        predictions_df = pl.DataFrame(predictions_data)

        # Save as parquet
        predictions_path = predictions_dir / f"predictions_{test_year}.parquet"
        predictions_df.write_parquet(predictions_path)
        logger.info(f"  Saved predictions to {predictions_path}")

        # Save as CSV for easy viewing
        csv_path = predictions_dir / f"predictions_{test_year}.csv"
        predictions_df.write_csv(csv_path)
        logger.info(f"  Saved predictions to {csv_path}")

        # Also save metrics as JSON
        import json
        metrics_path = predictions_dir / f"metrics_{test_year}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  Saved metrics to {metrics_path}")

        # Create scatter plots
        create_scatter_plots(predictions_df, test_year, predictions_dir)

        # Create top games plot
        create_top_games_plot(predictions_df, test_year, predictions_dir)

    # Build output
    output = {
        "test_year": test_year,
        "n_games": n_games,
        "n_samples": n_samples,
    }

    # Log results
    logger.info(f"\n  Results:")
    logger.info(f"  {'-'*50}")

    for outcome in ["complexity", "rating", "users_rated", "geek_rating"]:
        if outcome in metrics and metrics[outcome].get("n", 0) > 0:
            m = metrics[outcome]

            # Add to output
            output[f"{outcome}_rmse_point"] = m.get("rmse_point")
            output[f"{outcome}_rmse_sim"] = m.get("rmse_sim")
            output[f"{outcome}_mae_point"] = m.get("mae_point")
            output[f"{outcome}_mae_sim"] = m.get("mae_sim")
            output[f"{outcome}_r2_point"] = m.get("r2_point")
            output[f"{outcome}_r2_sim"] = m.get("r2_sim")
            output[f"{outcome}_coverage_90"] = m.get("coverage_90")
            output[f"{outcome}_coverage_50"] = m.get("coverage_50")
            output[f"{outcome}_interval_width_90"] = m.get("interval_width_90")
            output[f"{outcome}_interval_width_50"] = m.get("interval_width_50")

            # Log
            logger.info(f"\n  {outcome.upper()}:")
            logger.info(f"    RMSE (point): {m.get('rmse_point', 'N/A'):.4f}")
            logger.info(f"    RMSE (sim):   {m.get('rmse_sim', 'N/A'):.4f}")
            logger.info(f"    MAE (point):  {m.get('mae_point', 'N/A'):.4f}")
            logger.info(f"    MAE (sim):    {m.get('mae_sim', 'N/A'):.4f}")
            logger.info(f"    R² (point):   {m.get('r2_point', 'N/A'):.4f}")
            logger.info(f"    R² (sim):     {m.get('r2_sim', 'N/A'):.4f}")
            logger.info(f"    Coverage 90%: {m.get('coverage_90', 0):.1%} (expected 90%)")
            logger.info(f"    Coverage 50%: {m.get('coverage_50', 0):.1%} (expected 50%)")
            if m.get("interval_width_90"):
                logger.info(f"    Interval width 90%: {m.get('interval_width_90'):.3f}")

    return output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate simulation-based predictions vs point predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python evaluate_simulation.py --year 2023
  uv run python evaluate_simulation.py --start-year 2023 --end-year 2024
  uv run python evaluate_simulation.py --n-samples 1000
  uv run python evaluate_simulation.py --geek-rating-mode direct
  uv run python evaluate_simulation.py --complexity-experiment eval-complexity-2022 --rating-experiment eval-rating-2022 --users-rated-experiment eval-users_rated-2022
        """,
    )

    parser.add_argument(
        "--year",
        type=int,
        help="Single test year to evaluate",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="First test year for evaluation",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Last test year for evaluation",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="Number of posterior samples (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Base directory for experiments (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results CSV",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=None,
        help="Save game-level predictions (overrides config)",
    )
    parser.add_argument(
        "--geek-rating-mode",
        type=str,
        choices=["bayesian", "stacking", "direct"],
        help="Geek rating mode (overrides config)",
    )
    # Explicit experiment names (used by evaluate.py)
    parser.add_argument(
        "--complexity-experiment",
        type=str,
        help="Complexity experiment name to load",
    )
    parser.add_argument(
        "--rating-experiment",
        type=str,
        help="Rating experiment name to load",
    )
    parser.add_argument(
        "--users-rated-experiment",
        type=str,
        help="Users rated experiment name to load",
    )
    parser.add_argument(
        "--geek-rating-experiment",
        type=str,
        help="Geek rating experiment name to load",
    )

    args = parser.parse_args()

    setup_logging()

    # Load config
    config = load_config()

    # Get settings from config with CLI overrides
    n_samples = args.n_samples
    if n_samples is None:
        n_samples = config.simulation.n_samples if config.simulation else 500

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = config.simulation.output_dir if config.simulation else "./models/experiments"

    save_predictions = args.save_predictions
    if save_predictions is None:
        save_predictions = config.simulation.save_predictions if config.simulation else False

    geek_rating_mode = args.geek_rating_mode
    if geek_rating_mode is None:
        geek_rating_mode = config.simulation.geek_rating_mode if config.simulation else "bayesian"

    # Determine years to evaluate
    if args.year:
        years = [args.year]
    elif args.start_year and args.end_year:
        years = list(range(args.start_year, args.end_year + 1))
    else:
        years = list(range(config.years.eval.start, config.years.eval.end + 1))

    # Build experiment_names if explicit experiments provided
    experiment_names = None
    if args.complexity_experiment and args.rating_experiment and args.users_rated_experiment:
        experiment_names = {
            "complexity": args.complexity_experiment,
            "rating": args.rating_experiment,
            "users_rated": args.users_rated_experiment,
        }
        if args.geek_rating_experiment:
            experiment_names["geek_rating"] = args.geek_rating_experiment

    logger.info("=" * 60)
    logger.info("SIMULATION EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Years: {years}")
    logger.info(f"Samples: {n_samples}")
    logger.info(f"Geek rating mode: {geek_rating_mode}")
    logger.info(f"Output dir: {output_dir}")
    if experiment_names:
        logger.info(f"Experiments: {experiment_names}")
    else:
        logger.info("Using experiments from config")

    # Evaluate each year
    all_results = []
    for year in years:
        result = evaluate_year(
            test_year=year,
            config=config if experiment_names is None else None,
            base_dir="./models/experiments",
            n_samples=n_samples,
            save_predictions=save_predictions,
            output_dir=output_dir,
            geek_rating_mode=geek_rating_mode,
            experiment_names=experiment_names,
        )
        all_results.append(result)

    # Create summary DataFrame
    df_results = pd.DataFrame(all_results)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if "error" not in df_results.columns or df_results["error"].isna().all():
        # Successful runs
        print("\nPoint vs Simulation Comparison:")
        print("-" * 50)

        for outcome in ["complexity", "rating", "users_rated", "geek_rating"]:
            point_col = f"{outcome}_rmse_point"
            sim_col = f"{outcome}_rmse_sim"

            if point_col in df_results.columns and sim_col in df_results.columns:
                point_mean = df_results[point_col].mean()
                sim_mean = df_results[sim_col].mean()
                diff = sim_mean - point_mean
                pct = (diff / point_mean) * 100 if point_mean > 0 else 0

                print(f"\n{outcome.upper()}:")
                print(f"  Mean RMSE (point): {point_mean:.4f}")
                print(f"  Mean RMSE (sim):   {sim_mean:.4f}")
                print(f"  Difference:        {diff:+.4f} ({pct:+.1f}%)")

                cov_col = f"{outcome}_coverage_90"
                if cov_col in df_results.columns:
                    cov_mean = df_results[cov_col].mean()
                    print(f"  Mean Coverage 90%: {cov_mean:.1%}")

    # Save results
    if args.output:
        df_results.to_csv(args.output, index=False)
        logger.info(f"\nResults saved to: {args.output}")
    else:
        # Print formatted results by year
        print("\n\nResults by Year:")
        print("=" * 80)

        for _, row in df_results.iterrows():
            year = int(row["test_year"])
            n_games = int(row["n_games"])
            n_samples = int(row["n_samples"])

            print(f"\n{year} ({n_games} games, {n_samples} samples)")
            print("-" * 80)

            # Header
            print(f"{'Outcome':<12} {'RMSE_pt':>8} {'RMSE_sim':>9} {'MAE_pt':>8} {'MAE_sim':>8} {'R²_pt':>7} {'R²_sim':>7} {'Cov90':>6} {'Cov50':>6}")
            print("-" * 80)

            for outcome in ["complexity", "rating", "users_rated", "geek_rating"]:
                rmse_pt = row.get(f"{outcome}_rmse_point", float("nan"))
                rmse_sim = row.get(f"{outcome}_rmse_sim", float("nan"))
                mae_pt = row.get(f"{outcome}_mae_point", float("nan"))
                mae_sim = row.get(f"{outcome}_mae_sim", float("nan"))
                r2_pt = row.get(f"{outcome}_r2_point", float("nan"))
                r2_sim = row.get(f"{outcome}_r2_sim", float("nan"))
                cov90 = row.get(f"{outcome}_coverage_90", float("nan"))
                cov50 = row.get(f"{outcome}_coverage_50", float("nan"))

                # Format coverage as percentage
                cov90_str = f"{cov90:.1%}" if not pd.isna(cov90) else "N/A"
                cov50_str = f"{cov50:.1%}" if not pd.isna(cov50) else "N/A"

                print(f"{outcome:<12} {rmse_pt:>8.3f} {rmse_sim:>9.3f} {mae_pt:>8.3f} {mae_sim:>8.3f} {r2_pt:>7.3f} {r2_sim:>7.3f} {cov90_str:>6} {cov50_str:>6}")


if __name__ == "__main__":
    main()
