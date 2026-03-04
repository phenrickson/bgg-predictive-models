#!/usr/bin/env python3
"""
Evaluate models over time.

Trains models for each test year using the same pipeline as `make models`,
then runs simulation-based evaluation.

For each test year:
  - Train through: test_year - 2
  - Tune: test_year - 1
  - Test: test_year

Usage:
    uv run -m src.pipeline.evaluate --start-year 2021 --end-year 2022
    uv run -m src.pipeline.evaluate --year 2022
    uv run -m src.pipeline.evaluate --dry-run
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Any

from src.utils.config import load_config
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def run_command(cmd: List[str], description: str, dry_run: bool = False) -> bool:
    """Run a command and log output.

    Args:
        cmd: Command to run as list of strings.
        description: Description for logging.
        dry_run: If True, just log what would run.

    Returns:
        True if successful, False otherwise.
    """
    cmd_str = " ".join(cmd)

    if dry_run:
        logger.info(f"  [DRY RUN] {description}: {cmd_str}")
        return True

    logger.info(f"  {description}: {cmd_str}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output stream to console
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  Command failed with exit code {e.returncode}")
        return False


def train_models_for_year(
    test_year: int,
    config: Any,
    dry_run: bool = False,
) -> Dict[str, str]:
    """Train all models for a given test year.

    Uses the same training pipeline as `make models` but with year-specific
    splits and experiment names.

    Args:
        test_year: The test year to evaluate.
        config: Loaded config object.
        dry_run: If True, just log what would run.

    Returns:
        Dict mapping model type to experiment name.
    """
    train_through = test_year - 2
    tune_start = test_year - 1
    tune_through = test_year - 1
    test_start = test_year
    test_through = test_year

    logger.info(f"\n{'='*60}")
    logger.info(f"Training models for test year {test_year}")
    logger.info(f"  Train through: {train_through}")
    logger.info(f"  Tune: {tune_start}-{tune_through}")
    logger.info(f"  Test: {test_start}-{test_through}")
    logger.info(f"{'='*60}")

    experiment_names = {}

    # Train hurdle
    exp_name = f"eval-hurdle-{test_year}"
    experiment_names["hurdle"] = exp_name
    if not run_command(
        [
            "uv", "run", "-m", "src.pipeline.train",
            "--model", "hurdle",
            "--experiment", exp_name,
            "--train-through", str(train_through),
            "--tune-start", str(tune_start),
            "--tune-through", str(tune_through),
            "--test-start", str(test_start),
            "--test-through", str(test_through),
        ],
        f"Training hurdle: {exp_name}",
        dry_run=dry_run,
    ) and not dry_run:
        raise RuntimeError(f"Hurdle training failed for test year {test_year}")

    # Train complexity
    exp_name = f"eval-complexity-{test_year}"
    experiment_names["complexity"] = exp_name
    if not run_command(
        [
            "uv", "run", "-m", "src.pipeline.train",
            "--model", "complexity",
            "--experiment", exp_name,
            "--train-through", str(train_through),
            "--tune-start", str(tune_start),
            "--tune-through", str(tune_through),
            "--test-start", str(test_start),
            "--test-through", str(test_through),
        ],
        f"Training complexity: {exp_name}",
        dry_run=dry_run,
    ) and not dry_run:
        raise RuntimeError(f"Complexity training failed for test year {test_year}")

    # Score complexity to generate predictions for downstream models
    predictions_dir = Path(config.predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / f"{exp_name}.parquet"

    if not run_command(
        [
            "uv", "run", "-m", "src.pipeline.score",
            "--model", "complexity",
            "--experiment", exp_name,
            "--all-years",
        ],
        f"Scoring complexity: {exp_name}",
        dry_run=dry_run,
    ) and not dry_run:
        raise RuntimeError(f"Complexity scoring failed for test year {test_year}")

    # Train rating
    exp_name = f"eval-rating-{test_year}"
    experiment_names["rating"] = exp_name
    if not run_command(
        [
            "uv", "run", "-m", "src.pipeline.train",
            "--model", "rating",
            "--experiment", exp_name,
            "--train-through", str(train_through),
            "--tune-start", str(tune_start),
            "--tune-through", str(tune_through),
            "--test-start", str(test_start),
            "--test-through", str(test_through),
            "--complexity-predictions", str(predictions_path),
        ],
        f"Training rating: {exp_name}",
        dry_run=dry_run,
    ) and not dry_run:
        raise RuntimeError(f"Rating training failed for test year {test_year}")

    # Train users_rated
    exp_name = f"eval-users_rated-{test_year}"
    experiment_names["users_rated"] = exp_name
    if not run_command(
        [
            "uv", "run", "-m", "src.pipeline.train",
            "--model", "users_rated",
            "--experiment", exp_name,
            "--train-through", str(train_through),
            "--tune-start", str(tune_start),
            "--tune-through", str(tune_through),
            "--test-start", str(test_start),
            "--test-through", str(test_through),
            "--complexity-predictions", str(predictions_path),
        ],
        f"Training users_rated: {exp_name}",
        dry_run=dry_run,
    ) and not dry_run:
        raise RuntimeError(f"Users rated training failed for test year {test_year}")

    # Train geek_rating if simulation config requires it
    sim_config = config.simulation
    geek_rating_mode = sim_config.geek_rating_mode if sim_config else "bayesian"

    if geek_rating_mode in ["stacking", "direct"]:
        exp_name = f"eval-geek_rating-{test_year}"
        experiment_names["geek_rating"] = exp_name

        # Get include_predictions from config (defaults to True)
        geek_rating_config = config.models.get("geek_rating")
        include_predictions = getattr(geek_rating_config, "include_predictions", True) if geek_rating_config else True

        success = run_command(
            [
                "uv", "run", "-m", "src.pipeline.train",
                "--model", "geek_rating",
                "--experiment", exp_name,
                "--mode", geek_rating_mode,
                "--train-through", str(train_through),
                "--tune-start", str(tune_start),
                "--tune-through", str(tune_through),
                "--test-start", str(test_start),
                "--test-through", str(test_through),
                "--include-predictions", str(include_predictions).lower(),
                "--hurdle-experiment", experiment_names["hurdle"],
                "--complexity-experiment", experiment_names["complexity"],
                "--rating-experiment", experiment_names["rating"],
                "--users-rated-experiment", experiment_names["users_rated"],
            ],
            f"Training geek_rating: {exp_name}",
            dry_run=dry_run,
        )
        if not success and not dry_run:
            logger.error(f"  geek_rating training failed, removing from experiments")
            del experiment_names["geek_rating"]

    return experiment_names


def run_simulation_for_year(
    test_year: int,
    experiment_names: Dict[str, str],
    config: Any,
    run_name: str,
    output_dir: str,
    dry_run: bool = False,
) -> None:
    """Run simulation evaluation for a test year.

    Args:
        test_year: The test year to evaluate.
        experiment_names: Dict mapping model type to experiment name.
        config: Loaded config object.
        run_name: Name for this simulation run.
        output_dir: Base directory for simulation output.
        dry_run: If True, just log what would run.
    """
    logger.info(f"\n{'-'*60}")
    logger.info(f"Running simulation for test year {test_year}")
    logger.info(f"{'-'*60}")

    # Determine geek_rating_mode
    sim_config = config.simulation
    geek_rating_mode = sim_config.geek_rating_mode if sim_config else "bayesian"

    # Build command with explicit experiment names
    cmd = [
        "uv", "run", "-m", "src.pipeline.evaluate_simulation",
        "--year", str(test_year),
        "--save-predictions",
        "--run-name", run_name,
        "--output-dir", output_dir,
        "--geek-rating-mode", geek_rating_mode,
        "--complexity-experiment", experiment_names["complexity"],
        "--rating-experiment", experiment_names["rating"],
        "--users-rated-experiment", experiment_names["users_rated"],
    ]

    # Add geek_rating experiment if we trained one
    if "geek_rating" in experiment_names:
        cmd.extend(["--geek-rating-experiment", experiment_names["geek_rating"]])
    elif geek_rating_mode in ["stacking", "direct"]:
        # geek_rating training failed but mode requires it — fall back to bayesian
        logger.warning(
            f"  geek_rating experiment not available, falling back to bayesian mode"
        )
        cmd[cmd.index("--geek-rating-mode") + 1] = "bayesian"

    run_command(
        cmd,
        f"Simulation evaluation for {test_year}",
        dry_run=dry_run,
    )


def aggregate_run_results(run_dir: Path, years: List[int]) -> None:
    """Aggregate per-year predictions and metrics into run-level summaries.

    Creates:
      - predictions.parquet: all years concatenated with test_year column
      - summary_metrics.csv: one row per (test_year, outcome)

    Args:
        run_dir: Path to the run directory containing per-year files.
        years: List of test years that were evaluated.
    """
    import polars as pl

    # Concatenate predictions
    prediction_frames = []
    for year in years:
        path = run_dir / f"predictions_{year}.parquet"
        if path.exists():
            df = pl.read_parquet(path)
            df = df.with_columns(pl.lit(year).alias("test_year"))
            prediction_frames.append(df)

    if prediction_frames:
        combined = pl.concat(prediction_frames)
        combined_path = run_dir / "predictions.parquet"
        combined.write_parquet(combined_path)
        logger.info(f"Saved combined predictions ({len(combined)} rows) to {combined_path}")

    # Collect metrics into summary CSV
    outcomes = ["complexity", "rating", "users_rated", "geek_rating"]
    metric_keys = [
        "n", "rmse_point", "rmse_sim", "mae_point", "mae_sim",
        "r2_point", "r2_sim", "coverage_90", "coverage_50",
        "interval_width_90", "interval_width_50",
    ]

    metrics_rows = []
    for year in years:
        path = run_dir / f"metrics_{year}.json"
        if path.exists():
            with open(path) as f:
                year_metrics = json.load(f)
            for outcome in outcomes:
                if outcome in year_metrics:
                    row = {"test_year": year, "outcome": outcome}
                    for key in metric_keys:
                        row[key] = year_metrics[outcome].get(key)
                    metrics_rows.append(row)

    if metrics_rows:
        metrics_df = pl.DataFrame(metrics_rows)
        metrics_path = run_dir / "summary_metrics.csv"
        metrics_df.write_csv(metrics_path)
        logger.info(f"Saved summary metrics ({len(metrics_rows)} rows) to {metrics_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate models over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run -m src.pipeline.evaluate --year 2022
  uv run -m src.pipeline.evaluate --start-year 2021 --end-year 2022
  uv run -m src.pipeline.evaluate --dry-run
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
        help="First test year to evaluate",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Last test year to evaluate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only run simulation (assumes models already trained)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name for this simulation run (default: {geek_rating_mode}-{date})",
    )

    args = parser.parse_args()

    setup_logging()
    config = load_config()

    # Determine years to evaluate
    if args.year:
        years = [args.year]
    elif args.start_year and args.end_year:
        years = list(range(args.start_year, args.end_year + 1))
    else:
        # Default from config
        years = list(range(config.years.eval.start, config.years.eval.end + 1))

    # Determine simulation settings from config
    sim_config = config.simulation
    geek_rating_mode = sim_config.geek_rating_mode if sim_config else "bayesian"
    n_samples = sim_config.n_samples if sim_config else 500
    output_dir = sim_config.output_dir if sim_config else "./models/simulation"

    # Derive run name
    run_name = args.run_name or f"{geek_rating_mode}-{date.today().isoformat()}"

    # Create output directory
    run_dir = Path(output_dir) / run_name
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Test years: {years}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Output dir: {run_dir}")
    logger.info(f"Geek rating mode: {geek_rating_mode}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Skip training: {args.skip_training}")

    # Show what will happen for each year
    for year in years:
        logger.info(f"\nYear {year}:")
        logger.info(f"  Train ≤ {year - 2}, Tune {year - 1}, Test {year}")

    if args.dry_run:
        logger.info("\n[DRY RUN MODE - No commands will be executed]")

    # Process each year
    all_experiment_names = {}
    for test_year in years:
        if args.skip_training:
            # Build expected experiment names
            experiment_names = {
                "hurdle": f"eval-hurdle-{test_year}",
                "complexity": f"eval-complexity-{test_year}",
                "rating": f"eval-rating-{test_year}",
                "users_rated": f"eval-users_rated-{test_year}",
            }
            if sim_config and geek_rating_mode in ["stacking", "direct"]:
                experiment_names["geek_rating"] = f"eval-geek_rating-{test_year}"
        else:
            # Train models
            experiment_names = train_models_for_year(
                test_year=test_year,
                config=config,
                dry_run=args.dry_run,
            )

        all_experiment_names[test_year] = experiment_names

        # Run simulation
        run_simulation_for_year(
            test_year=test_year,
            experiment_names=experiment_names,
            config=config,
            run_name=run_name,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )

    # Write top-level run metadata
    if not args.dry_run:
        run_metadata = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "geek_rating_mode": geek_rating_mode,
            "n_samples": n_samples,
            "years": years,
            "experiments": {
                str(year): experiments
                for year, experiments in all_experiment_names.items()
            },
        }
        metadata_path = run_dir / "run_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(run_metadata, f, indent=2)
        logger.info(f"\nSaved run metadata to {metadata_path}")

        # Aggregate per-year results into run-level summaries
        aggregate_run_results(run_dir, years)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
