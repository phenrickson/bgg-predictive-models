#!/usr/bin/env python3
"""
Finalize models for production.

Reads model configurations from config.yaml and finalizes each model
in dependency order, including intermediate complexity scoring.

For each model:
  1. hurdle - finalized independently
  2. complexity - finalized independently
  3. complexity scored - generates predictions for downstream models
  4. rating - finalized with complexity predictions
  5. users_rated - finalized with complexity predictions
  6. geek_rating - trained via src.models.outcomes.geek_rating with all upstream experiments

Usage:
    uv run -m src.pipeline.finalize
    uv run -m src.pipeline.finalize --dry-run
    uv run -m src.pipeline.finalize --model complexity
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.models.outcomes.train import finalize_model
from src.models.experiments import ExperimentTracker


logger = logging.getLogger(__name__)


def resolve_experiment_version(model_type: str, experiment_name: str) -> Optional[int]:
    """Resolve the latest version number for an experiment on disk.

    Args:
        model_type: Model type (hurdle, complexity, etc.).
        experiment_name: Experiment name.

    Returns:
        Latest version number, or None if experiment not found.
    """
    try:
        tracker = ExperimentTracker(model_type)
        experiment_dir = tracker.model_dir / experiment_name
        if not experiment_dir.exists():
            return None
        versions = [
            int(v.name[1:])
            for v in experiment_dir.iterdir()
            if v.is_dir() and v.name.startswith("v") and v.name[1:].isdigit()
        ]
        return max(versions) if versions else None
    except Exception:
        return None


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
        subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  Command failed with exit code {e.returncode}")
        return False


def finalize_single_model(
    model_type: str,
    experiment_name: str,
    config,
    dry_run: bool = False,
    complexity_predictions_path: Optional[str] = None,
) -> bool:
    """Finalize a single model.

    Args:
        model_type: Model type (hurdle, complexity, rating, users_rated).
        experiment_name: Experiment name from config.
        config: Loaded config object.
        dry_run: If True, just log what would run.
        complexity_predictions_path: Path to complexity predictions (for rating/users_rated).

    Returns:
        True if successful, False otherwise.
    """
    model_config = config.models.get(model_type)
    use_embeddings = getattr(model_config, "use_embeddings", False)
    recent_year_threshold = getattr(config.years, "recent_year_threshold", 2)

    if dry_run:
        logger.info(f"  [DRY RUN] Finalize {model_type}: {experiment_name}")
        if use_embeddings:
            logger.info(f"    use_embeddings: True")
        if complexity_predictions_path and model_type in ("rating", "users_rated"):
            logger.info(f"    complexity_predictions: {complexity_predictions_path}")
        return True

    logger.info(f"  Finalizing {model_type}: {experiment_name}")

    try:
        finalize_model(
            model_type=model_type,
            experiment_name=experiment_name,
            use_embeddings=use_embeddings if use_embeddings else None,
            complexity_predictions_path=(
                complexity_predictions_path
                if model_type in ("rating", "users_rated")
                else None
            ),
            recent_year_threshold=recent_year_threshold,
        )
        return True
    except Exception as e:
        logger.error(f"  Failed to finalize {model_type}: {e}")
        return False


def score_complexity(
    experiment_name: str,
    dry_run: bool = False,
) -> bool:
    """Score all data using complexity model to generate predictions for downstream models.

    Args:
        experiment_name: Complexity experiment name.
        dry_run: If True, just log what would run.

    Returns:
        True if successful, False otherwise.
    """
    cmd = [
        "uv", "run", "-m", "src.pipeline.score",
        "--model", "complexity",
        "--experiment", experiment_name,
        "--all-years",
    ]

    return run_command(cmd, f"Scoring complexity: {experiment_name}", dry_run=dry_run)


def train_geek_rating(
    experiment_names: dict,
    config,
    dry_run: bool = False,
) -> bool:
    """Train geek_rating model using upstream finalized experiments.

    Args:
        experiment_names: Dict mapping model type to experiment name.
        config: Loaded config object.
        dry_run: If True, just log what would run.

    Returns:
        True if successful, False otherwise.
    """
    geek_config = config.models.get("geek_rating")
    geek_experiment = getattr(geek_config, "experiment_name", "geek_rating")
    mode = getattr(geek_config, "mode", "direct")
    include_predictions = getattr(geek_config, "include_predictions", True)

    cmd = [
        "uv", "run", "-m", "src.pipeline.train",
        "--model", "geek_rating",
        "--experiment", geek_experiment,
        "--mode", mode,
        "--include-predictions", str(include_predictions).lower(),
        "--hurdle-experiment", experiment_names["hurdle"],
        "--complexity-experiment", experiment_names["complexity"],
        "--rating-experiment", experiment_names["rating"],
        "--users-rated-experiment", experiment_names["users_rated"],
    ]

    return run_command(cmd, f"Training geek_rating: {geek_experiment}", dry_run=dry_run)


def finalize_all(config, dry_run: bool = False, single_model: Optional[str] = None) -> None:
    """Finalize all models in dependency order.

    Args:
        config: Loaded config object.
        dry_run: If True, just log what would run.
        single_model: If set, only finalize this model.
    """
    predictions_dir = Path(config.predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Build experiment name map from config
    experiment_names = {}
    for model_type in ("hurdle", "complexity", "rating", "users_rated"):
        model_config = config.models.get(model_type)
        if model_config:
            experiment_names[model_type] = getattr(model_config, "experiment_name")

    # Define dependency order
    ordered_steps = [
        "hurdle",
        "complexity",
        "score_complexity",
        "rating",
        "users_rated",
        "geek_rating",
    ]

    # If single model requested, determine which steps are needed
    if single_model:
        if single_model in ("hurdle", "complexity"):
            ordered_steps = [single_model]
        elif single_model in ("rating", "users_rated"):
            # Need complexity scored first
            ordered_steps = ["score_complexity", single_model]
        elif single_model == "geek_rating":
            ordered_steps = ["geek_rating"]
        else:
            logger.error(f"Unknown model: {single_model}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("FINALIZATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Steps: {ordered_steps}")
    logger.info(f"Dry run: {dry_run}")

    for model_type in ("hurdle", "complexity", "rating", "users_rated"):
        if model_type in experiment_names:
            exp_name = experiment_names[model_type]
            version = resolve_experiment_version(model_type, exp_name)
            version_str = f"v{version}" if version else "not found"
            logger.info(f"  {model_type}: {exp_name} ({version_str})")

    geek_config = config.models.get("geek_rating")
    if geek_config:
        geek_exp = getattr(geek_config, "experiment_name", "N/A")
        version = resolve_experiment_version("geek_rating", geek_exp)
        version_str = f"v{version}" if version else "not found"
        logger.info(f"  geek_rating: {geek_exp} ({version_str})")

    if dry_run:
        logger.info("\n[DRY RUN MODE - No commands will be executed]")

    # Complexity predictions path
    complexity_experiment = experiment_names.get("complexity", "")
    complexity_predictions_path = str(predictions_dir / f"{complexity_experiment}.parquet")

    # Execute steps
    for step in ordered_steps:
        logger.info(f"\n{'-'*60}")

        if step == "score_complexity":
            success = score_complexity(
                experiment_name=complexity_experiment,
                dry_run=dry_run,
            )
        elif step == "geek_rating":
            success = train_geek_rating(
                experiment_names=experiment_names,
                config=config,
                dry_run=dry_run,
            )
        else:
            success = finalize_single_model(
                model_type=step,
                experiment_name=experiment_names[step],
                config=config,
                dry_run=dry_run,
                complexity_predictions_path=complexity_predictions_path,
            )

        if not success:
            logger.error(f"Step '{step}' failed. Stopping pipeline.")
            sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("FINALIZATION COMPLETE")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Finalize models for production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run -m src.pipeline.finalize                     # finalize all from config
  uv run -m src.pipeline.finalize --model complexity   # just one
  uv run -m src.pipeline.finalize --dry-run            # show what would run
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["hurdle", "complexity", "rating", "users_rated", "geek_rating"],
        help="Finalize only this model (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args()

    setup_logging()
    config = load_config()

    finalize_all(
        config=config,
        dry_run=args.dry_run,
        single_model=args.model,
    )


if __name__ == "__main__":
    main()
