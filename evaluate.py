#!/usr/bin/env python3
"""
Evaluate models over time using configuration from config.yaml.

This script runs time-based evaluation of all models using settings from the config file,
making it easier to maintain and modify evaluation parameters.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.config import load_config
from src.models.time_based_evaluation import (
    run_time_based_evaluation,
    generate_time_splits,
)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_model_args_from_config(config) -> Dict[str, Dict[str, Any]]:
    """
    Build model arguments dictionary from config for time_based_evaluation.

    Args:
        config: Loaded configuration object

    Returns:
        Dictionary of model arguments in the format expected by time_based_evaluation
    """
    model_args = {}

    # Load raw config to access evaluate section
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    evaluate_config = raw_config.get("evaluate", {})
    if not evaluate_config:
        raise ValueError("No 'evaluate' section found in config")

    # Get model defaults and settings
    defaults = evaluate_config.get("defaults", {})
    settings = evaluate_config.get("settings", {})

    # Build args for each model type
    model_types = ["hurdle", "complexity", "rating", "users_rated"]

    for model_name in model_types:
        # Get model type from defaults (e.g., hurdle_model: lightgbm)
        model_key = f"{model_name}_model"
        model_type = defaults.get(model_key)

        if not model_type:
            raise ValueError(f"No default model type found for {model_name}")

        args = {"model": model_type}

        # Add model-specific settings
        model_settings = settings.get(model_name, {})
        if model_settings is None:
            model_settings = {}

        if model_settings.get("use_sample_weights", False):
            args["use-sample-weights"] = True

        if model_settings.get("min_ratings", 0) > 0:
            args["min-ratings"] = model_settings["min_ratings"]

        model_args[model_name] = args

    return model_args


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate models over time using config.yaml settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py                    # Use default config settings
  python evaluate.py --verbose         # Enable debug logging
  python evaluate.py --config custom.yaml  # Use custom config file
  python evaluate.py --output-dir ./custom_experiments  # Custom output directory
  python evaluate.py --start-year 2018 --end-year 2023  # Override year range
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: config.yaml in project root)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for experiments (default: from config or ./models/experiments)",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Override evaluation start year (default: from config)",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Override evaluation end year (default: from config)",
    )

    parser.add_argument(
        "--min-ratings",
        type=int,
        default=0,
        help="Minimum ratings threshold (default: 0)",
    )

    parser.add_argument(
        "--local-data", type=str, default=None, help="Optional path to local data file"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        logger.info(f"Using environment: {config.get_current_environment()}")

        # Determine year range
        start_year = (
            args.start_year if args.start_year is not None else config.years.eval.start
        )
        end_year = args.end_year if args.end_year is not None else config.years.eval.end

        # Determine output directory
        output_dir = args.output_dir if args.output_dir else "./models/experiments"

        # Build model arguments from config
        model_args = build_model_args_from_config(config)

        # Generate time splits
        splits = generate_time_splits(start_year=start_year, end_year=end_year)

        # Log configuration
        logger.info("=== Evaluation Configuration ===")
        logger.info(f"Year range: {start_year} to {end_year}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Number of time splits: {len(splits)}")
        logger.info(f"Min ratings threshold: {args.min_ratings}")
        if args.local_data:
            logger.info(f"Local data path: {args.local_data}")

        logger.info("\n=== Model Configuration ===")
        for model_name, model_config in config.models.items():
            logger.info(f"{model_name}:")
            logger.info(f"  Type: {model_config.type}")
            logger.info(f"  Experiment: {model_config.experiment_name}")
            if (
                hasattr(model_config, "use_sample_weights")
                and model_config.use_sample_weights
            ):
                logger.info(f"  Sample weights: enabled")
            if hasattr(model_config, "min_ratings") and model_config.min_ratings > 0:
                logger.info(f"  Min ratings: {model_config.min_ratings}")

        logger.info("\n=== Time Splits ===")
        for i, split in enumerate(splits):
            logger.info(
                f"Split {i+1}: Train <{split['train_end_year']}, "
                f"Validate {split['tune_start_year']}-{split['tune_end_year']}, "
                f"Test {split['test_start_year']}-{split['test_end_year']}"
            )

        if args.dry_run:
            logger.info("\n=== DRY RUN MODE ===")
            logger.info("Would execute time-based evaluation with above configuration")
            logger.info("Use --verbose to see detailed model arguments")
            if args.verbose:
                logger.debug(f"Model arguments: {model_args}")
            return

        # Run the evaluation
        logger.info("\n=== Starting Evaluation ===")
        run_time_based_evaluation(
            splits=splits,
            min_ratings=args.min_ratings,
            output_dir=output_dir,
            local_data_path=args.local_data,
            model_args=model_args,
        )

        logger.info("=== Evaluation Complete ===")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
