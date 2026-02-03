#!/usr/bin/env python3
"""
Evaluate models over time using configuration from config.yaml.

This script runs time-based evaluation of all models using settings from the config file,
making it easier to maintain and modify evaluation parameters.

Usage:
    uv run python evaluate.py                    # Use default config settings
    uv run python evaluate.py --dry-run          # Show what would be run without executing
    uv run python evaluate.py --start-year 2020  # Start evaluation from 2020
    uv run python evaluate.py --end-year 2022    # End evaluation at 2022
"""

import argparse
import logging
import sys

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.models.time_based_evaluation import (
    run_time_based_evaluation,
    generate_time_splits,
)


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate models over time using config.yaml settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python evaluate.py                    # Use default config settings (2018-2024)
  uv run python evaluate.py --dry-run          # Show configuration without running
  uv run python evaluate.py --start-year 2020  # Start evaluation from 2020
  uv run python evaluate.py --end-year 2022    # End evaluation at 2022
  uv run python evaluate.py --simulation       # Include simulation-based uncertainty evaluation
  uv run python evaluate.py --output-dir ./custom_eval  # Custom output directory
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/experiments",
        help="Base directory for experiments",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First test year for evaluation (default: from config.years.eval.start)",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last test year for evaluation (default: from config.years.eval.end)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run simulation-based evaluation after training models",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        config = load_config()

        # Determine year range
        start_year = args.start_year or config.years.eval.start
        end_year = args.end_year or config.years.eval.end

        # Generate time splits
        splits = generate_time_splits(start_year=start_year, end_year=end_year)

        # Log configuration
        logger.info("=" * 60)
        logger.info("EVALUATION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Test years: {start_year} to {end_year}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Number of splits: {len(splits)}")

        logger.info("\nModel Configuration (from config.yaml):")
        for model_type in ["hurdle", "complexity", "rating", "users_rated"]:
            model_config = config.models.get(model_type)
            if model_config:
                extras = []
                if getattr(model_config, "use_embeddings", False):
                    extras.append("embeddings")
                if getattr(model_config, "use_sample_weights", False):
                    extras.append("sample_weights")
                extras_str = f" ({', '.join(extras)})" if extras else ""
                logger.info(f"  {model_type}: {model_config.type}{extras_str}")

        if args.simulation:
            logger.info("\nSimulation Configuration:")
            if config.simulation:
                logger.info(f"  n_samples: {config.simulation.n_samples}")
                logger.info(f"  geek_rating_mode: {config.simulation.geek_rating_mode}")
                if config.simulation.geek_rating_mode in ["stacking", "direct"]:
                    geek_rating_config = config.models.get("geek_rating")
                    if geek_rating_config:
                        logger.info(f"  geek_rating model: {geek_rating_config.type}")
                    else:
                        logger.info(f"  geek_rating model: ard (default)")
            else:
                logger.info("  Using defaults (n_samples=500, geek_rating_mode=bayesian)")

        logger.info("\nTime Splits:")
        for i, split in enumerate(splits):
            logger.info(
                f"  {i+1}. Train ≤{split['train_through']}, "
                f"Tune {split['tune_start']}-{split['tune_through']}, "
                f"Test {split['test_start']}-{split['test_through']}"
            )

        if args.dry_run:
            logger.info("\n" + "=" * 60)
            logger.info("DRY RUN - Not executing")
            logger.info("=" * 60)
            return

        # Run the evaluation
        logger.info("\n" + "=" * 60)
        logger.info("STARTING EVALUATION")
        logger.info("=" * 60)

        run_time_based_evaluation(
            splits=splits,
            output_dir=args.output_dir,
            run_simulation=args.simulation,
        )

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
