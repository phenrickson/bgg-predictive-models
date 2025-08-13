#!/usr/bin/env python3
"""Script to run time-based model evaluation with specific configuration."""

import sys
import argparse
from src.models.time_based_evaluation import (
    run_time_based_evaluation,
    generate_time_splits,
)


def get_default_model_args():
    """Get default model arguments."""
    return {
        "hurdle": {
            "preprocessor-type": "linear",
            "model": "logistic",
        },
        "complexity": {
            "preprocessor-type": "tree",
            "model": "catboost",
            "use-sample-weights": True,
        },
        "rating": {
            "preprocessor-type": "tree",
            "model": "catboost",
            "min-ratings": 5,
            "use-sample-weights": True,
        },
        "users_rated": {
            "preprocessor-type": "tree",
            "model": "lightgbm",
            "min-ratings": 0,
        },
    }


def parse_model_args(args_list):
    """Parse model arguments from command line format into dictionary."""
    # Start with default arguments
    model_args = get_default_model_args()

    if not args_list:
        return model_args

    for arg in args_list:
        try:
            # Split into model.key=value format
            model_key, value = arg.split("=")
            model, key = model_key.split(".")

            # Initialize model dict if not exists
            if model not in model_args:
                model_args[model] = {}

            # Convert value to appropriate type
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)

            model_args[model][key] = value
        except ValueError:
            print(f"Warning: Skipping invalid model argument: {arg}")

    return model_args


def main():
    parser = argparse.ArgumentParser(description="Run time-based model evaluation")
    parser.add_argument(
        "--start-year",
        type=int,
        default=2017,
        help="First year for evaluation",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2021,
        help="Last year for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/experiments",
        help="Output directory for experiments",
    )
    parser.add_argument(
        "--model-args",
        nargs="+",
        help="Model arguments in format 'model.key=value'",
    )

    args = parser.parse_args()

    # Parse model arguments (will include defaults if none provided)
    model_args = parse_model_args(args.model_args)

    # Generate time splits
    splits = generate_time_splits(
        start_year=args.start_year,
        end_year=args.end_year,
    )

    try:
        # Run evaluation
        run_time_based_evaluation(
            splits=splits,
            model_args=model_args,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"Error running evaluation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
