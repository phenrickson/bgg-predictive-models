"""
Script to load and validate predictions from multiple experiments.

This script allows loading predictions from multiple experiments,
checking metadata consistency, and providing a summary of the loaded predictions.
"""

import argparse
import logging
from pathlib import Path
import polars as pl
import json

from src.models.experiments import ExperimentTracker


def load_experiment_predictions(experiment_names, model_type="rating"):
    """
    Load predictions from multiple experiments and validate their metadata.

    Args:
        experiment_names (list): List of experiment names to load
        model_type (str, optional): Type of model. Defaults to 'rating'.

    Returns:
        dict: A dictionary containing predictions and metadata for each experiment
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Initialize experiment tracker
    tracker = ExperimentTracker(model_type)

    # Dictionary to store results
    experiment_results = {}

    # Track metadata for consistency checking
    train_years = set()
    tune_years = set()
    test_years = set()

    # Load predictions for each experiment
    for exp_name in experiment_names:
        try:
            # Load the latest version of the experiment
            experiment = tracker.load_experiment(exp_name)

            # Load metadata
            metadata = experiment.metadata

            # Extract years from metadata
            train_year = metadata.get("train_year")
            tune_year = metadata.get("tune_year")
            test_year = metadata.get("test_year")

            # Add years to tracking sets
            if train_year is not None:
                train_years.add(train_year)
            if tune_year is not None:
                tune_years.add(tune_year)
            if test_year is not None:
                test_years.add(test_year)

            # Load predictions for train, tune, and test sets
            experiment_data = {
                "name": exp_name,
                "metadata": metadata,
                "train_predictions": None,
                "tune_predictions": None,
                "test_predictions": None,
            }

            # Try to load predictions for each dataset
            for dataset in ["train", "tune", "test"]:
                try:
                    predictions = experiment.get_predictions(dataset)
                    experiment_data[f"{dataset}_predictions"] = predictions
                    logger.info(f"Loaded {dataset} predictions for {exp_name}")
                except ValueError as e:
                    logger.warning(f"No {dataset} predictions for {exp_name}: {e}")

            experiment_results[exp_name] = experiment_data

        except Exception as e:
            logger.error(f"Error loading experiment {exp_name}: {e}")

    # Check year consistency
    if len(train_years) > 1:
        logger.warning(f"Inconsistent train years across experiments: {train_years}")
    if len(tune_years) > 1:
        logger.warning(f"Inconsistent tune years across experiments: {tune_years}")
    if len(test_years) > 1:
        logger.warning(f"Inconsistent test years across experiments: {test_years}")

    return experiment_results


def main():
    """
    Command-line interface for loading experiment predictions.
    """
    parser = argparse.ArgumentParser(
        description="Load and validate experiment predictions"
    )
    parser.add_argument(
        "experiments", nargs="+", help="Names of experiments to load predictions from"
    )
    parser.add_argument(
        "--model-type", default="rating", help="Type of model (default: rating)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output JSON file with experiment predictions",
    )

    args = parser.parse_args()

    # Load predictions
    results = load_experiment_predictions(args.experiments, args.model_type)

    # Output results
    if args.output:
        # Save results to JSON file
        with open(args.output, "w") as f:
            json.dump(
                {
                    exp: {
                        "metadata": data["metadata"],
                        "train_predictions_count": (
                            len(data["train_predictions"])
                            if data["train_predictions"] is not None
                            else 0
                        ),
                        "tune_predictions_count": (
                            len(data["tune_predictions"])
                            if data["tune_predictions"] is not None
                            else 0
                        ),
                        "test_predictions_count": (
                            len(data["test_predictions"])
                            if data["test_predictions"] is not None
                            else 0
                        ),
                    }
                    for exp, data in results.items()
                },
                f,
                indent=2,
            )
        print(f"Results saved to {args.output}")

    # Print summary to console
    print("\nExperiment Predictions Summary:")
    for exp, data in results.items():
        print(f"\n{exp}:")
        print(
            f"  Train Predictions: {len(data['train_predictions']) if data['train_predictions'] is not None else 'None'}"
        )
        print(
            f"  Tune Predictions:  {len(data['tune_predictions']) if data['tune_predictions'] is not None else 'None'}"
        )
        print(
            f"  Test Predictions:  {len(data['test_predictions']) if data['test_predictions'] is not None else 'None'}"
        )

        # Print key metadata
        metadata = data["metadata"]
        print("  Metadata:")
        for key in ["train_year", "tune_year", "test_year", "target", "model_type"]:
            if key in metadata:
                print(f"    {key}: {metadata[key]}")


if __name__ == "__main__":
    main()
