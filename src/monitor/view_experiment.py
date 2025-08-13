"""
Script to load and inspect a complexity model experiment.

This script demonstrates how to load an experiment for the complexity model
and print out its metadata and details.
"""

import argparse
import json
from typing import Optional

from src.models.experiments import ExperimentTracker


def load_and_print_experiment(
    model_type: str = "complexity",
    experiment_name: str = "baseline_complexity",
    version: Optional[int] = None,
):
    """
    Load and print details of a specific experiment.

    Args:
        model_type: Type of model (default: 'complexity')
        experiment_name: Name of the experiment to load
        version: Optional specific version of the experiment
    """
    try:
        # Initialize ExperimentTracker
        tracker = ExperimentTracker(model_type)

        # Load the experiment
        experiment = tracker.load_experiment(experiment_name, version)

        # Print experiment details
        print("Experiment Details:")
        print("-" * 40)
        print(f"Name: {experiment_name}")
        print(f"Model Type: {model_type}")
        print(f"Version: {version or 'Latest'}")
        print(f"Experiment Directory: {experiment.exp_dir}")

        # Print metadata
        print("\nMetadata:")
        print("-" * 40)
        print(json.dumps(experiment.metadata, indent=2))

        # Print additional experiment information
        print("\nExperiment Information:")
        print("-" * 40)

        # Retrieve metrics for different datasets
        print("\nMetrics:")
        print("-" * 40)
        datasets = ["train", "tune", "test"]

        for dataset in datasets:
            try:
                print(f"\n{dataset.upper()} Dataset Metrics:")
                metrics = experiment.get_metrics(dataset=dataset)
                print(json.dumps(metrics, indent=2))
            except Exception as e:
                print(f"Could not retrieve {dataset} metrics: {e}")

        # Try to get model parameters
        try:
            parameters = experiment.get_parameters()
            print("\nModel Parameters:")
            print(json.dumps(parameters, indent=2))
        except Exception as e:
            print(f"Could not retrieve parameters: {e}")

    except Exception as e:
        print(f"Error loading experiment: {e}")


def main():
    parser = argparse.ArgumentParser(description="Load and inspect an experiment")
    parser.add_argument(
        "--model-type",
        type=str,
        default="complexity",
        help="Type of model (default: complexity)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline_complexity",
        help="Name of experiment to load (default: baseline_complexity)",
    )
    parser.add_argument(
        "--version",
        type=int,
        required=False,
        help="Optional specific version of the experiment",
    )

    args = parser.parse_args()

    load_and_print_experiment(
        model_type=args.model_type,
        experiment_name=args.experiment,
        version=args.version,
    )


if __name__ == "__main__":
    main()
