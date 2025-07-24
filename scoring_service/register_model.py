"""Command line interface for registering models."""

import argparse
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import logging 

from src.models.experiments import ExperimentTracker
from scoring_service.registered_model import RegisteredModel, ModelValidationError

# load environment variaables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"{os.getenv("GCP_PROJECT_ID")}")


# Default validation metrics for each model type
DEFAULT_VALIDATION_METRICS: Dict[str, Dict[str, float]] = {
    "hurdle": {"min_auc": 0.7, "min_accuracy": 0.7},
    "rating": {"min_r2": 0.5, "max_rmse": 1.0},
    "complexity": {"min_r2": 0.4, "max_rmse": 0.8},
    "users_rated": {"min_r2": 0.5, "max_rmse": 1.0},
}


def register_model(
    model_type: str,
    experiment_name: str,
    registered_name: str,
    description: str,
    bucket_name: Optional[str] = None,
    validation_metrics: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register a model for production use.

    Args:
        model_type: Type of model to register
        experiment_name: Name of experiment to register
        registered_name: Name to give the registered model
        description: Description of the model
        bucket_name: GCS bucket for storing registered models
        validation_metrics: Optional metric thresholds for validation
        metadata: Optional additional metadata

    Returns:
        Registration details
    """
    # Get bucket name from environment if not provided
    if bucket_name is None:
        bucket_name = os.getenv("GCS_BUCKET_NAME", "bgg-models")

    # Load experiment
    tracker = ExperimentTracker(model_type)
    experiments = tracker.list_experiments()
    matching_experiments = [
        exp for exp in experiments if exp["name"] == experiment_name
    ]

    if not matching_experiments:
        raise ValueError(f"No experiments found matching '{experiment_name}'")

    # Get latest version
    latest_experiment = max(matching_experiments, key=lambda x: x["version"])
    experiment = tracker.load_experiment(
        latest_experiment["name"], latest_experiment["version"]
    )

    # Use default validation metrics if none provided
    if validation_metrics is None:
        validation_metrics = DEFAULT_VALIDATION_METRICS.get(model_type, {})

    # Create registered model manager
    project_id = os.getenv("GCP_PROJECT_ID")
    registered_model = RegisteredModel(
        model_type=model_type, bucket_name=bucket_name, project_id=project_id
    )

    try:
        # Register the model
        registration = registered_model.register(
            experiment=experiment,
            name=registered_name,
            description=description,
            validation_metrics=validation_metrics,
            metadata=metadata,
        )

        print(f"\nSuccessfully registered model:")
        print(f"  Name: {registration['name']}")
        print(f"  Version: {registration['version']}")
        print(f"  Description: {registration['description']}")
        print(f"  Registered at: {registration['registered_at']}")
        print("\nValidation Metrics:")
        for metric, value in registration["validation_metrics"].items():
            print(f"  {metric}: {value}")

        return registration

    except ModelValidationError as e:
        print(f"\nModel validation failed:")
        print(f"  {str(e)}")
        print(
            "\nModel was not registered. Please check the validation metrics and try again."
        )
        raise

    except Exception as e:
        print(f"\nError registering model:")
        print(f"  {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Register a model for production use")

    parser.add_argument(
        "--model-type",
        required=True,
        choices=["hurdle", "rating", "complexity", "users_rated"],
        help="Type of model to register",
    )

    parser.add_argument(
        "--experiment", required=True, help="Name of experiment to register"
    )

    parser.add_argument(
        "--name", required=True, help="Name to give the registered model"
    )

    parser.add_argument("--description", required=True, help="Description of the model")

    parser.add_argument("--bucket", help="GCS bucket for storing registered models")

    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip validation metrics check"
    )

    args = parser.parse_args()

    try:
        # Register the model
        registration = register_model(
            model_type=args.model_type,
            experiment_name=args.experiment,
            registered_name=args.name,
            description=args.description,
            bucket_name=args.bucket,
            validation_metrics=(
                None
                if args.skip_validation
                else DEFAULT_VALIDATION_METRICS.get(args.model_type)
            ),
        )

        print("\nModel registration successful!")

    except (ModelValidationError, ValueError) as e:
        print(f"\nError: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
