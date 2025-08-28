"""Command line interface for registering models."""

import argparse
import os
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.models.experiments import ExperimentTracker  # noqa: E402
from scoring_service.registered_model import RegisteredModel  # noqa: E402

# load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"{os.getenv('GCP_PROJECT_ID')}")


def register_model(
    model_type: str,
    experiment_name: str,
    registered_name: str,
    description: str,
    bucket_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register a model for production use.

    Args:
        model_type: Type of model to register
        experiment_name: Name of experiment to register
        registered_name: Name to give the registered model
        description: Description of the model
        bucket_name: GCS bucket for storing registered models
        metadata: Optional additional metadata

    Returns:
        Registration details
    """
    # Get bucket name from environment if not provided
    if bucket_name is None:
        bucket_name = os.getenv("GCS_BUCKET_NAME", "bgg-predictive-models")

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
            metadata=metadata,
        )

        print("\nSuccessfully registered model:")
        print(f"  Name: {registration['name']}")
        print(f"  Version: {registration['version']}")
        print(f"  Description: {registration['description']}")
        print(f"  Registered at: {registration['registered_at']}")

        return registration

    except Exception as e:
        print("\nError registering model:")
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

    args = parser.parse_args()

    try:
        # Register the model
        registration = register_model(  # noqa: F841
            model_type=args.model_type,
            experiment_name=args.experiment,
            registered_name=args.name,
            description=args.description,
            bucket_name=args.bucket,
        )

        print("\nModel registration successful!")

    except ValueError as e:
        print(f"\nError: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
