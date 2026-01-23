"""Command line interface for registering text embedding models."""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.models.experiments import ExperimentTracker  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from text_embeddings_service.registered_model import RegisteredTextEmbeddingModel  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_project_id():
    """Get the ML project ID from environment or config."""
    project_id = os.getenv("ML_PROJECT_ID") or os.getenv("GCP_PROJECT_ID")
    if project_id:
        return project_id

    config = load_config()
    return config.ml_project_id


def validate_environment():
    """Validate that we can get a project ID."""
    project_id = get_project_id()
    if not project_id:
        raise ValueError(
            "Could not determine ML project ID from environment or config"
        )


def register_model(
    experiment_name: str,
    registered_name: str,
    description: str,
    bucket_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register a text embedding model for production use.

    Args:
        experiment_name: Name of experiment to register.
        registered_name: Name to give the registered model.
        description: Description of the model.
        bucket_name: GCS bucket for storing registered models.
        metadata: Optional additional metadata.

    Returns:
        Registration details.
    """
    validate_environment()

    if bucket_name is None:
        config = load_config()
        bucket_name = config.get_bucket_name()
        logger.info(f"Using bucket from config: {bucket_name}")

    # Load experiment from text_embeddings tracker
    tracker = ExperimentTracker("text_embeddings")
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
    project_id = get_project_id()
    registered_model = RegisteredTextEmbeddingModel(
        bucket_name=bucket_name, project_id=project_id
    )

    try:
        registration = registered_model.register(
            experiment=experiment,
            name=registered_name,
            description=description,
            metadata=metadata,
        )

        print("\nSuccessfully registered text embedding model:")
        print(f"  Name: {registration['name']}")
        print(f"  Version: {registration['version']}")
        print(f"  Description: {registration['description']}")
        print(f"  Algorithm: {registration['model_info'].get('algorithm')}")
        print(f"  Embedding dim: {registration['model_info'].get('embedding_dim')}")
        print(f"  Vocab size: {registration['model_info'].get('vocab_size')}")
        print(f"  Registered at: {registration['registered_at']}")
        print(f"  Project: {project_id}")
        print(f"  Bucket: {bucket_name}")

        return registration

    except Exception as e:
        print("\nError registering model:")
        print(f"  {str(e)}")
        raise


def get_default_model_name() -> str:
    """Get default model name from config."""
    config = load_config()
    current_year = config.years.current
    return f"text-embeddings-v{current_year}"


def main():
    config = load_config()
    default_name = get_default_model_name()

    parser = argparse.ArgumentParser(
        description="Register a text embedding model for production use"
    )

    parser.add_argument(
        "--experiment", required=True, help="Name of experiment to register"
    )

    parser.add_argument(
        "--name",
        default=default_name,
        help=f"Name to give the registered model (default: {default_name})",
    )

    parser.add_argument(
        "--description",
        default=f"Production (v{config.years.current}) text embeddings for game descriptions",
        help="Description of the model",
    )

    parser.add_argument("--bucket", help="GCS bucket for storing registered models")

    args = parser.parse_args()

    try:
        register_model(
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
