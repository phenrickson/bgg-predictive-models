import argparse
import requests
from typing import Optional
import os
from dotenv import load_dotenv
from google.cloud import storage

import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils.logging import setup_logging  # noqa: E402

# Load environment variables
load_dotenv()


def submit_scoring_request(
    service_url: str,
    start_year: int,
    end_year: int,
    hurdle_model: str = "hurdle-v2025",
    complexity_model: str = "complexity-v2025",
    rating_model: str = "rating-v2025",
    users_rated_model: str = "users_rated-v2025",
    output_path: Optional[str] = None,
    prior_rating: float = 5.5,
    prior_weight: float = 2000,
) -> dict:
    """
    Submit request to scoring service and return response.

    Args:
        service_url: URL of the scoring service
        start_year: Start year for predictions
        end_year: End year for predictions
        hurdle_model: Name of hurdle model to use
        complexity_model: Name of complexity model to use
        rating_model: Name of rating model to use
        users_rated_model: Name of users rated model to use
        output_path: Optional local path to save predictions
        prior_rating: Prior mean rating for Bayesian average
        prior_weight: Weight given to prior rating

    Returns:
        Response from scoring service
    """
    logger = setup_logging()

    payload = {
        "hurdle_model_name": hurdle_model,
        "complexity_model_name": complexity_model,
        "rating_model_name": rating_model,
        "users_rated_model_name": users_rated_model,
        "start_year": start_year,
        "end_year": end_year,
        "prior_rating": prior_rating,
        "prior_weight": prior_weight,
    }

    if output_path:
        payload["output_path"] = output_path

    try:
        logger.info(f"Submitting scoring request to {service_url}")
        response = requests.post(
            f"{service_url}/predict_games",
            json=payload,
            timeout=600,  # 10-minute timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error submitting scoring request: {e}")
        raise


def download_predictions(gcs_path: str, local_path: Optional[str] = None) -> str:
    """
    Download predictions from Google Cloud Storage.

    Args:
        gcs_path: Full GCS path to predictions file
        local_path: Optional local path to save predictions

    Returns:
        Path where predictions were saved
    """
    logger = setup_logging()

    # Remove 'gs://' prefix if present
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    # Split bucket and path
    bucket_name, blob_path = gcs_path.split("/", 1)

    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Determine local save path
    if not local_path:
        local_path = os.path.join("data", "predictions", os.path.basename(blob_path))

    # Ensure directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Download file
    logger.info(f"Downloading predictions from {gcs_path} to {local_path}")
    blob.download_to_filename(local_path)

    logger.info(f"Predictions downloaded to: {local_path}")
    return local_path


def main():
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Submit scoring request to BGG Scoring Service"
    )

    # Model selection arguments
    parser.add_argument(
        "--hurdle-model", default="hurdle-v2025", help="Name of hurdle model to use"
    )
    parser.add_argument(
        "--complexity-model",
        default="complexity-v2025",
        help="Name of complexity model to use",
    )
    parser.add_argument(
        "--rating-model", default="rating-v2025", help="Name of rating model to use"
    )
    parser.add_argument(
        "--users-rated-model",
        default="users_rated-v2025",
        help="Name of users rated model to use",
    )

    # Year range arguments
    parser.add_argument(
        "--start-year", type=int, default=2024, help="Start year for predictions"
    )
    parser.add_argument(
        "--end-year", type=int, default=2029, help="End year for predictions"
    )

    # Service and output arguments
    parser.add_argument(
        "--service-url", default="http://localhost:8080", help="URL of scoring service"
    )
    parser.add_argument(
        "--output-path", help="Optional local output path for predictions"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download predictions after scoring"
    )

    # Prior rating arguments
    parser.add_argument(
        "--prior-rating",
        type=float,
        default=5.5,
        help="Prior mean rating for Bayesian average",
    )
    parser.add_argument(
        "--prior-weight", type=float, default=2000, help="Weight given to prior rating"
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Submit scoring request
        logger.info("Submitting scoring request...")
        response = submit_scoring_request(
            service_url=args.service_url,
            start_year=args.start_year,
            end_year=args.end_year,
            hurdle_model=args.hurdle_model,
            complexity_model=args.complexity_model,
            rating_model=args.rating_model,
            users_rated_model=args.users_rated_model,
            output_path=args.output_path,
            prior_rating=args.prior_rating,
            prior_weight=args.prior_weight,
        )

        # Log job details
        logger.info(f"Scoring Job ID: {response['job_id']}")
        logger.info("Model Details:")
        for model_type, details in response["model_details"].items():
            logger.info(f"{model_type.capitalize()} Model:")
            logger.info(f"  Name: {details['name']}")
            logger.info(f"  Version: {details['version']}")
            logger.info(f"  Experiment: {details['experiment']}")

        logger.info("Scoring Parameters:")
        for param, value in response["scoring_parameters"].items():
            logger.info(f"{param}: {value}")

        logger.info(f"Predictions Location: {response['output_location']}")

        # Download predictions if requested
        if args.download:
            download_predictions(response["output_location"])

    except Exception as e:
        logger.error(f"Error during scoring: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
