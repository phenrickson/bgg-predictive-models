import argparse
import requests
from typing import Optional
import os
from dotenv import load_dotenv

import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

load_dotenv()

from src.utils.logging import setup_logging  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# Add scoring_service directory to path for auth module
scoring_service_path = os.path.dirname(__file__)
sys.path.insert(0, scoring_service_path)

from auth import get_authenticated_storage_client  # noqa: E402


def submit_scoring_request(
    service_url: str,
    start_year: int,
    end_year: int,
    hurdle_model: Optional[str] = None,
    complexity_model: Optional[str] = None,
    rating_model: Optional[str] = None,
    users_rated_model: Optional[str] = None,
    output_path: Optional[str] = None,
    prior_rating: Optional[float] = None,
    prior_weight: Optional[float] = None,
    upload_to_data_warehouse: bool = True,
) -> dict:
    """
    Submit request to scoring service and return response.

    Args:
        service_url: URL of the scoring service
        start_year: Start year for predictions
        end_year: End year for predictions
        hurdle_model: Optional override for hurdle model name
        complexity_model: Optional override for complexity model name
        rating_model: Optional override for rating model name
        users_rated_model: Optional override for users rated model name
        output_path: Optional override for predictions output path
        prior_rating: Optional override for prior mean rating
        prior_weight: Optional override for prior weight
        upload_to_data_warehouse: Whether to upload results to data warehouse

    Returns:
        Response from scoring service
    """
    logger = setup_logging()
    config = load_config()

    # Get model names and parameters from config
    if config.scoring:
        model_config = config.scoring.models
        param_config = config.scoring.parameters
        output_config = config.scoring.output

        payload = {
            "hurdle_model_name": hurdle_model or model_config.get("hurdle"),
            "complexity_model_name": complexity_model or model_config.get("complexity"),
            "rating_model_name": rating_model or model_config.get("rating"),
            "users_rated_model_name": users_rated_model
            or model_config.get("users_rated"),
            "start_year": start_year,
            "end_year": end_year,
            "prior_rating": prior_rating or param_config.get("prior_rating", 5.5),
            "prior_weight": prior_weight or param_config.get("prior_weight", 2000),
            "upload_to_data_warehouse": upload_to_data_warehouse,
        }

        if output_path:
            payload["output_path"] = output_path
        elif "predictions_path" in output_config:
            payload["output_path"] = output_config["predictions_path"]
    else:
        # If no scoring config, use provided values or defaults
        payload = {
            "hurdle_model_name": hurdle_model,
            "complexity_model_name": complexity_model,
            "rating_model_name": rating_model,
            "users_rated_model_name": users_rated_model,
            "start_year": start_year,
            "end_year": end_year,
            "prior_rating": prior_rating or 5.5,
            "prior_weight": prior_weight or 2000,
            "upload_to_data_warehouse": upload_to_data_warehouse,
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

    # Initialize GCS client using new authentication
    storage_client = get_authenticated_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Determine local save path from config
    if not local_path:
        config = load_config()
        predictions_path = os.path.join("data", "predictions")
        if (
            "predictions" in config.models
            and "predictions_path" in config.models["predictions"]
        ):
            predictions_path = config.models["predictions"]["predictions_path"]
        local_path = os.path.join(predictions_path, os.path.basename(blob_path))

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

    # Model selection arguments (optional now since they can come from config)
    parser.add_argument("--hurdle-model", help="Override hurdle model name from config")
    parser.add_argument(
        "--complexity-model",
        help="Override complexity model name from config",
    )
    parser.add_argument("--rating-model", help="Override rating model name from config")
    parser.add_argument(
        "--users-rated-model",
        help="Override users rated model name from config",
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

    # Data warehouse arguments
    parser.add_argument(
        "--upload-to-bigquery",
        action="store_true",
        default=True,
        help="Upload predictions to BigQuery (default: True)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading predictions to data warehouse",
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

    # Determine upload setting
    upload_to_data_warehouse = args.upload_to_bigquery and not args.no_upload

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
            upload_to_data_warehouse=upload_to_data_warehouse,
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

        # Log data warehouse information if available
        if response.get("data_warehouse_job_id"):
            logger.info(f"Data Warehouse Job ID: {response['data_warehouse_job_id']}")
            logger.info(f"Data Warehouse Table: {response['data_warehouse_table']}")

        # Download predictions if requested
        if args.download:
            download_predictions(response["output_location"])

    except Exception as e:
        logger.error(f"Error during scoring: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
