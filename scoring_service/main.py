import os
from dotenv import load_dotenv
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from registered_model import RegisteredModel  # noqa: E402
from src.data.loader import BGGDataLoader  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.bigquery_uploader import BigQueryUploader  # noqa: E402
from src.models.geek_rating import calculate_geek_rating  # noqa: E402
from auth import GCPAuthenticator, AuthenticationError  # noqa: E402

load_dotenv()

# Set up proper credentials path relative to project root
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credentials_path and not os.path.isabs(credentials_path):
    # Convert relative path to absolute path based on project root
    absolute_credentials_path = os.path.join(project_root, credentials_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = absolute_credentials_path
    logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to: {absolute_credentials_path}")

# Initialize authentication
try:
    authenticator = GCPAuthenticator()
    GCP_PROJECT_ID = authenticator.project_id
    BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "bgg-models")

    # Verify bucket access
    if not authenticator.verify_bucket_access(BUCKET_NAME):
        logger.warning(
            f"Cannot access bucket {BUCKET_NAME}. Service may not function properly."
        )

    # Log authentication info
    auth_info = authenticator.get_authentication_info()
    logger.info(f"Authentication initialized successfully:")
    logger.info(f"  Project ID: {auth_info['project_id']}")
    logger.info(f"  Credentials Source: {auth_info['credentials_source']}")
    logger.info(f"  Running on GCP: {auth_info.get('running_on_gcp', False)}")
    logger.info(f"  Bucket Name: {BUCKET_NAME}")

except AuthenticationError as e:
    logger.error(f"Authentication failed: {str(e)}")
    raise
except Exception as e:
    logger.error(f"Unexpected error during authentication setup: {str(e)}")
    raise

# Debug logging for environment and paths
logger.info(f"Current Working Directory: {os.getcwd()}")
logger.info(f"Project Root: {project_root}")


class PredictGamesRequest(BaseModel):
    hurdle_model_name: str
    complexity_model_name: str
    rating_model_name: str
    users_rated_model_name: str
    hurdle_model_version: Optional[int] = None
    complexity_model_version: Optional[int] = None
    rating_model_version: Optional[int] = None
    users_rated_model_version: Optional[int] = None
    start_year: Optional[int] = 2024
    end_year: Optional[int] = 2029
    prior_rating: float = 5.5
    prior_weight: float = 2000
    output_path: Optional[str] = "data/predictions/game_predictions.parquet"
    upload_to_bigquery: bool = False
    bigquery_environment: str = "dev"


class PredictGamesResponse(BaseModel):
    job_id: str
    model_details: Dict[str, Any]
    scoring_parameters: Dict[str, Any]
    output_location: str
    bigquery_job_id: Optional[str] = None
    bigquery_table: Optional[str] = None


def construct_year_filter(
    start_year: Optional[int] = None, end_year: Optional[int] = None
) -> str:
    """
    Construct SQL WHERE clause for year filtering.
    """
    where_clauses = []
    if start_year is not None:
        where_clauses.append(f"year_published >= {start_year}")
    if end_year is not None:
        where_clauses.append(f"year_published < {end_year}")

    return " AND ".join(where_clauses) if where_clauses else ""


def load_game_data(
    start_year: Optional[int] = None, end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Load game data with optional year filtering.
    """
    config = load_config()
    loader = BGGDataLoader(config)

    where_clause = construct_year_filter(start_year, end_year)
    df = loader.load_data(where_clause=where_clause, preprocessor=None)

    return df.to_pandas()


def predict_hurdle_probabilities(
    hurdle_model: Any, features: pd.DataFrame, threshold: float = 0.5
) -> pd.Series:
    """
    Predict hurdle probabilities for games.
    """
    predicted_hurdle_prob = hurdle_model.predict_proba(features)[:, 1]
    return pd.Series(predicted_hurdle_prob, name="predicted_hurdle_prob")


def predict_game_characteristics(
    features: pd.DataFrame,
    complexity_model: Any,
    rating_model: Any,
    users_rated_model: Any,
    likely_games_mask: pd.Series,  # Kept for compatibility but not used
) -> pd.DataFrame:
    """
    Predict game complexity, rating, and users rated for all games.
    """
    results = pd.DataFrame(index=features.index)

    # Predict complexity for all games
    results["predicted_complexity"] = complexity_model.predict(features)

    # Add predicted complexity to features
    features_with_complexity = features.copy()
    features_with_complexity["predicted_complexity"] = results["predicted_complexity"]

    # Predict rating and users rated for all games
    results["predicted_rating"] = rating_model.predict(features_with_complexity)
    results["predicted_users_rated"] = users_rated_model.predict(
        features_with_complexity
    )

    # Ensure users is at least minimum threshold
    results["predicted_users_rated"] = np.maximum(
        np.round(np.expm1(results["predicted_users_rated"]) / 50) * 50,
        25,
    )

    return results


app = FastAPI(title="BGG Model Scoring Service")


@app.get("/health")
async def health_check():
    """Health check endpoint with authentication status."""
    try:
        auth_info = authenticator.get_authentication_info()
        bucket_accessible = authenticator.verify_bucket_access(BUCKET_NAME)

        return {
            "status": "healthy",
            "authentication": {
                "project_id": auth_info["project_id"],
                "credentials_source": auth_info["credentials_source"],
                "running_on_gcp": auth_info.get("running_on_gcp", False),
                "bucket_accessible": bucket_accessible,
                "bucket_name": BUCKET_NAME,
            },
            "service": "BGG Model Scoring Service",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "BGG Model Scoring Service",
        }


@app.get("/auth/status")
async def authentication_status():
    """Detailed authentication status endpoint."""
    try:
        # Use the existing authenticator instance that has the corrected credentials path
        auth_info = authenticator.get_authentication_info()

        # Test storage client creation
        try:
            client = authenticator.get_storage_client()
            auth_info["storage_client_created"] = True
        except Exception as e:
            auth_info["storage_client_created"] = False
            auth_info["storage_error"] = str(e)

        # Test bucket access
        try:
            bucket_accessible = authenticator.verify_bucket_access(BUCKET_NAME)
            auth_info["bucket_accessible"] = bucket_accessible
        except Exception as e:
            auth_info["bucket_accessible"] = False
            auth_info["bucket_error"] = str(e)

        auth_info["status"] = "success"
        auth_info["bucket_name"] = BUCKET_NAME

        return auth_info
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/predict_games", response_model=PredictGamesResponse)
async def predict_games_endpoint(request: PredictGamesRequest):
    """
    Predict game characteristics using registered models.
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Load models
        registered_hurdle_model = RegisteredModel(
            "hurdle", BUCKET_NAME, project_id=GCP_PROJECT_ID
        )
        registered_complexity_model = RegisteredModel(
            "complexity", BUCKET_NAME, project_id=GCP_PROJECT_ID
        )
        registered_rating_model = RegisteredModel(
            "rating", BUCKET_NAME, project_id=GCP_PROJECT_ID
        )
        registered_users_rated_model = RegisteredModel(
            "users_rated", BUCKET_NAME, project_id=GCP_PROJECT_ID
        )

        # Load model pipelines and registrations
        hurdle_pipeline, hurdle_registration = (
            registered_hurdle_model.load_registered_model(
                request.hurdle_model_name, request.hurdle_model_version
            )
        )
        complexity_pipeline, complexity_registration = (
            registered_complexity_model.load_registered_model(
                request.complexity_model_name, request.complexity_model_version
            )
        )
        rating_pipeline, rating_registration = (
            registered_rating_model.load_registered_model(
                request.rating_model_name, request.rating_model_version
            )
        )
        users_rated_pipeline, users_rated_registration = (
            registered_users_rated_model.load_registered_model(
                request.users_rated_model_name, request.users_rated_model_version
            )
        )

        # Extract threshold from hurdle model metadata
        threshold = hurdle_registration.get("metadata", {}).get("threshold", 0.5)

        # Load game data
        df_pandas = load_game_data(request.start_year, request.end_year)

        # Predict hurdle probabilities
        predicted_hurdle_prob = predict_hurdle_probabilities(
            hurdle_pipeline, df_pandas, threshold
        )

        # Prepare results DataFrame
        results = pd.DataFrame(
            {
                "game_id": df_pandas["game_id"],
                "name": df_pandas["name"],
                "year_published": df_pandas["year_published"],
                "predicted_hurdle_prob": predicted_hurdle_prob,
            }
        )

        # Identify games likely to receive ratings
        likely_games_mask = predicted_hurdle_prob >= threshold

        # Predict game characteristics
        characteristics = predict_game_characteristics(
            df_pandas,
            complexity_pipeline,
            rating_pipeline,
            users_rated_pipeline,
            likely_games_mask,
        )

        # Combine results
        results = pd.concat([results, characteristics], axis=1)

        # Calculate predicted geek rating
        results["predicted_geek_rating"] = calculate_geek_rating(
            results,
            prior_rating=request.prior_rating,
            prior_weight=request.prior_weight,
        )

        # Add model experiment identifiers and metadata
        results["hurdle_experiment"] = hurdle_registration["original_experiment"][
            "name"
        ]
        results["complexity_experiment"] = complexity_registration[
            "original_experiment"
        ]["name"]
        results["rating_experiment"] = rating_registration["original_experiment"][
            "name"
        ]
        results["users_rated_experiment"] = users_rated_registration[
            "original_experiment"
        ]["name"]

        # Add timestamp of scoring
        results["score_ts"] = datetime.now(timezone.utc).isoformat()

        # Save predictions locally
        local_output_path = request.output_path or f"/tmp/{job_id}_predictions.parquet"
        results.to_parquet(local_output_path, index=False)

        # Upload predictions to Google Cloud Storage
        storage_client = authenticator.get_storage_client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # Construct GCS path for predictions
        gcs_output_path = f"predictions/{job_id}_predictions.parquet"
        blob = bucket.blob(gcs_output_path)
        blob.upload_from_filename(local_output_path)

        # Use GCS path as output location
        output_path = f"gs://{BUCKET_NAME}/{gcs_output_path}"

        # Upload to BigQuery if requested
        bigquery_job_id = None
        bigquery_table = None
        if request.upload_to_bigquery:
            try:
                logger.info(
                    f"Uploading predictions to BigQuery ({request.bigquery_environment} environment)"
                )
                uploader = BigQueryUploader(environment=request.bigquery_environment)
                bigquery_job_id = uploader.upload_predictions(results, job_id)

                # Construct table reference using uploader's actual configuration
                bigquery_table = (
                    f"{uploader.project_id}.{uploader.dataset_id}.predictions"
                )

                logger.info(
                    f"Successfully uploaded to BigQuery table: {bigquery_table}"
                )
                logger.info(f"BigQuery job ID: {bigquery_job_id}")
            except Exception as e:
                import traceback

                logger.error(f"Failed to upload to BigQuery: {str(e)}")
                logger.error(f"BigQuery error traceback: {traceback.format_exc()}")
                # Raise HTTP exception to inform the client of BigQuery upload failure
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload predictions to BigQuery: {str(e)}",
                )

        # Prepare model details
        model_details = {
            "hurdle_model": {
                "name": hurdle_registration["name"],
                "version": hurdle_registration["version"],
                "experiment": hurdle_registration["original_experiment"]["name"],
            },
            "complexity_model": {
                "name": complexity_registration["name"],
                "version": complexity_registration["version"],
                "experiment": complexity_registration["original_experiment"]["name"],
            },
            "rating_model": {
                "name": rating_registration["name"],
                "version": rating_registration["version"],
                "experiment": rating_registration["original_experiment"]["name"],
            },
            "users_rated_model": {
                "name": users_rated_registration["name"],
                "version": users_rated_registration["version"],
                "experiment": users_rated_registration["original_experiment"]["name"],
            },
        }

        # Construct response
        response = PredictGamesResponse(
            job_id=job_id,
            model_details=model_details,
            scoring_parameters={
                "start_year": request.start_year,
                "end_year": request.end_year,
                "threshold": threshold,
                "prior_rating": request.prior_rating,
                "prior_weight": request.prior_weight,
            },
            output_location=output_path,
            bigquery_job_id=bigquery_job_id,
            bigquery_table=bigquery_table,
        )

        return response

    except Exception as e:
        import traceback

        logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Existing endpoints from previous implementation
@app.get("/models")
async def list_available_models():
    """List all available registered models across different types."""
    model_types = ["hurdle", "rating", "complexity", "users_rated"]
    available_models = {}

    for model_type in model_types:
        try:
            registered_model = RegisteredModel(
                model_type, BUCKET_NAME, project_id=GCP_PROJECT_ID
            )
            available_models[model_type] = registered_model.list_registered_models()
        except Exception:
            available_models[model_type] = []

    return available_models


@app.get("/model/{model_type}/{model_name}/info")
async def get_model_info(
    model_type: str, model_name: str, version: Optional[int] = None
):
    """Get detailed information about a specific registered model."""
    try:
        registered_model = RegisteredModel(
            model_type, BUCKET_NAME, project_id=GCP_PROJECT_ID
        )
        pipeline, registration = registered_model.load_registered_model(
            model_name, version
        )

        return {
            "registration": registration,
            "model_info": registration["model_info"],
            "original_experiment": registration["original_experiment"],
            "validation_metrics": registration.get("validation_metrics"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
