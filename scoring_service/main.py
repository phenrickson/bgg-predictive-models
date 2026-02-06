import os
from dotenv import load_dotenv
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import logging

import numpy as np
import pandas as pd
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import bigquery

import sys

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from registered_model import RegisteredModel  # noqa: E402
from src.data.loader import BGGDataLoader  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.data.bigquery_uploader import DataWarehousePredictionUploader  # noqa: E402
from src.models.outcomes.simulation import simulate_batch, precompute_cholesky  # noqa: E402
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

    # Get bucket name and environment prefix from config
    config = load_config()
    BUCKET_NAME = config.get_bucket_name()
    ENVIRONMENT_PREFIX = config.get_environment_prefix()

    # Verify bucket access
    if not authenticator.verify_bucket_access(BUCKET_NAME):
        logger.warning(
            f"Cannot access bucket {BUCKET_NAME}. Service may not function properly."
        )

    # Log authentication info
    auth_info = authenticator.get_authentication_info()
    logger.info("Authentication initialized successfully:")
    logger.info(f"  Project ID: {auth_info['project_id']}")
    logger.info(f"  Credentials Source: {auth_info['credentials_source']}")
    logger.info(f"  Running on GCP: {auth_info.get('running_on_gcp', False)}")
    logger.info(f"  Bucket Name: {BUCKET_NAME}")
    logger.info(f"  Environment: {ENVIRONMENT_PREFIX}")

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
    geek_rating_model_name: str
    hurdle_model_version: Optional[int] = None
    complexity_model_version: Optional[int] = None
    rating_model_version: Optional[int] = None
    users_rated_model_version: Optional[int] = None
    geek_rating_model_version: Optional[int] = None
    start_year: Optional[int] = 2024
    end_year: Optional[int] = 2029
    output_path: Optional[str] = "data/predictions/game_predictions.parquet"
    upload_to_data_warehouse: bool = True
    game_ids: Optional[List[int]] = None
    use_change_detection: bool = False  # NEW: Enable incremental scoring
    max_games: Optional[int] = 50000    # NEW: Limit for change detection mode


class PredictGamesResponse(BaseModel):
    job_id: str
    model_details: Dict[str, Any]
    scoring_parameters: Dict[str, Any]
    output_location: Optional[str] = None
    data_warehouse_job_id: Optional[str] = None
    data_warehouse_table: Optional[str] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    games_scored: Optional[int] = None      # NEW: Number of games actually scored
    skipped_reason: Optional[str] = None    # NEW: Why scoring was skipped (e.g., "no_changes")


class SimulateGamesRequest(BaseModel):
    hurdle_model_name: str
    complexity_model_name: str
    rating_model_name: str
    users_rated_model_name: str
    geek_rating_model_name: str
    hurdle_model_version: Optional[int] = None
    complexity_model_version: Optional[int] = None
    rating_model_version: Optional[int] = None
    users_rated_model_version: Optional[int] = None
    geek_rating_model_version: Optional[int] = None
    game_ids: Optional[List[int]] = None
    start_year: Optional[int] = 2024
    end_year: Optional[int] = 2029
    n_samples: int = 500
    random_state: int = 42
    output_path: Optional[str] = "data/predictions/game_simulations.parquet"
    upload_to_data_warehouse: bool = True
    use_change_detection: bool = False
    max_games: Optional[int] = 50000


class SimulateGamesResponse(BaseModel):
    job_id: str
    model_details: Dict[str, Any]
    n_samples: int
    games_simulated: int
    results: Optional[List[Dict[str, Any]]] = None
    output_location: Optional[str] = None
    data_warehouse_job_id: Optional[str] = None
    data_warehouse_table: Optional[str] = None
    skipped_reason: Optional[str] = None


class PredictComplexityRequest(BaseModel):
    complexity_model_name: str
    complexity_model_version: Optional[int] = None
    max_games: int = 25000
    game_ids: Optional[List[int]] = None


class PredictComplexityResponse(BaseModel):
    job_id: str
    model_details: Dict[str, Any]
    games_scored: int
    table_id: Optional[str] = None
    bq_job_id: Optional[str] = None
    predictions: Optional[List[Dict[str, Any]]] = None


class PredictHurdleRequest(BaseModel):
    hurdle_model_name: str
    hurdle_model_version: Optional[int] = None
    game_ids: Optional[List[int]] = None


class PredictHurdleResponse(BaseModel):
    job_id: str
    model_details: Dict[str, Any]
    games_scored: int
    table_id: Optional[str] = None
    bq_job_id: Optional[str] = None
    predictions: Optional[List[Dict[str, Any]]] = None


class PredictRatingRequest(BaseModel):
    rating_model_name: str
    rating_model_version: Optional[int] = None
    game_ids: Optional[List[int]] = None


class PredictRatingResponse(BaseModel):
    job_id: str
    model_details: Dict[str, Any]
    games_scored: int
    table_id: Optional[str] = None
    bq_job_id: Optional[str] = None
    predictions: Optional[List[Dict[str, Any]]] = None


class PredictUsersRatedRequest(BaseModel):
    users_rated_model_name: str
    users_rated_model_version: Optional[int] = None
    game_ids: Optional[List[int]] = None


class PredictUsersRatedResponse(BaseModel):
    job_id: str
    model_details: Dict[str, Any]
    games_scored: int
    table_id: Optional[str] = None
    bq_job_id: Optional[str] = None
    predictions: Optional[List[Dict[str, Any]]] = None



def get_registered_model(model_type: str) -> RegisteredModel:
    """Create a RegisteredModel for the given model type."""
    return RegisteredModel(model_type, BUCKET_NAME, project_id=GCP_PROJECT_ID)


def load_game_data(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    game_ids: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load game data with embeddings from data warehouse via BGGDataLoader.

    Joins games_features with description embeddings so all models
    get the emb_0..emb_N columns they were trained with.

    If game_ids is provided, it takes precedence over year filtering.
    """
    config = load_config()
    dw_config = config.get_data_warehouse_config()
    loader = BGGDataLoader(dw_config)

    # Build WHERE clause with f. prefix for the joined query
    if game_ids:
        game_ids_str = ",".join(str(gid) for gid in game_ids)
        where_clause = f"f.game_id IN ({game_ids_str})"
    else:
        where_parts = []
        if start_year is not None:
            where_parts.append(f"f.year_published >= {start_year}")
        if end_year is not None:
            where_parts.append(f"f.year_published < {end_year}")
        where_clause = " AND ".join(where_parts) if where_parts else ""

    df = loader.load_data_with_embeddings(where_clause=where_clause)
    return df.to_pandas()


def load_games_for_main_scoring(
    start_year: int,
    end_year: int,
    max_games: int = 50000,
    hurdle_model_version: Optional[int] = None,
    complexity_model_version: Optional[int] = None,
    rating_model_version: Optional[int] = None,
    users_rated_model_version: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load games that need main predictions (hurdle, rating, users_rated).

    Returns games that are:
    - In the year range AND
    - Either never scored OR have changed features since last scoring
      OR have been scored with a different model version

    Args:
        start_year: Start year for predictions (inclusive)
        end_year: End year for predictions (exclusive)
        max_games: Maximum number of games to load
        hurdle_model_version: Target hurdle model version (rescore if different)
        complexity_model_version: Target complexity model version (rescore if different)
        rating_model_version: Target rating model version (rescore if different)
        users_rated_model_version: Target users_rated model version (rescore if different)

    Returns:
        DataFrame with game features for scoring
    """
    config = load_config()
    dw_config = config.get_data_warehouse_config()
    loader = BGGDataLoader(dw_config)

    df = loader.load_changed_games_with_embeddings(
        start_year=start_year,
        end_year=end_year,
        ml_project_id=config.ml_project_id,
        max_games=max_games,
        hurdle_model_version=hurdle_model_version,
        complexity_model_version=complexity_model_version,
        rating_model_version=rating_model_version,
        users_rated_model_version=users_rated_model_version,
    )
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
    raw_log_users_rated = users_rated_model.predict(features_with_complexity)

    # Transform from log scale to count scale
    results["predicted_users_rated"] = np.maximum(
        np.round(np.expm1(raw_log_users_rated) / 50) * 50,
        25,
    )

    # Keep log-scale version for geek_rating model
    results["predicted_users_rated_log"] = np.log1p(results["predicted_users_rated"])

    return results


def load_games_for_complexity_scoring(
    game_ids: Optional[List[int]] = None,
    max_games: int = 25000,
    complexity_model_version: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load games that need complexity predictions.

    If game_ids is provided, load only those games (skip change detection).
    Otherwise, returns games that are:
    - New (never scored)
    - Have changed features (detected via game_features_hash.last_updated)
    - Have been scored with a different model version

    Args:
        game_ids: Optional list of specific game IDs to load
        max_games: Maximum number of games to load (default: 25000)
        complexity_model_version: Target model version (rescore if different)
    """
    if game_ids:
        # Load specific games by ID using existing helper
        logger.info(f"Loading {len(game_ids)} specific games for complexity predictions...")
        return load_game_data(game_ids=game_ids)
    else:
        # Use change detection logic with BGGDataLoader
        logger.info(f"Loading up to {max_games} games needing complexity predictions...")
        config = load_config()
        data_warehouse_config = config.get_data_warehouse_config()
        loader = BGGDataLoader(data_warehouse_config)

        # Build version mismatch condition
        version_condition = ""
        if complexity_model_version is not None:
            version_condition = f"OR lp.complexity_model_version != {complexity_model_version}"

        where_clause = f"""
        game_id IN (
          SELECT gf.game_id
          FROM `bgg-data-warehouse.analytics.games_features` gf
          LEFT JOIN `bgg-data-warehouse.staging.game_features_hash` fh
            ON gf.game_id = fh.game_id
          LEFT JOIN (
            SELECT
              game_id,
              score_ts,
              complexity_model_version,
              ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY score_ts DESC) as rn
            FROM `bgg-predictive-models.raw.complexity_predictions`
          ) lp ON gf.game_id = lp.game_id AND lp.rn = 1
          WHERE
            gf.year_published IS NOT NULL
            AND (
              lp.game_id IS NULL
              OR fh.last_updated > lp.score_ts
              {version_condition}
            )
          LIMIT {max_games}
        )
        """

        df = loader.load_data(where_clause=where_clause, preprocessor=None)
        logger.info(f"Found {len(df)} games to score")
        return df.to_pandas()


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
            client = authenticator.get_storage_client()  # noqa
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
        registered_hurdle_model = get_registered_model("hurdle")
        registered_complexity_model = get_registered_model("complexity")
        registered_rating_model = get_registered_model("rating")
        registered_users_rated_model = get_registered_model("users_rated")
        registered_geek_rating_model = get_registered_model("geek_rating")

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
        geek_rating_pipeline, geek_rating_registration = (
            registered_geek_rating_model.load_registered_model(
                request.geek_rating_model_name, request.geek_rating_model_version
            )
        )

        # Extract threshold from hurdle model metadata
        threshold = hurdle_registration.get("metadata", {}).get("threshold", 0.5)

        # Load game data
        if request.game_ids:
            # Specific games requested - load directly
            logger.info(f"Loading {len(request.game_ids)} specific games")
            df_pandas = load_game_data(game_ids=request.game_ids)
        elif request.use_change_detection:
            # Use change detection to find games needing scoring
            logger.info("Using change detection to find games needing scoring")
            df_pandas = load_games_for_main_scoring(
                request.start_year or 2024,
                request.end_year or 2029,
                max_games=request.max_games or 50000,
                hurdle_model_version=hurdle_registration["version"],
                complexity_model_version=complexity_registration["version"],
                rating_model_version=rating_registration["version"],
                users_rated_model_version=users_rated_registration["version"],
            )
            if len(df_pandas) == 0:
                logger.info("No games need scoring - all features unchanged")
                return PredictGamesResponse(
                    job_id=job_id,
                    model_details={
                        "hurdle": {"name": request.hurdle_model_name},
                        "complexity": {"name": request.complexity_model_name},
                        "rating": {"name": request.rating_model_name},
                        "users_rated": {"name": request.users_rated_model_name},
                        "geek_rating": {"name": request.geek_rating_model_name},
                    },
                    scoring_parameters={
                        "start_year": request.start_year,
                        "end_year": request.end_year,
                    },
                    games_scored=0,
                    skipped_reason="no_changes"
                )
        else:
            # Original behavior - load by year range
            logger.info(f"Loading all games for years {request.start_year}-{request.end_year}")
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

        # Predict geek rating using the direct model pipeline
        # The direct model expects game features + sub-model prediction columns
        geek_rating_features = df_pandas.copy()
        geek_rating_features["predicted_complexity"] = characteristics["predicted_complexity"]
        geek_rating_features["predicted_rating"] = characteristics["predicted_rating"]
        geek_rating_features["predicted_users_rated_log"] = characteristics["predicted_users_rated_log"]
        results["predicted_geek_rating"] = np.clip(
            geek_rating_pipeline.predict(geek_rating_features), 1.0, 10.0
        )

        # Drop predicted_users_rated_log from results (internal use only)
        results.drop(columns=["predicted_users_rated_log"], inplace=True, errors="ignore")

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
        results["geek_rating_experiment"] = geek_rating_registration[
            "original_experiment"
        ]["name"]

        # Add timestamp of scoring
        results["score_ts"] = datetime.now(timezone.utc).isoformat()

        # When game_ids is provided, skip uploads and return predictions
        if request.game_ids:
            output_path = None
            data_warehouse_job_id = None
            data_warehouse_table = None
            predictions_list = results.to_dict(orient="records")
        else:
            # Save predictions locally
            local_output_path = request.output_path or f"/tmp/{job_id}_predictions.parquet"
            results.to_parquet(local_output_path, index=False)

            # Upload predictions to Google Cloud Storage
            storage_client = authenticator.get_storage_client()
            bucket = storage_client.bucket(BUCKET_NAME)

            # Construct GCS path for predictions (with environment prefix)
            gcs_output_path = f"{ENVIRONMENT_PREFIX}/predictions/{job_id}_predictions.parquet"
            blob = bucket.blob(gcs_output_path)
            blob.upload_from_filename(local_output_path)

            # Use GCS path as output location
            output_path = f"gs://{BUCKET_NAME}/{gcs_output_path}"

            # Upload to data warehouse landing table (recommended)
            data_warehouse_job_id = None
            data_warehouse_table = None
            predictions_list = None
            if request.upload_to_data_warehouse:
                try:
                    logger.info("Uploading predictions to data warehouse landing table")

                    # Prepare model versions for metadata
                    model_versions = {
                        "hurdle": hurdle_registration["name"],
                        "hurdle_version": hurdle_registration["version"],
                        "hurdle_experiment": hurdle_registration["original_experiment"]["name"],
                        "complexity": complexity_registration["name"],
                        "complexity_version": complexity_registration["version"],
                        "complexity_experiment": complexity_registration["original_experiment"]["name"],
                        "rating": rating_registration["name"],
                        "rating_version": rating_registration["version"],
                        "rating_experiment": rating_registration["original_experiment"]["name"],
                        "users_rated": users_rated_registration["name"],
                        "users_rated_version": users_rated_registration["version"],
                        "users_rated_experiment": users_rated_registration["original_experiment"]["name"],
                        "geek_rating": geek_rating_registration["name"],
                        "geek_rating_version": geek_rating_registration["version"],
                        "geek_rating_experiment": geek_rating_registration["original_experiment"]["name"],
                    }

                    # Prepare predictions DataFrame for data warehouse
                    dw_predictions = results[
                        [
                            "game_id",
                            "name",
                            "year_published",
                            "predicted_hurdle_prob",
                            "predicted_complexity",
                            "predicted_rating",
                            "predicted_users_rated",
                            "predicted_geek_rating",
                        ]
                    ].copy()

                    dw_uploader = DataWarehousePredictionUploader()
                    data_warehouse_job_id = dw_uploader.upload_predictions(
                        dw_predictions, job_id, model_versions=model_versions
                    )
                    data_warehouse_table = dw_uploader.table_id

                    logger.info(
                        f"Successfully uploaded to data warehouse: {data_warehouse_table}"
                    )
                    logger.info(f"Data warehouse job ID: {data_warehouse_job_id}")
                except Exception as e:
                    import traceback

                    logger.error(f"Failed to upload to data warehouse: {str(e)}")
                    logger.error(f"Data warehouse error traceback: {traceback.format_exc()}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to upload predictions to data warehouse: {str(e)}",
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
            "geek_rating_model": {
                "name": geek_rating_registration["name"],
                "version": geek_rating_registration["version"],
                "experiment": geek_rating_registration["original_experiment"]["name"],
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
            },
            output_location=output_path,
            data_warehouse_job_id=data_warehouse_job_id,
            data_warehouse_table=data_warehouse_table,
            predictions=predictions_list,
            games_scored=len(results),
        )

        return response

    except Exception as e:
        import traceback

        logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_complexity", response_model=PredictComplexityResponse)
async def predict_complexity_endpoint(request: PredictComplexityRequest):
    """
    Predict complexity for games that need scoring (new/changed/stale).
    """
    try:
        from google.cloud import bigquery

        job_id = str(uuid.uuid4())

        # Load registered complexity model
        registered_complexity = get_registered_model("complexity")
        complexity_pipeline, complexity_registration = (
            registered_complexity.load_registered_model(
                request.complexity_model_name,
                request.complexity_model_version
            )
        )

        # Load games needing predictions
        games_df = load_games_for_complexity_scoring(
            game_ids=request.game_ids,
            max_games=request.max_games,
            complexity_model_version=complexity_registration["version"],
        )

        if len(games_df) == 0:
            logger.info("No games need complexity predictions")
            return PredictComplexityResponse(
                job_id=job_id,
                model_details={
                    "name": complexity_registration["name"],
                    "version": complexity_registration["version"],
                    "experiment": complexity_registration["original_experiment"]["name"],
                },
                games_scored=0,
                table_id="bgg-predictive-models.raw.complexity_predictions" if not request.game_ids else None,
                bq_job_id=None,
                predictions=None
            )

        logger.info(f"Scoring complexity for {len(games_df)} games")

        # Score complexity
        predictions = complexity_pipeline.predict(games_df)
        predictions = np.clip(predictions, 1, 5)

        # Build results DataFrame
        results = pd.DataFrame({
            "game_id": games_df["game_id"],
            "name": games_df["name"],
            "year_published": games_df["year_published"],
            "predicted_complexity": predictions,
            "complexity_model_name": complexity_registration["name"],
            "complexity_model_version": complexity_registration["version"],
            "complexity_experiment": complexity_registration["original_experiment"]["name"],
            "score_ts": datetime.now(timezone.utc),
            "job_id": job_id
        })

        # When game_ids is provided, skip BigQuery upload and return predictions
        if request.game_ids:
            predictions_list = results.to_dict(orient="records")
            table_id = None
            bq_job_id = None
            logger.info(f"Scored {len(results)} games (no upload for game_ids mode)")
        else:
            # Upload to BigQuery
            bq_client = bigquery.Client(project=GCP_PROJECT_ID)
            table_id = "bgg-predictive-models.raw.complexity_predictions"

            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            )

            load_job = bq_client.load_table_from_dataframe(
                results, table_id, job_config=job_config
            )
            load_job.result()  # Wait for completion

            logger.info(f"Uploaded {len(results)} predictions to {table_id}")
            predictions_list = None
            bq_job_id = load_job.job_id

        return PredictComplexityResponse(
            job_id=job_id,
            model_details={
                "name": complexity_registration["name"],
                "version": complexity_registration["version"],
                "experiment": complexity_registration["original_experiment"]["name"],
            },
            games_scored=len(results),
            table_id=table_id,
            bq_job_id=bq_job_id,
            predictions=predictions_list
        )

    except Exception as e:
        import traceback

        logger.error(f"Error in complexity prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_hurdle", response_model=PredictHurdleResponse)
async def predict_hurdle_endpoint(request: PredictHurdleRequest):
    """
    Predict hurdle probabilities for games (returns predictions without persisting).
    """
    try:
        job_id = str(uuid.uuid4())

        # Load registered hurdle model
        registered_hurdle = get_registered_model("hurdle")
        hurdle_pipeline, hurdle_registration = (
            registered_hurdle.load_registered_model(
                request.hurdle_model_name,
                request.hurdle_model_version
            )
        )

        # Load all games from games_features
        games_df = load_game_data(game_ids=request.game_ids)
        logger.info(f"Scoring hurdle for {len(games_df)} games")

        # Score hurdle probabilities
        predictions = hurdle_pipeline.predict_proba(games_df)[:, 1]

        # Build results DataFrame
        results = pd.DataFrame({
            "game_id": games_df["game_id"],
            "name": games_df["name"],
            "year_published": games_df["year_published"],
            "predicted_hurdle_prob": predictions,
        })

        logger.info(f"Scored {len(games_df)} games")

        predictions_list = results.to_dict(orient="records")

        return PredictHurdleResponse(
            job_id=job_id,
            model_details={
                "name": hurdle_registration["name"],
                "version": hurdle_registration["version"],
                "experiment": hurdle_registration["original_experiment"]["name"],
            },
            games_scored=len(games_df),
            table_id=None,
            bq_job_id=None,
            predictions=predictions_list
        )

    except Exception as e:
        import traceback
        logger.error(f"Error in hurdle prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_rating", response_model=PredictRatingResponse)
async def predict_rating_endpoint(request: PredictRatingRequest):
    """
    Predict ratings for games (returns predictions without persisting).
    """
    try:
        job_id = str(uuid.uuid4())

        # Load registered rating model
        registered_rating = get_registered_model("rating")
        rating_pipeline, rating_registration = (
            registered_rating.load_registered_model(
                request.rating_model_name,
                request.rating_model_version
            )
        )

        # Load all games from games_features
        games_df = load_game_data(game_ids=request.game_ids)
        logger.info(f"Scoring rating for {len(games_df)} games")

        # Score ratings
        predictions = rating_pipeline.predict(games_df)

        # Build results DataFrame
        results = pd.DataFrame({
            "game_id": games_df["game_id"],
            "name": games_df["name"],
            "year_published": games_df["year_published"],
            "predicted_rating": predictions,
        })

        logger.info(f"Scored {len(games_df)} games")

        predictions_list = results.to_dict(orient="records")

        return PredictRatingResponse(
            job_id=job_id,
            model_details={
                "name": rating_registration["name"],
                "version": rating_registration["version"],
                "experiment": rating_registration["original_experiment"]["name"],
            },
            games_scored=len(games_df),
            table_id=None,
            bq_job_id=None,
            predictions=predictions_list
        )

    except Exception as e:
        import traceback
        logger.error(f"Error in rating prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_users_rated", response_model=PredictUsersRatedResponse)
async def predict_users_rated_endpoint(request: PredictUsersRatedRequest):
    """
    Predict users_rated for games (returns predictions without persisting).
    """
    try:
        job_id = str(uuid.uuid4())

        # Load registered users_rated model
        registered_users_rated = get_registered_model("users_rated")
        users_rated_pipeline, users_rated_registration = (
            registered_users_rated.load_registered_model(
                request.users_rated_model_name,
                request.users_rated_model_version
            )
        )

        # Load all games from games_features
        games_df = load_game_data(game_ids=request.game_ids)
        logger.info(f"Scoring users_rated for {len(games_df)} games")

        # Score users_rated
        predictions = users_rated_pipeline.predict(games_df)

        # Apply transformation
        predictions = np.maximum(
            np.round(np.expm1(predictions) / 50) * 50,
            25,
        )

        # Build results DataFrame
        results = pd.DataFrame({
            "game_id": games_df["game_id"],
            "name": games_df["name"],
            "year_published": games_df["year_published"],
            "predicted_users_rated": predictions,
        })

        logger.info(f"Scored {len(games_df)} games")

        predictions_list = results.to_dict(orient="records")

        return PredictUsersRatedResponse(
            job_id=job_id,
            model_details={
                "name": users_rated_registration["name"],
                "version": users_rated_registration["version"],
                "experiment": users_rated_registration["original_experiment"]["name"],
            },
            games_scored=len(games_df),
            table_id=None,
            bq_job_id=None,
            predictions=predictions_list
        )

    except Exception as e:
        import traceback
        logger.error(f"Error in users_rated prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate_games", response_model=SimulateGamesResponse)
async def simulate_games_endpoint(request: SimulateGamesRequest):
    """Simulate games with full uncertainty propagation through model chain.

    Runs Bayesian posterior sampling through the dependency chain:
    complexity -> rating/users_rated (conditional on complexity) -> geek_rating (direct model).
    Also predicts hurdle probabilities. Returns per-game summary statistics with
    credible intervals, and optionally uploads to BigQuery data warehouse.
    """
    try:
        import uuid

        job_id = str(uuid.uuid4())

        # Load all model pipelines (including hurdle)
        registered_hurdle = get_registered_model("hurdle")
        registered_complexity = get_registered_model("complexity")
        registered_rating = get_registered_model("rating")
        registered_users_rated = get_registered_model("users_rated")
        registered_geek_rating = get_registered_model("geek_rating")

        hurdle_pipeline, hurdle_reg = registered_hurdle.load_registered_model(
            request.hurdle_model_name, request.hurdle_model_version
        )
        complexity_pipeline, complexity_reg = registered_complexity.load_registered_model(
            request.complexity_model_name, request.complexity_model_version
        )
        rating_pipeline, rating_reg = registered_rating.load_registered_model(
            request.rating_model_name, request.rating_model_version
        )
        users_rated_pipeline, users_rated_reg = registered_users_rated.load_registered_model(
            request.users_rated_model_name, request.users_rated_model_version
        )
        geek_rating_pipeline, geek_rating_reg = registered_geek_rating.load_registered_model(
            request.geek_rating_model_name, request.geek_rating_model_version
        )

        # Load game data
        if request.game_ids:
            logger.info(f"Loading {len(request.game_ids)} specific games for simulation")
            df_pandas = load_game_data(game_ids=request.game_ids)
        elif request.use_change_detection:
            logger.info(f"Loading changed games for years {request.start_year}-{request.end_year}")
            df_pandas = load_games_for_main_scoring(
                start_year=request.start_year,
                end_year=request.end_year,
                max_games=request.max_games,
                hurdle_model_version=hurdle_reg["version"],
                complexity_model_version=complexity_reg["version"],
                rating_model_version=rating_reg["version"],
                users_rated_model_version=users_rated_reg["version"],
            )
            if len(df_pandas) == 0:
                return SimulateGamesResponse(
                    job_id=job_id, model_details={}, n_samples=request.n_samples,
                    games_simulated=0, skipped_reason="no_changes",
                )
        else:
            logger.info(f"Loading games for years {request.start_year}-{request.end_year}")
            df_pandas = load_game_data(request.start_year, request.end_year)

        logger.info(f"Simulating {len(df_pandas)} games with {request.n_samples} samples")

        # Predict hurdle probabilities
        predicted_hurdle_prob = predict_hurdle_probabilities(hurdle_pipeline, df_pandas)

        # Pre-compute Cholesky decompositions for speed
        cholesky_cache = precompute_cholesky(
            complexity_pipeline, rating_pipeline, users_rated_pipeline,
            geek_rating_pipeline=geek_rating_pipeline,
        )

        # Run batch simulation
        sim_results = simulate_batch(
            games=df_pandas,
            complexity_pipeline=complexity_pipeline,
            rating_pipeline=rating_pipeline,
            users_rated_pipeline=users_rated_pipeline,
            n_samples=request.n_samples,
            random_state=request.random_state,
            cholesky_cache=cholesky_cache,
            geek_rating_mode="direct",
            geek_rating_pipeline=geek_rating_pipeline,
        )

        # Flatten simulation results into a DataFrame for upload/storage
        flat_rows = []
        for i, r in enumerate(sim_results):
            flat_rows.append({
                "game_id": r.game_id,
                "name": r.game_name,
                "year_published": df_pandas.iloc[i]["year_published"],
                "predicted_hurdle_prob": float(predicted_hurdle_prob.iloc[i]),
                "predicted_complexity": float(np.median(r.complexity_samples)),
                "predicted_rating": float(np.median(r.rating_samples)),
                "predicted_users_rated": float(np.median(r.users_rated_count_samples)),
                "predicted_geek_rating": float(np.median(r.geek_rating_samples)),
                "complexity_lower_90": float(r.interval(r.complexity_samples, 0.90)[0]),
                "complexity_upper_90": float(r.interval(r.complexity_samples, 0.90)[1]),
                "rating_lower_90": float(r.interval(r.rating_samples, 0.90)[0]),
                "rating_upper_90": float(r.interval(r.rating_samples, 0.90)[1]),
                "users_rated_lower_90": float(r.interval(r.users_rated_count_samples, 0.90)[0]),
                "users_rated_upper_90": float(r.interval(r.users_rated_count_samples, 0.90)[1]),
                "geek_rating_lower_90": float(r.interval(r.geek_rating_samples, 0.90)[0]),
                "geek_rating_upper_90": float(r.interval(r.geek_rating_samples, 0.90)[1]),
            })
        flat_results = pd.DataFrame(flat_rows)

        # Build model details
        model_details = {
            "hurdle": {
                "name": hurdle_reg["name"],
                "version": hurdle_reg["version"],
                "experiment": hurdle_reg["original_experiment"]["name"],
            },
            "complexity": {
                "name": complexity_reg["name"],
                "version": complexity_reg["version"],
                "experiment": complexity_reg["original_experiment"]["name"],
            },
            "rating": {
                "name": rating_reg["name"],
                "version": rating_reg["version"],
                "experiment": rating_reg["original_experiment"]["name"],
            },
            "users_rated": {
                "name": users_rated_reg["name"],
                "version": users_rated_reg["version"],
                "experiment": users_rated_reg["original_experiment"]["name"],
            },
            "geek_rating": {
                "name": geek_rating_reg["name"],
                "version": geek_rating_reg["version"],
                "experiment": geek_rating_reg["original_experiment"]["name"],
            },
        }

        # When game_ids provided, return full simulation summaries in response
        if request.game_ids:
            results_list = [r.summary() for r in sim_results]
            output_path = None
            data_warehouse_job_id = None
            data_warehouse_table = None
        else:
            results_list = None

            # Save locally + upload to GCS
            local_output_path = request.output_path or f"/tmp/{job_id}_simulations.parquet"
            flat_results.to_parquet(local_output_path, index=False)

            storage_client = authenticator.get_storage_client()
            bucket = storage_client.bucket(BUCKET_NAME)
            gcs_output_path = f"{ENVIRONMENT_PREFIX}/predictions/{job_id}_simulations.parquet"
            blob = bucket.blob(gcs_output_path)
            blob.upload_from_filename(local_output_path)
            output_path = f"gs://{BUCKET_NAME}/{gcs_output_path}"

            # Upload to BigQuery data warehouse
            data_warehouse_job_id = None
            data_warehouse_table = None
            if request.upload_to_data_warehouse:
                try:
                    logger.info("Uploading simulation results to data warehouse landing table")

                    model_versions = {
                        "hurdle": hurdle_reg["name"],
                        "hurdle_version": hurdle_reg["version"],
                        "hurdle_experiment": hurdle_reg["original_experiment"]["name"],
                        "complexity": complexity_reg["name"],
                        "complexity_version": complexity_reg["version"],
                        "complexity_experiment": complexity_reg["original_experiment"]["name"],
                        "rating": rating_reg["name"],
                        "rating_version": rating_reg["version"],
                        "rating_experiment": rating_reg["original_experiment"]["name"],
                        "users_rated": users_rated_reg["name"],
                        "users_rated_version": users_rated_reg["version"],
                        "users_rated_experiment": users_rated_reg["original_experiment"]["name"],
                        "geek_rating": geek_rating_reg["name"],
                        "geek_rating_version": geek_rating_reg["version"],
                        "geek_rating_experiment": geek_rating_reg["original_experiment"]["name"],
                    }

                    dw_uploader = DataWarehousePredictionUploader()
                    data_warehouse_job_id = dw_uploader.upload_predictions(
                        flat_results, job_id, model_versions=model_versions
                    )
                    data_warehouse_table = dw_uploader.table_id

                    logger.info(f"Successfully uploaded to data warehouse: {data_warehouse_table}")
                    logger.info(f"Data warehouse job ID: {data_warehouse_job_id}")
                except Exception as e:
                    import traceback
                    logger.error(f"Failed to upload to data warehouse: {str(e)}")
                    logger.error(f"Data warehouse error traceback: {traceback.format_exc()}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to upload simulation results to data warehouse: {str(e)}",
                    )

        return SimulateGamesResponse(
            job_id=job_id,
            model_details=model_details,
            n_samples=request.n_samples,
            games_simulated=len(sim_results),
            results=results_list,
            output_location=output_path,
            data_warehouse_job_id=data_warehouse_job_id,
            data_warehouse_table=data_warehouse_table,
        )

    except Exception as e:
        import traceback
        logger.error(f"Error in simulation: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Existing endpoints from previous implementation
@app.get("/models")
async def list_available_models():
    """List all available registered models across different types."""
    model_types = ["hurdle", "rating", "complexity", "users_rated", "geek_rating"]
    available_models = {}

    for model_type in model_types:
        try:
            registered_model = get_registered_model(model_type)
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
        registered_model = get_registered_model(model_type)
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
