import os
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timezone

import polars as pl
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from scoring_service.registered_model import RegisteredModel
from src.data.loader import BGGDataLoader
from src.data.config import load_config
from src.models.geek_rating import calculate_geek_rating
from src.models.score import load_model

# Get bucket name from environment variable
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "bgg-models")


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


class PredictGamesResponse(BaseModel):
    job_id: str
    model_details: Dict[str, Any]
    scoring_parameters: Dict[str, Any]
    output_location: str


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
    likely_games_mask: pd.Series,
) -> pd.DataFrame:
    """
    Predict game complexity, rating, and users rated.
    """
    results = pd.DataFrame(index=features.index)

    # Default values for all games
    results["predicted_complexity"] = 1.0
    results["predicted_rating"] = 5.5
    results["predicted_users_rated"] = 25

    # Predict for likely games
    if likely_games_mask.any():
        likely_features = features[likely_games_mask]

        # Predict complexity
        results.loc[likely_games_mask, "predicted_complexity"] = (
            complexity_model.predict(likely_features)
        )

        # Add predicted complexity to features
        likely_features_with_complexity = likely_features.copy()
        likely_features_with_complexity["predicted_complexity"] = results.loc[
            likely_games_mask, "predicted_complexity"
        ]

        # Predict rating and users rated
        results.loc[likely_games_mask, "predicted_rating"] = rating_model.predict(
            likely_features_with_complexity
        )
        results.loc[likely_games_mask, "predicted_users_rated"] = (
            users_rated_model.predict(likely_features_with_complexity)
        )

        # Ensure users is at least minimum threshold
        results.loc[likely_games_mask, "predicted_users_rated"] = np.maximum(
            np.round(
                np.expm1(results.loc[likely_games_mask, "predicted_users_rated"]) / 50
            )
            * 50,
            25,
        )

    return results


app = FastAPI(title="BGG Model Scoring Service")


@app.post("/predict_games", response_model=PredictGamesResponse)
async def predict_games_endpoint(request: PredictGamesRequest):
    """
    Predict game characteristics using registered models.
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Load models
        registered_hurdle_model = RegisteredModel("hurdle", BUCKET_NAME)
        registered_complexity_model = RegisteredModel("complexity", BUCKET_NAME)
        registered_rating_model = RegisteredModel("rating", BUCKET_NAME)
        registered_users_rated_model = RegisteredModel("users_rated", BUCKET_NAME)

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

        # Save predictions
        output_path = request.output_path or f"/tmp/{job_id}_predictions.parquet"
        results.to_parquet(output_path, index=False)

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
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Existing endpoints from previous implementation
@app.get("/models")
async def list_available_models():
    """List all available registered models across different types."""
    model_types = ["hurdle", "rating", "complexity", "users_rated"]
    available_models = {}

    for model_type in model_types:
        try:
            registered_model = RegisteredModel(model_type, BUCKET_NAME)
            available_models[model_type] = registered_model.list_registered_models()
        except Exception as e:
            available_models[model_type] = []

    return available_models


@app.get("/model/{model_type}/{model_name}/info")
async def get_model_info(
    model_type: str, model_name: str, version: Optional[int] = None
):
    """Get detailed information about a specific registered model."""
    try:
        registered_model = RegisteredModel(model_type, BUCKET_NAME)
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
