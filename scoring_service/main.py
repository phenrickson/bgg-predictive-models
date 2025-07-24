import os
import json
import uuid
from typing import Dict, Any, Optional

import polars as pl
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from scoring_service.score import score_data
from scoring_service.registered_model import RegisteredModel

# Get bucket name from environment variable
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "bgg-models")


class ScoringRequest(BaseModel):
    model_type: str = Field(..., description="Type of model to use for scoring")
    model_name: str = Field(..., description="Name of registered model to use")
    model_version: Optional[int] = Field(
        None, description="Specific model version (latest if not provided)"
    )
    start_year: Optional[int] = Field(None, description="First year of data to include")
    end_year: Optional[int] = Field(None, description="Last year of data to include")
    data_source: str = Field("query", description="Source of data for scoring")
    data: Optional[Dict[str, Any]] = Field(None, description="Data for scoring")
    output_location: Optional[str] = Field(None, description="Location to store output")


class ScoringResponse(BaseModel):
    job_id: str
    model_details: Dict[str, Any]
    scoring_parameters: Dict[str, Any]
    performance_diagnostics: Dict[str, Any]
    output_location: Optional[str]


app = FastAPI(title="BGG Model Scoring Service")


@app.post("/score", response_model=ScoringResponse)
async def score_models(
    request: ScoringRequest, data_file: Optional[UploadFile] = File(None)
):
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Prepare data source
        data_path = None
        if data_file:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{job_id}_input.csv"
            with open(temp_path, "wb") as buffer:
                buffer.write(await data_file.read())
            data_path = temp_path

        # Determine output location
        output_location = (
            request.output_location or f"/tmp/{job_id}_predictions.parquet"
        )

        # Perform scoring
        results = score_data(
            model_type=request.model_type,
            model_name=request.model_name,
            bucket_name=BUCKET_NAME,
            data_path=data_path,
            start_year=request.start_year,
            end_year=request.end_year,
            model_version=request.model_version,
            output_path=output_location,
        )

        # Compute performance diagnostics
        performance_diagnostics = {
            "total_games_scored": len(results),
            "prediction_distribution": {
                "mean": (
                    float(results["predicted_prob"].mean())
                    if "predicted_prob" in results.columns
                    else None
                ),
                "median": (
                    float(results["predicted_prob"].median())
                    if "predicted_prob" in results.columns
                    else None
                ),
                "std_dev": (
                    float(results["predicted_prob"].std())
                    if "predicted_prob" in results.columns
                    else None
                ),
            },
        }

        # Get registered model details
        registered_model = RegisteredModel(request.model_type, BUCKET_NAME)
        pipeline, registration = registered_model.load_registered_model(
            request.model_name, request.model_version
        )

        # Use registration details for model info
        model_details = {
            "name": registration["name"],
            "version": registration["version"],
            "description": registration["description"],
            "registered_at": registration["registered_at"],
            "original_experiment": registration["original_experiment"],
            "model_info": registration["model_info"],
        }

        # Construct response
        response = ScoringResponse(
            job_id=job_id,
            model_details=model_details,
            scoring_parameters={
                "start_year": request.start_year,
                "end_year": request.end_year,
                "model_type": request.model_type,
            },
            performance_diagnostics=performance_diagnostics,
            output_location=output_location,
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            "validation_metrics": registration["validation_metrics"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
