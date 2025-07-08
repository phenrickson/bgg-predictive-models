import os
import json
import uuid
from typing import Dict, Any, Optional

import polars as pl
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.models.score import score_data
from src.models.experiments import ExperimentTracker

class ScoringRequest(BaseModel):
    model_type: str = Field(..., description="Type of model to use for scoring")
    experiment_name: Optional[str] = Field(None, description="Specific experiment name")
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
    request: ScoringRequest,
    data_file: Optional[UploadFile] = File(None)
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
        output_location = request.output_location or f"/tmp/{job_id}_predictions.parquet"
        
        # Perform scoring
        results = score_data(
            experiment_name=request.experiment_name,
            data_path=data_path,
            start_year=request.start_year,
            end_year=request.end_year,
            model_type=request.model_type,
            output_path=output_location
        )
        
        # Compute performance diagnostics
        performance_diagnostics = {
            "total_games_scored": len(results),
            "prediction_distribution": {
                "mean": float(results["predicted_prob"].mean()) if "predicted_prob" in results.columns else None,
                "median": float(results["predicted_prob"].median()) if "predicted_prob" in results.columns else None,
                "std_dev": float(results["predicted_prob"].std()) if "predicted_prob" in results.columns else None
            }
        }
        
        # Get model details
        tracker = ExperimentTracker(request.model_type)
        experiments = tracker.list_experiments()
        matching_experiments = [
            exp for exp in experiments 
            if exp['name'] == request.experiment_name
        ]
        
        model_details = matching_experiments[0] if matching_experiments else {}
        
        # Construct response
        response = ScoringResponse(
            job_id=job_id,
            model_details=model_details,
            scoring_parameters={
                "start_year": request.start_year,
                "end_year": request.end_year,
                "model_type": request.model_type
            },
            performance_diagnostics=performance_diagnostics,
            output_location=output_location
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_available_models():
    """List all available models across different types."""
    model_types = ['hurdle', 'rating', 'complexity', 'users_rated']
    available_models = {}
    
    for model_type in model_types:
        try:
            tracker = ExperimentTracker(model_type)
            available_models[model_type] = tracker.list_experiments()
        except Exception as e:
            available_models[model_type] = []
    
    return available_models

@app.get("/model/{model_type}/{experiment_name}/info")
async def get_model_info(model_type: str, experiment_name: str):
    """Get detailed information about a specific model."""
    try:
        tracker = ExperimentTracker(model_type)
        experiments = tracker.list_experiments()
        matching_experiments = [
            exp for exp in experiments 
            if exp['name'] == experiment_name
        ]
        
        if not matching_experiments:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Load the latest version of the experiment
        latest_experiment = max(
            matching_experiments, 
            key=lambda x: x['version']
        )
        
        experiment = tracker.load_experiment(
            latest_experiment['name'], 
            latest_experiment['version']
        )
        
        # Retrieve additional metadata
        try:
            metrics = experiment.get_metrics('test')
        except Exception:
            metrics = {}
        
        try:
            parameters = experiment.get_parameters()
        except Exception:
            parameters = {}
        
        try:
            model_info = experiment.get_model_info()
        except Exception:
            model_info = {}
        
        return {
            "experiment": latest_experiment,
            "metrics": metrics,
            "parameters": parameters,
            "model_info": model_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
