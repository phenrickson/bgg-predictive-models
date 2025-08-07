#!/usr/bin/env python3
import os
import json
import sys
from typing import Dict, Any

# Import local modules
from scoring_service.registered_model import RegisteredModel
from src.models.training import predict_games


def get_latest_models(bucket_name: str = "bgg-models") -> Dict[str, str]:
    """
    Simulate the model retrieval process locally.

    In the GitHub Action, this uses gcloud run services invoke.
    Locally, we'll use the RegisteredModel class to list and select models.
    """
    model_types = ["hurdle", "complexity", "rating", "users_rated"]
    latest_models = {}

    for model_type in model_types:
        try:
            registered_model = RegisteredModel(model_type, bucket_name)
            models = registered_model.list_registered_models()

            # Select the latest model
            if models:
                latest_model = max(models, key=lambda x: x["version"])
                latest_models[f"{model_type}_model"] = latest_model["name"]
            else:
                print(f"No models found for type: {model_type}")
                latest_models[f"{model_type}_model"] = None

        except Exception as e:
            print(f"Error retrieving {model_type} models: {e}")
            latest_models[f"{model_type}_model"] = None

    return latest_models


def local_predict_games(
    hurdle_model_name: str,
    complexity_model_name: str,
    rating_model_name: str,
    users_rated_model_name: str,
    start_year: int = 2024,
    end_year: int = 2029,
    output_path: str = "data/predictions/local_game_predictions.parquet",
):
    """
    Local version of the prediction workflow.

    Simulates the Cloud Run service prediction process.
    """
    # Retrieve models
    models = get_latest_models()

    # Validate model names match retrieved models
    for model_type, model_name in [
        ("hurdle_model", hurdle_model_name),
        ("complexity_model", complexity_model_name),
        ("rating_model", rating_model_name),
        ("users_rated_model", users_rated_model_name),
    ]:
        if models.get(model_type) != model_name:
            print(
                f"Warning: {model_type} name mismatch. Retrieved: {models.get(model_type)}, Provided: {model_name}"
            )

    # Call prediction function
    predict_games(
        hurdle_model_name=hurdle_model_name,
        complexity_model_name=complexity_model_name,
        rating_model_name=rating_model_name,
        users_rated_model_name=users_rated_model_name,
        start_year=start_year,
        end_year=end_year,
        output_path=output_path,
    )

    print(f"Predictions saved to {output_path}")


def main():
    # Default parameters matching GitHub Action workflow
    default_params = {
        "hurdle_model_name": None,
        "complexity_model_name": None,
        "rating_model_name": None,
        "users_rated_model_name": None,
        "start_year": 2024,
        "end_year": 2029,
        "output_path": "data/predictions/local_game_predictions.parquet",
    }

    # Override with command-line arguments
    for i, arg in enumerate(sys.argv[1:], 1):
        if "=" in arg:
            key, value = arg.split("=")
            default_params[key] = value

    local_predict_games(**default_params)


if __name__ == "__main__":
    main()
