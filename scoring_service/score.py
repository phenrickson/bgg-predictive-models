"""Score data using registered models."""

import numpy as np
import polars as pl
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from scoring_service.registered_model import RegisteredModel


def load_scoring_data(
    data_path: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    game_ids: Optional[List[int]] = None,
) -> pl.DataFrame:
    """Load data for scoring based on provided parameters.

    Args:
        data_path: Optional path to a CSV file for scoring
        start_year: First year of data to include
        end_year: Last year of data to include
        game_ids: Optional list of specific game IDs to load

    Returns:
        Polars DataFrame with data to be scored
    """
    from src.data.loader import BGGDataLoader
    from src.data.config import load_config

    # If data path is provided, load directly from CSV
    if data_path:
        return pl.read_csv(data_path)

    # Load configuration and data loader
    config = load_config()
    loader = BGGDataLoader(config)

    # Construct where clauses
    where_clauses = []

    # Year filtering
    if start_year is not None:
        where_clauses.append(f"year_published >= {start_year}")

    if end_year is None:
        # Default to 5 years greater than the current year
        end_year = datetime.now().year + 5

    where_clauses.append(f"year_published <= {end_year}")

    # Combine where clauses
    where_str = " AND ".join(where_clauses)

    # If game_ids provided, override where_str
    if game_ids:
        where_str = f"game_id IN ({','.join(map(str, game_ids))})"

    # Load data with filtering
    return loader.load_data(where_clause=where_str, preprocessor=None)


def predict_data(
    pipeline: Any, df: pl.DataFrame, model_type: str, registration: Dict[str, Any]
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
    """Predict data using the given pipeline and model type.

    Args:
        pipeline: Trained model pipeline
        df: Input DataFrame to predict
        model_type: Type of model being used
        registration: Model registration metadata

    Returns:
        Tuple of (predicted_values, predicted_class, threshold)
    """
    # Convert to pandas for prediction
    df_pandas = df.to_pandas()

    # Get threshold from registration metadata if available
    threshold = registration.get("metadata", {}).get("optimal_threshold")

    # Predict based on model type
    if model_type == "hurdle":
        # Use predict_proba for hurdle model
        predictions = pipeline.predict_proba(df_pandas)[:, 1]

        # Use default threshold of 0.5 if none found
        threshold = threshold if threshold is not None else 0.5
        print(f"Using classification threshold: {threshold}")

        predicted_class = predictions >= threshold
        predicted_values = predictions

    elif model_type == "complexity":
        # Predict and constrain to 1-5 range
        predictions = pipeline.predict(df_pandas)
        predicted_values = np.clip(predictions, 1, 5)
        predicted_class = None
        threshold = None

    elif model_type == "users_rated":
        # Predict and inverse transform log predictions
        predictions = pipeline.predict(df_pandas)
        predicted_values = np.maximum(np.round(np.expm1(predictions) / 50) * 50, 25)
        predicted_class = None
        threshold = None

    elif model_type == "rating":
        # Predict and constrain to 1-10 range
        predictions = pipeline.predict(df_pandas)
        predicted_values = np.clip(predictions, 1, 10)
        predicted_class = None
        threshold = None

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return predicted_values, predicted_class, threshold


def prepare_results(
    df: pl.DataFrame,
    predicted_values: np.ndarray,
    predicted_class: Optional[np.ndarray],
    model_type: str,
    threshold: Optional[float] = None,
) -> pl.DataFrame:
    """Prepare results DataFrame based on model type.

    Args:
        df: Original input DataFrame
        predicted_values: Predicted values
        predicted_class: Predicted classes (for classification models)
        model_type: Type of model being used
        threshold: Optional threshold used for classification

    Returns:
        Results DataFrame with predictions
    """
    # Select base columns
    results = df.select(["game_id", "name", "year_published"])

    # Add predictions based on model type
    if model_type == "complexity":
        results = results.with_columns(
            [
                pl.Series("predicted_complexity", predicted_values),
                pl.Series("complexity", df.select("complexity").to_pandas().squeeze()),
            ]
        )

    elif model_type == "rating":
        results = results.with_columns(
            [
                pl.Series("predicted_rating", predicted_values),
                pl.Series("rating", df.select("rating").to_pandas().squeeze()),
            ]
        )

    elif model_type == "users_rated":
        results = results.with_columns(
            [
                pl.Series("predicted_users_rated", predicted_values),
                pl.Series(
                    "users_rated", df.select("users_rated").to_pandas().squeeze()
                ),
            ]
        )

    elif model_type == "hurdle":
        results = results.with_columns(
            [
                pl.Series("predicted_prob", predicted_values),
                pl.Series("predicted_class", predicted_class),
                pl.Series("hurdle", df.select("hurdle").to_pandas().squeeze()),
                pl.Series(
                    "threshold",
                    (
                        [threshold] * len(df)
                        if threshold is not None
                        else [None] * len(df)
                    ),
                ),
            ]
        )

    return results


def score_data(
    model_type: str,
    model_name: str,
    bucket_name: str,
    data_path: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    model_version: Optional[int] = None,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """Score data using a registered model.

    Args:
        model_type: Type of model to use
        model_name: Name of registered model
        bucket_name: GCS bucket containing registered models
        data_path: Optional path to a CSV file for scoring
        start_year: First year of data to include
        end_year: Last year of data to include
        model_version: Optional specific model version
        output_path: Optional path to save predictions

    Returns:
        DataFrame containing predictions
    """
    # Load registered model
    registered_model = RegisteredModel(model_type, bucket_name)
    pipeline, registration = registered_model.load_registered_model(
        model_name, model_version
    )

    # Load data
    df = load_scoring_data(
        data_path=data_path, start_year=start_year, end_year=end_year
    )

    # Generate predictions
    predicted_values, predicted_class, threshold = predict_data(
        pipeline, df, model_type, registration
    )

    # Prepare results
    results = prepare_results(
        df, predicted_values, predicted_class, model_type, threshold
    )

    # Save results if output path provided
    if output_path:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as parquet
        if not output_path.suffix == ".parquet":
            output_path = output_path.with_suffix(".parquet")

        results.write_parquet(str(output_path))
        print(f"Predictions saved to {output_path}")

    return results
