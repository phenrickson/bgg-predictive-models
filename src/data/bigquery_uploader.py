"""BigQuery uploader for model predictions to data warehouse."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from google.cloud import bigquery

from src.utils.config import load_config, PredictionsDestinationConfig


logger = logging.getLogger(__name__)


# Schema for the predictions landing table in data warehouse
PREDICTIONS_LANDING_SCHEMA = [
    bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("game_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("year_published", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("predicted_hurdle_prob", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("predicted_complexity", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("predicted_rating", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("predicted_users_rated", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("predicted_geek_rating", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("model_versions", "JSON", mode="NULLABLE"),
    bigquery.SchemaField("score_ts", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("source_environment", "STRING", mode="NULLABLE"),
]


class DataWarehousePredictionUploader:
    """Uploads predictions to BigQuery landing table for Dataform processing.

    The target table is managed by Terraform in the bgg-predictive-models project.
    Dataform in bgg-data-warehouse consumes this table.
    """

    def __init__(self, config: Optional[PredictionsDestinationConfig] = None):
        """Initialize uploader for data warehouse predictions.

        Args:
            config: Predictions destination configuration. If not provided,
                loads from config.yaml.
        """
        if config is None:
            app_config = load_config()
            config = app_config.get_predictions_destination()

        self.config = config
        self.client = config.get_client()
        self.table_id = config.get_table_id()
        self.environment = os.getenv("ENVIRONMENT", "dev")

        logger.info(f"Initialized DataWarehousePredictionUploader")
        logger.info(f"Target table: {self.table_id}")

    def upload_predictions(
        self,
        predictions_df: pd.DataFrame,
        job_id: str,
        model_versions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Upload predictions to the data warehouse landing table.

        Args:
            predictions_df: DataFrame containing predictions with columns:
                - game_id (required)
                - game_name (optional)
                - year_published (optional)
                - predicted_hurdle_prob (optional)
                - predicted_complexity (optional)
                - predicted_rating (optional)
                - predicted_users_rated (optional)
                - predicted_geek_rating (optional)
            job_id: Unique identifier for this prediction job
            model_versions: Optional dict with model version info, e.g.:
                {"hurdle": "hurdle-v2025", "complexity": "complexity-v2025", ...}

        Returns:
            BigQuery load job ID
        """
        logger.info(f"Uploading {len(predictions_df)} predictions to {self.table_id}")

        # Prepare DataFrame
        df = predictions_df.copy()

        # Add required columns
        df["job_id"] = job_id
        df["score_ts"] = datetime.utcnow()
        df["source_environment"] = self.environment

        # Add model versions as JSON
        if model_versions:
            df["model_versions"] = json.dumps(model_versions)
        else:
            df["model_versions"] = None

        # Ensure game_id is integer
        if "game_id" in df.columns:
            df["game_id"] = df["game_id"].astype("Int64")

        # Configure load job
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        )

        # Upload
        load_job = self.client.load_table_from_dataframe(df, self.table_id, job_config=job_config)
        load_job.result()  # Wait for completion

        logger.info(f"Successfully uploaded predictions to {self.table_id}")
        logger.info(f"Load job ID: {load_job.job_id}")

        return load_job.job_id

    def query_latest_predictions(
        self,
        game_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query latest predictions from the landing table.

        Args:
            game_ids: Optional list of game IDs to filter
            limit: Optional limit on results

        Returns:
            DataFrame with latest predictions per game
        """
        query = f"""
        SELECT *
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY score_ts DESC) as rn
            FROM `{self.table_id}`
        )
        WHERE rn = 1
        """

        if game_ids:
            game_ids_str = ",".join(str(g) for g in game_ids)
            query = query.replace("WHERE rn = 1", f"WHERE rn = 1 AND game_id IN ({game_ids_str})")

        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Querying latest predictions")
        return self.client.query(query).to_dataframe()


