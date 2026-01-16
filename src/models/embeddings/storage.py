"""BigQuery storage for game embeddings."""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

import pandas as pd
import polars as pl
from google.cloud import bigquery

from src.utils.config import Config, load_config

logger = logging.getLogger(__name__)


class BigQueryEmbeddingStorage:
    """Handles embedding storage in BigQuery for Vector Search."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize BigQuery embedding storage.

        Args:
            config: Configuration object. If None, loads from config.yaml.
        """
        self.config = config or load_config()
        self.project_id = self.config.ml_project_id

        # Get dataset and table from embeddings upload config (raw table)
        if self.config.embeddings:
            self.dataset = self.config.embeddings.upload.dataset
            self.table = self.config.embeddings.upload.table
        else:
            self.dataset = "raw"
            self.table = "game_embeddings"

        self.table_id = f"{self.project_id}.{self.dataset}.{self.table}"
        self.client = bigquery.Client(project=self.project_id)

    def table_exists(self) -> bool:
        """Check if the embeddings table exists."""
        try:
            self.client.get_table(self.table_id)
            return True
        except Exception:
            return False

    def upload_embeddings(
        self,
        embeddings_df: pl.DataFrame,
        model_name: str,
        model_version: int,
        algorithm: str,
        embedding_dim: int,
    ) -> str:
        """Upload embeddings to BigQuery.

        Args:
            embeddings_df: DataFrame with columns: game_id, name, embedding (list).
            model_name: Name of the embedding model.
            model_version: Version number of the model.
            algorithm: Algorithm used (pca, svd, umap, autoencoder).
            embedding_dim: Dimension of embeddings.

        Returns:
            Job ID for the upload.
        """
        job_id = str(uuid.uuid4())
        created_ts = datetime.utcnow()

        # Convert to pandas for BigQuery upload
        df = embeddings_df.to_pandas()

        # Add metadata columns
        df["embedding_model"] = model_name
        df["embedding_version"] = model_version
        df["embedding_dim"] = embedding_dim
        df["algorithm"] = algorithm
        df["created_ts"] = created_ts
        df["job_id"] = job_id

        # Ensure embedding column is a list (BigQuery ARRAY)
        if "embedding" in df.columns:
            df["embedding"] = df["embedding"].apply(
                lambda x: list(x) if not isinstance(x, list) else x
            )

        # Upload to BigQuery
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )

        try:
            load_job = self.client.load_table_from_dataframe(
                df, self.table_id, job_config=job_config
            )
            load_job.result()  # Wait for completion

            logger.info(
                f"Uploaded {len(df)} embeddings to {self.table_id}, job_id={job_id}"
            )
            return job_id

        except Exception as e:
            logger.error(f"Failed to upload embeddings: {e}")
            raise

    def get_latest_version(self, model_name: str) -> Optional[int]:
        """Get the latest version number for a model.

        Args:
            model_name: Name of the embedding model.

        Returns:
            Latest version number or None if no versions exist.
        """
        query = f"""
        SELECT MAX(embedding_version) as latest_version
        FROM `{self.table_id}`
        WHERE embedding_model = @model_name
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("model_name", "STRING", model_name)
            ]
        )

        try:
            result = self.client.query(query, job_config=job_config).result()
            row = list(result)[0]
            return row.latest_version
        except Exception:
            return None

    def get_embeddings(
        self,
        model_name: Optional[str] = None,
        model_version: Optional[int] = None,
        game_ids: Optional[List[int]] = None,
    ) -> pl.DataFrame:
        """Retrieve embeddings from BigQuery.

        Args:
            model_name: Filter by model name.
            model_version: Filter by version. If None, gets latest.
            game_ids: Filter by specific game IDs.

        Returns:
            DataFrame with embeddings.
        """
        conditions = ["TRUE"]
        params = []

        if model_name:
            conditions.append("embedding_model = @model_name")
            params.append(
                bigquery.ScalarQueryParameter("model_name", "STRING", model_name)
            )

        if model_version:
            conditions.append("embedding_version = @model_version")
            params.append(
                bigquery.ScalarQueryParameter("model_version", "INT64", model_version)
            )

        if game_ids:
            conditions.append("game_id IN UNNEST(@game_ids)")
            params.append(
                bigquery.ArrayQueryParameter("game_ids", "INT64", game_ids)
            )

        where_clause = " AND ".join(conditions)

        # Get latest per game_id if no version specified
        if model_version is None:
            query = f"""
            SELECT * FROM `{self.table_id}`
            WHERE {where_clause}
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY game_id
                ORDER BY embedding_version DESC, created_ts DESC
            ) = 1
            """
        else:
            query = f"""
            SELECT * FROM `{self.table_id}`
            WHERE {where_clause}
            """

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        result = self.client.query(query, job_config=job_config).to_dataframe()

        return pl.from_pandas(result)

    def delete_version(self, model_name: str, model_version: int) -> int:
        """Delete a specific model version.

        Args:
            model_name: Model name.
            model_version: Version to delete.

        Returns:
            Number of rows deleted.
        """
        query = f"""
        DELETE FROM `{self.table_id}`
        WHERE embedding_model = @model_name
          AND embedding_version = @model_version
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("model_name", "STRING", model_name),
                bigquery.ScalarQueryParameter("model_version", "INT64", model_version),
            ]
        )

        result = self.client.query(query, job_config=job_config).result()
        deleted = result.num_dml_affected_rows

        logger.info(
            f"Deleted {deleted} rows for {model_name} version {model_version}"
        )
        return deleted
