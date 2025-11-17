"""Storage layer for BGG collection data in BigQuery."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import yaml
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class CollectionStorage:
    """Handles storing and retrieving user collections in BigQuery."""

    def __init__(self, environment: str = "dev", config_path: Optional[str] = None):
        """Initialize collection storage.

        Args:
            environment: Environment to use (dev/prod)
            config_path: Path to BigQuery config file (optional)
        """
        self.environment = environment

        # Load configuration
        config = load_config()
        self.bq_config = config.get_bigquery_config()
        self.client = self.bq_config.get_client()

        # Get environment-specific settings
        self.project_id = self.bq_config.project_id
        self.dataset_id = self.bq_config.dataset
        self.location = self.bq_config.location

        # Load table configurations
        self.table_config_path = (
            config_path
            or Path(__file__).parent.parent.parent / "config" / "bigquery.yaml"
        )

        with open(self.table_config_path) as f:
            self.table_config = yaml.safe_load(f)

        logger.info(f"Initialized collection storage for {environment} environment")
        logger.info(f"Project: {self.project_id}, Dataset: {self.dataset_id}")

    def _get_table_schema(self) -> list[bigquery.SchemaField]:
        """Get BigQuery schema for collections table.

        Returns:
            List of BigQuery schema fields
        """
        if "collections" not in self.table_config["tables"]:
            raise ValueError("Collections table not found in configuration")

        table_def = self.table_config["tables"]["collections"]
        if "schema" not in table_def:
            raise ValueError("No schema defined for collections table")

        schema_fields = []
        for field_def in table_def["schema"]:
            field = bigquery.SchemaField(
                name=field_def["name"],
                field_type=field_def["type"],
                mode=field_def["mode"],
                description=field_def.get("description"),
            )
            schema_fields.append(field)

        return schema_fields

    def _create_table_if_not_exists(self) -> bigquery.Table:
        """Create collections table if it doesn't exist.

        Returns:
            BigQuery table object
        """
        table_id = f"{self.project_id}.{self.dataset_id}.collections"

        try:
            table = self.client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
            return table
        except NotFound:
            logger.info(f"Creating table {table_id}")

            # Get table configuration
            table_def = self.table_config["tables"]["collections"]

            # Create table with schema
            schema = self._get_table_schema()
            table = bigquery.Table(table_id, schema=schema)

            # Add time partitioning
            if "time_partitioning" in table_def:
                partition_field = table_def["time_partitioning"]
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY, field=partition_field
                )

            # Add clustering
            if "clustering_fields" in table_def:
                table.clustering_fields = table_def["clustering_fields"]

            # Create the table
            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")

            return table

    def save_collection(
        self,
        username: str,
        collection_df: pl.DataFrame,
        collection_version: Optional[str] = None,
    ) -> bool:
        """Save collection to BigQuery.

        Args:
            username: BGG username
            collection_df: Collection DataFrame
            collection_version: Optional version identifier (defaults to timestamp)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure table exists
            self._create_table_if_not_exists()

            # Add metadata columns
            loaded_at = datetime.now()
            if collection_version is None:
                collection_version = loaded_at.strftime("%Y%m%d_%H%M%S")

            # Add username, loaded_at, and collection_version to DataFrame
            df_with_metadata = collection_df.with_columns(
                [
                    pl.lit(username).alias("username"),
                    pl.lit(loaded_at).alias("loaded_at"),
                    pl.lit(collection_version).alias("collection_version"),
                ]
            )

            # Convert to pandas for BigQuery upload
            pandas_df = df_with_metadata.to_pandas()

            # Convert timestamp strings to datetime if needed
            if "last_modified" in pandas_df.columns:
                pandas_df["last_modified"] = pd.to_datetime(
                    pandas_df["last_modified"], errors="coerce"
                )

            # Upload to BigQuery
            table_id = f"{self.project_id}.{self.dataset_id}.collections"

            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                schema=self._get_table_schema(),
            )

            logger.info(
                f"Uploading {len(pandas_df)} items for user '{username}' to {table_id}"
            )
            job = self.client.load_table_from_dataframe(
                pandas_df, table_id, job_config=job_config
            )

            # Wait for the job to complete
            job.result()

            logger.info(
                f"Successfully uploaded collection for '{username}' "
                f"(version: {collection_version})"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving collection for '{username}': {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_latest_collection(self, username: str) -> Optional[pl.DataFrame]:
        """Get the most recent collection for a user.

        Args:
            username: BGG username

        Returns:
            Collection DataFrame or None if not found
        """
        try:
            query = f"""
            SELECT * EXCEPT (loaded_at, collection_version)
            FROM `{self.project_id}.{self.dataset_id}.collections`
            WHERE username = @username
            AND loaded_at = (
                SELECT MAX(loaded_at)
                FROM `{self.project_id}.{self.dataset_id}.collections`
                WHERE username = @username
            )
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("username", "STRING", username)
                ]
            )

            logger.info(f"Fetching latest collection for user '{username}'")
            query_job = self.client.query(query, job_config=job_config)
            pandas_df = query_job.to_dataframe()

            if len(pandas_df) == 0:
                logger.info(f"No collection found for user '{username}'")
                return None

            df = pl.from_pandas(pandas_df)
            logger.info(f"Retrieved {len(df)} items for user '{username}'")

            return df

        except Exception as e:
            logger.error(f"Error retrieving collection for '{username}': {e}")
            return None

    def get_collection_history(self, username: str) -> Optional[pl.DataFrame]:
        """Get collection history showing all versions.

        Args:
            username: BGG username

        Returns:
            DataFrame with collection versions and metadata
        """
        try:
            query = f"""
            SELECT
                username,
                collection_version,
                loaded_at,
                COUNT(*) as item_count,
                COUNT(DISTINCT game_id) as unique_games,
                SUM(CASE WHEN owned THEN 1 ELSE 0 END) as owned_count
            FROM `{self.project_id}.{self.dataset_id}.collections`
            WHERE username = @username
            GROUP BY username, collection_version, loaded_at
            ORDER BY loaded_at DESC
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("username", "STRING", username)
                ]
            )

            logger.info(f"Fetching collection history for user '{username}'")
            query_job = self.client.query(query, job_config=job_config)
            pandas_df = query_job.to_dataframe()

            if len(pandas_df) == 0:
                logger.info(f"No collection history found for user '{username}'")
                return None

            df = pl.from_pandas(pandas_df)
            logger.info(f"Retrieved {len(df)} collection versions for '{username}'")

            return df

        except Exception as e:
            logger.error(f"Error retrieving collection history for '{username}': {e}")
            return None

    def delete_collection(self, username: str, collection_version: str) -> bool:
        """Delete a specific collection version.

        Args:
            username: BGG username
            collection_version: Collection version to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            query = f"""
            DELETE FROM `{self.project_id}.{self.dataset_id}.collections`
            WHERE username = @username
            AND collection_version = @version
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("username", "STRING", username),
                    bigquery.ScalarQueryParameter("version", "STRING", collection_version),
                ]
            )

            logger.info(
                f"Deleting collection version '{collection_version}' for user '{username}'"
            )
            query_job = self.client.query(query, job_config=job_config)
            query_job.result()

            logger.info(
                f"Successfully deleted collection version '{collection_version}' for '{username}'"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error deleting collection version '{collection_version}' for '{username}': {e}"
            )
            return False

    def get_owned_game_ids(self, username: str) -> Optional[list[int]]:
        """Get list of game IDs owned by user from latest collection.

        Args:
            username: BGG username

        Returns:
            List of game IDs or None if not found
        """
        df = self.get_latest_collection(username)
        if df is None:
            return None

        # Filter to owned boardgames only
        owned_games = df.filter(
            (pl.col("owned") == True) & (pl.col("subtype") == "boardgame")
        )

        return owned_games.select("game_id").to_series().to_list()
