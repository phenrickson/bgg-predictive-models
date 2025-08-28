"""BigQuery uploader for model predictions."""

import logging
from typing import Optional, List
from pathlib import Path
import pandas as pd
import yaml
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from .config import load_config


logger = logging.getLogger(__name__)


class BigQueryUploader:
    """Handles uploading predictions to BigQuery."""

    def __init__(self, environment: str = "dev", config_path: Optional[str] = None):
        """Initialize BigQuery uploader.

        Args:
            environment: Environment to use (dev/prod)
            config_path: Path to BigQuery config file (optional, uses config.yaml by default)
        """
        self.environment = environment

        # Load BigQuery configuration from config.yaml
        self.bq_config = load_config()
        self.client = self.bq_config.get_client()

        # Use project_id from BigQuery config (which gets it from environment variables)
        self.project_id = self.bq_config.project_id

        # Set dataset based on environment suffix
        base_dataset = self.bq_config.dataset
        if environment == "prod":
            # Remove _dev suffix for production
            self.dataset_id = base_dataset.replace("_dev", "")
        else:
            # Keep _dev suffix for dev environment
            self.dataset_id = base_dataset

        # Default location
        self.location = "US"

        # Load table configurations from bigquery.yaml for schema definitions
        self.table_config_path = (
            config_path
            or Path(__file__).parent.parent.parent / "config" / "bigquery.yaml"
        )

        with open(self.table_config_path) as f:
            self.table_config = yaml.safe_load(f)

        logger.info(f"Initialized BigQuery uploader for {environment} environment")
        logger.info(f"Project: {self.project_id}, Dataset: {self.dataset_id}")

    def _get_table_schema(self, table_name: str) -> List[bigquery.SchemaField]:
        """Get BigQuery schema for a table.

        Args:
            table_name: Name of the table

        Returns:
            List of BigQuery schema fields
        """
        if table_name not in self.table_config["tables"]:
            raise ValueError(f"Table {table_name} not found in configuration")

        table_def = self.table_config["tables"][table_name]
        if "schema" not in table_def:
            raise ValueError(f"No schema defined for table {table_name}")

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

    def _create_table_if_not_exists(self, table_name: str) -> bigquery.Table:
        """Create BigQuery table if it doesn't exist.

        Args:
            table_name: Name of the table to create

        Returns:
            BigQuery table object
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"

        try:
            table = self.client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
            return table
        except NotFound:
            logger.info(f"Creating table {table_id}")

            # Get table configuration
            table_def = self.table_config["tables"][table_name]
            schema = self._get_table_schema(table_name)

            # Create table
            table = bigquery.Table(table_id, schema=schema)
            table.description = table_def.get("description", "")

            # Set up partitioning if specified
            if "time_partitioning" in table_def:
                partition_field = table_def["time_partitioning"]
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY, field=partition_field
                )

            # Set up clustering if specified
            if "clustering_fields" in table_def:
                table.clustering_fields = table_def["clustering_fields"]

            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")
            return table

    def _prepare_dataframe_for_bigquery(
        self, df: pd.DataFrame, table_name: str
    ) -> pd.DataFrame:
        """Prepare DataFrame for BigQuery upload.

        Args:
            df: DataFrame to prepare
            table_name: Target table name

        Returns:
            Prepared DataFrame
        """
        df_prepared = df.copy()

        # Get schema to understand expected types
        schema = self._get_table_schema(table_name)
        schema_dict = {field.name: field.field_type for field in schema}

        # Convert data types to match BigQuery schema
        for column, bq_type in schema_dict.items():
            if column not in df_prepared.columns:
                continue

            if bq_type == "TIMESTAMP":
                # Ensure timestamp columns are datetime
                df_prepared[column] = pd.to_datetime(df_prepared[column])
            elif bq_type == "INTEGER":
                # Convert to nullable integer
                df_prepared[column] = df_prepared[column].astype("Int64")
            elif bq_type == "FLOAT":
                # Ensure float type
                df_prepared[column] = df_prepared[column].astype("float64")
            elif bq_type == "STRING":
                # Ensure string type
                df_prepared[column] = df_prepared[column].astype("string")

        return df_prepared

    def upload_predictions(
        self, predictions_df: pd.DataFrame, job_id: str, table_name: str = "predictions"
    ) -> str:
        """Upload predictions DataFrame to BigQuery.

        Args:
            predictions_df: DataFrame containing predictions
            job_id: Unique job identifier for this prediction run
            table_name: Target table name (default: predictions)

        Returns:
            Job ID of the BigQuery load job
        """
        logger.info(f"Uploading {len(predictions_df)} predictions to BigQuery")

        # Ensure table exists
        table = self._create_table_if_not_exists(table_name)

        # Add job_id to all rows if not present
        if "job_id" not in predictions_df.columns:
            predictions_df = predictions_df.copy()
            predictions_df["job_id"] = job_id

        # Prepare DataFrame for BigQuery
        df_prepared = self._prepare_dataframe_for_bigquery(predictions_df, table_name)

        # Configure load job
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,  # Append mode
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        )

        # Start load job
        load_job = self.client.load_table_from_dataframe(
            df_prepared, table, job_config=job_config
        )

        # Wait for job to complete
        load_job.result()

        logger.info(f"Successfully uploaded predictions to {table.table_id}")
        logger.info(f"Load job ID: {load_job.job_id}")

        return load_job.job_id

    def upload_predictions_from_parquet(
        self,
        parquet_path: str,
        job_id: Optional[str] = None,
        table_name: str = "predictions",
    ) -> str:
        """Upload predictions from Parquet file to BigQuery.

        Args:
            parquet_path: Path to Parquet file (local or GCS)
            job_id: Optional job ID (will be extracted from filename if not provided)
            table_name: Target table name (default: predictions)

        Returns:
            Job ID of the BigQuery load job
        """
        logger.info(f"Loading predictions from {parquet_path}")

        # Load DataFrame from Parquet
        df = pd.read_parquet(parquet_path)

        # Extract job_id from filename if not provided
        if job_id is None:
            filename = Path(parquet_path).stem
            if "_predictions" in filename:
                job_id = filename.replace("_predictions", "")
            else:
                raise ValueError(
                    "Could not extract job_id from filename and none provided"
                )

        return self.upload_predictions(df, job_id, table_name)

    def upload_predictions_from_gcs(
        self,
        gcs_path: str,
        job_id: Optional[str] = None,
        table_name: str = "predictions",
    ) -> str:
        """Upload predictions directly from GCS to BigQuery.

        Args:
            gcs_path: GCS path to Parquet file (gs://bucket/path)
            job_id: Optional job ID for tracking
            table_name: Target table name (default: predictions)

        Returns:
            Job ID of the BigQuery load job
        """
        logger.info(f"Loading predictions directly from GCS: {gcs_path}")

        # Ensure table exists
        table = self._create_table_if_not_exists(table_name)

        # Configure load job for Parquet
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        )

        # Start load job
        load_job = self.client.load_table_from_uri(
            gcs_path, table, job_config=job_config
        )

        # Wait for job to complete
        load_job.result()

        logger.info(f"Successfully loaded predictions from GCS to {table.table_id}")
        logger.info(f"Load job ID: {load_job.job_id}")

        return load_job.job_id

    def query_predictions(
        self,
        job_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query predictions from BigQuery.

        Args:
            job_id: Optional job ID to filter by
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            limit: Optional limit on number of rows

        Returns:
            DataFrame with query results
        """
        table_id = f"{self.project_id}.{self.dataset_id}.predictions"

        query = f"SELECT * FROM `{table_id}`"
        where_clauses = []

        if job_id:
            where_clauses.append(f"job_id = '{job_id}'")

        if start_date:
            where_clauses.append(f"DATE(score_ts) >= '{start_date}'")

        if end_date:
            where_clauses.append(f"DATE(score_ts) <= '{end_date}'")

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY score_ts DESC"

        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Executing query: {query}")

        return self.client.query(query).to_dataframe()

    def get_prediction_summary(self) -> pd.DataFrame:
        """Get summary statistics of predictions in BigQuery.

        Returns:
            DataFrame with summary statistics
        """
        table_id = f"{self.project_id}.{self.dataset_id}.predictions"

        query = f"""
        SELECT 
            job_id,
            COUNT(*) as num_predictions,
            MIN(score_ts) as earliest_prediction,
            MAX(score_ts) as latest_prediction,
            MIN(year_published) as min_year,
            MAX(year_published) as max_year,
            AVG(predicted_geek_rating) as avg_predicted_rating,
            hurdle_experiment,
            complexity_experiment,
            rating_experiment,
            users_rated_experiment
        FROM `{table_id}`
        GROUP BY 
            job_id, 
            hurdle_experiment, 
            complexity_experiment, 
            rating_experiment, 
            users_rated_experiment
        ORDER BY latest_prediction DESC
        """

        logger.info("Getting prediction summary")
        return self.client.query(query).to_dataframe()


def upload_predictions_cli():
    """Command-line interface for uploading predictions."""
    import argparse

    parser = argparse.ArgumentParser(description="Upload predictions to BigQuery")
    parser.add_argument("--parquet-path", required=True, help="Path to Parquet file")
    parser.add_argument("--job-id", help="Job ID for tracking")
    parser.add_argument(
        "--environment",
        default="dev",
        choices=["dev", "prod"],
        help="Environment to upload to",
    )
    parser.add_argument("--table-name", default="predictions", help="Target table name")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create uploader and upload
    uploader = BigQueryUploader(environment=args.environment)

    if args.parquet_path.startswith("gs://"):
        job_id = uploader.upload_predictions_from_gcs(
            args.parquet_path, args.job_id, args.table_name
        )
    else:
        job_id = uploader.upload_predictions_from_parquet(
            args.parquet_path, args.job_id, args.table_name
        )

    print(f"Upload completed. BigQuery job ID: {job_id}")


if __name__ == "__main__":
    upload_predictions_cli()
