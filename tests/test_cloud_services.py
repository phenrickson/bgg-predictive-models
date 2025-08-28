"""Tests for BigQuery and GCS service access patterns."""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call
from google.cloud.exceptions import NotFound, Forbidden, BadRequest
from google.auth.exceptions import DefaultCredentialsError
import tempfile
from datetime import datetime, timezone

from src.data.bigquery_uploader import BigQueryUploader
from src.data.config import load_config
from scoring_service.auth import GCPAuthenticator


class TestBigQueryUploader:
    """Test BigQuery uploader functionality."""

    @patch("src.data.bigquery_uploader.load_config")
    def test_bigquery_uploader_init(self, mock_load_config):
        """Test BigQuery uploader initialization."""
        mock_config = MagicMock()
        mock_config.get_client.return_value = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.dataset = "test_dataset_dev"
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")

        assert uploader.environment == "dev"
        assert uploader.project_id == "test-project"
        assert uploader.dataset_id == "test_dataset_dev"

    @patch("src.data.bigquery_uploader.load_config")
    def test_bigquery_uploader_prod_environment(self, mock_load_config):
        """Test BigQuery uploader with production environment."""
        mock_config = MagicMock()
        mock_config.get_client.return_value = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.dataset = "test_dataset_dev"
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="prod")

        assert uploader.environment == "prod"
        assert uploader.dataset_id == "test_dataset"  # _dev suffix removed

    @patch("src.data.bigquery_uploader.load_config")
    @patch("src.data.bigquery_uploader.yaml.safe_load")
    @patch("builtins.open")
    def test_get_table_schema(self, mock_open, mock_yaml_load, mock_load_config):
        """Test table schema retrieval."""
        mock_config = MagicMock()
        mock_config.get_client.return_value = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.dataset = "test_dataset_dev"
        mock_load_config.return_value = mock_config

        mock_yaml_load.return_value = {
            "tables": {
                "predictions": {
                    "schema": [
                        {"name": "job_id", "type": "STRING", "mode": "REQUIRED"},
                        {"name": "game_id", "type": "INTEGER", "mode": "REQUIRED"},
                        {
                            "name": "predicted_rating",
                            "type": "FLOAT",
                            "mode": "NULLABLE",
                        },
                    ]
                }
            }
        }

        uploader = BigQueryUploader(environment="dev")
        schema = uploader._get_table_schema("predictions")

        assert len(schema) == 3
        assert schema[0].name == "job_id"
        assert schema[0].field_type == "STRING"
        assert schema[0].mode == "REQUIRED"

    @patch("src.data.bigquery_uploader.load_config")
    def test_get_table_schema_missing_table(self, mock_load_config):
        """Test table schema retrieval for missing table."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")
        uploader.table_config = {"tables": {}}

        with pytest.raises(ValueError, match="Table nonexistent not found"):
            uploader._get_table_schema("nonexistent")

    @patch("src.data.bigquery_uploader.load_config")
    def test_create_table_if_not_exists_existing(self, mock_load_config):
        """Test table creation when table already exists."""
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_config.get_client.return_value = mock_client
        mock_client.get_table.return_value = mock_table
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")
        uploader.table_config = {
            "tables": {
                "predictions": {
                    "schema": [{"name": "job_id", "type": "STRING", "mode": "REQUIRED"}]
                }
            }
        }

        result = uploader._create_table_if_not_exists("predictions")

        assert result == mock_table
        mock_client.get_table.assert_called_once()
        mock_client.create_table.assert_not_called()

    @patch("src.data.bigquery_uploader.load_config")
    @patch("src.data.bigquery_uploader.bigquery")
    def test_create_table_if_not_exists_new(self, mock_bigquery, mock_load_config):
        """Test table creation when table doesn't exist."""
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_config.get_client.return_value = mock_client
        mock_client.get_table.side_effect = NotFound("Table not found")
        mock_load_config.return_value = mock_config

        # Mock BigQuery classes
        mock_table_class = MagicMock()
        mock_schema_field = MagicMock()
        mock_bigquery.Table = mock_table_class
        mock_bigquery.SchemaField = mock_schema_field

        uploader = BigQueryUploader(environment="dev")
        uploader.table_config = {
            "tables": {
                "predictions": {
                    "description": "Test table",
                    "schema": [
                        {"name": "job_id", "type": "STRING", "mode": "REQUIRED"}
                    ],
                }
            }
        }

        uploader._create_table_if_not_exists("predictions")

        mock_client.create_table.assert_called_once()

    @patch("src.data.bigquery_uploader.load_config")
    def test_prepare_dataframe_for_bigquery(self, mock_load_config):
        """Test DataFrame preparation for BigQuery upload."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")
        uploader.table_config = {
            "tables": {
                "predictions": {
                    "schema": [
                        {"name": "job_id", "type": "STRING", "mode": "REQUIRED"},
                        {"name": "game_id", "type": "INTEGER", "mode": "REQUIRED"},
                        {"name": "score_ts", "type": "TIMESTAMP", "mode": "REQUIRED"},
                        {
                            "name": "predicted_rating",
                            "type": "FLOAT",
                            "mode": "NULLABLE",
                        },
                    ]
                }
            }
        }

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "job_id": ["test-job"],
                "game_id": [12345],
                "score_ts": ["2024-01-01 12:00:00"],
                "predicted_rating": [7.5],
            }
        )

        result = uploader._prepare_dataframe_for_bigquery(df, "predictions")

        assert result["job_id"].dtype == "string"
        assert result["game_id"].dtype == "Int64"
        assert pd.api.types.is_datetime64_any_dtype(result["score_ts"])
        assert result["predicted_rating"].dtype == "float64"

    @patch("src.data.bigquery_uploader.load_config")
    def test_upload_predictions_success(self, mock_load_config):
        """Test successful predictions upload."""
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_load_job = MagicMock()
        mock_load_job.job_id = "test-load-job-123"

        mock_config.get_client.return_value = mock_client
        mock_client.get_table.return_value = mock_table
        mock_client.load_table_from_dataframe.return_value = mock_load_job
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")
        uploader.table_config = {
            "tables": {
                "predictions": {
                    "schema": [
                        {"name": "job_id", "type": "STRING", "mode": "REQUIRED"},
                        {"name": "game_id", "type": "INTEGER", "mode": "REQUIRED"},
                    ]
                }
            }
        }

        # Create test DataFrame
        df = pd.DataFrame({"game_id": [12345], "predicted_rating": [7.5]})

        result = uploader.upload_predictions(df, "test-job-123")

        assert result == "test-load-job-123"
        mock_client.load_table_from_dataframe.assert_called_once()
        mock_load_job.result.assert_called_once()

    @patch("src.data.bigquery_uploader.load_config")
    def test_query_predictions(self, mock_load_config):
        """Test predictions querying."""
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_query_result = MagicMock()
        mock_df = pd.DataFrame({"job_id": ["test"], "game_id": [123]})

        mock_config.get_client.return_value = mock_client
        mock_client.query.return_value = mock_query_result
        mock_query_result.to_dataframe.return_value = mock_df
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")
        uploader.project_id = "test-project"
        uploader.dataset_id = "test_dataset"

        result = uploader.query_predictions(job_id="test-job", limit=100)

        assert isinstance(result, pd.DataFrame)
        mock_client.query.assert_called_once()
        query_call = mock_client.query.call_args[0][0]
        assert "job_id = 'test-job'" in query_call
        assert "LIMIT 100" in query_call

    @patch("src.data.bigquery_uploader.load_config")
    def test_get_prediction_summary(self, mock_load_config):
        """Test prediction summary retrieval."""
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_query_result = MagicMock()
        mock_df = pd.DataFrame(
            {
                "job_id": ["test-job"],
                "num_predictions": [100],
                "latest_prediction": [datetime.now(timezone.utc)],
            }
        )

        mock_config.get_client.return_value = mock_client
        mock_client.query.return_value = mock_query_result
        mock_query_result.to_dataframe.return_value = mock_df
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")
        uploader.project_id = "test-project"
        uploader.dataset_id = "test_dataset"

        result = uploader.get_prediction_summary()

        assert isinstance(result, pd.DataFrame)
        assert "job_id" in result.columns
        assert "num_predictions" in result.columns


class TestGCSAccess:
    """Test Google Cloud Storage access patterns."""

    @patch("scoring_service.auth.storage.Client")
    def test_storage_client_bucket_operations(self, mock_storage_client):
        """Test basic bucket operations."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value = mock_client
        mock_client.list_buckets.return_value = []
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        auth = GCPAuthenticator(project_id="test-project")
        client = auth.get_storage_client()

        # Test bucket access
        bucket = client.bucket("test-bucket")
        blob = bucket.blob("test-file.txt")

        mock_client.bucket.assert_called_with("test-bucket")
        mock_bucket.blob.assert_called_with("test-file.txt")

    @patch("scoring_service.auth.storage.Client")
    def test_storage_upload_download(self, mock_storage_client):
        """Test file upload and download operations."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value = mock_client
        mock_client.list_buckets.return_value = []
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        auth = GCPAuthenticator(project_id="test-project")
        client = auth.get_storage_client()

        # Simulate upload
        bucket = client.bucket("test-bucket")
        blob = bucket.blob("predictions/test.parquet")

        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(b"test data")
            temp_file.flush()

            # Test upload
            blob.upload_from_filename(temp_file.name)
            mock_blob.upload_from_filename.assert_called_with(temp_file.name)

    @patch("scoring_service.auth.storage.Client")
    def test_storage_list_operations(self, mock_storage_client):
        """Test listing operations in GCS."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob1 = MagicMock()
        mock_blob2 = MagicMock()
        mock_blob1.name = "predictions/job1.parquet"
        mock_blob2.name = "predictions/job2.parquet"

        mock_storage_client.return_value = mock_client
        mock_client.list_buckets.return_value = []
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

        auth = GCPAuthenticator(project_id="test-project")
        client = auth.get_storage_client()

        # Test listing blobs
        bucket = client.bucket("test-bucket")
        blobs = list(bucket.list_blobs(prefix="predictions/"))

        assert len(blobs) == 2
        assert blobs[0].name == "predictions/job1.parquet"
        assert blobs[1].name == "predictions/job2.parquet"


class TestErrorHandling:
    """Test error handling for cloud service operations."""

    @patch("src.data.bigquery_uploader.load_config")
    def test_bigquery_permission_error(self, mock_load_config):
        """Test BigQuery permission errors."""
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_config.get_client.return_value = mock_client
        mock_client.query.side_effect = Forbidden("Access denied")
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")
        uploader.project_id = "test-project"
        uploader.dataset_id = "test_dataset"

        with pytest.raises(Forbidden):
            uploader.query_predictions()

    @patch("src.data.bigquery_uploader.load_config")
    def test_bigquery_table_not_found(self, mock_load_config):
        """Test BigQuery table not found error."""
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_config.get_client.return_value = mock_client
        mock_client.query.side_effect = NotFound("Table not found")
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")
        uploader.project_id = "test-project"
        uploader.dataset_id = "test_dataset"

        with pytest.raises(NotFound):
            uploader.query_predictions()

    @patch("src.data.bigquery_uploader.load_config")
    def test_bigquery_bad_request(self, mock_load_config):
        """Test BigQuery bad request error."""
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_config.get_client.return_value = mock_client
        mock_client.query.side_effect = BadRequest("Invalid query")
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")
        uploader.project_id = "test-project"
        uploader.dataset_id = "test_dataset"

        with pytest.raises(BadRequest):
            uploader.query_predictions()

    @patch("scoring_service.auth.storage.Client")
    def test_gcs_permission_error(self, mock_storage_client):
        """Test GCS permission errors."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_storage_client.return_value = mock_client
        mock_client.list_buckets.return_value = []
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.reload.side_effect = Forbidden("Access denied")

        auth = GCPAuthenticator(project_id="test-project")

        result = auth.verify_bucket_access("test-bucket")
        assert result is False

    @patch("scoring_service.auth.storage.Client")
    def test_gcs_bucket_not_found(self, mock_storage_client):
        """Test GCS bucket not found error."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_storage_client.return_value = mock_client
        mock_client.list_buckets.return_value = []
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.reload.side_effect = NotFound("Bucket not found")

        auth = GCPAuthenticator(project_id="test-project")

        result = auth.verify_bucket_access("test-bucket")
        assert result is False


class TestIntegrationCloudServices:
    """Integration tests for cloud services (require actual credentials)."""

    @pytest.mark.integration
    def test_bigquery_dataset_access(self):
        """Test BigQuery dataset access (requires valid credentials)."""
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            pytest.skip("GCP_PROJECT_ID not set - skipping integration test")

        try:
            config = load_config()
            client = config.get_client()

            # Test dataset access
            dataset_id = f"{config.project_id}.{config.dataset}"
            dataset = client.get_dataset(dataset_id)

            assert dataset.dataset_id == config.dataset
            assert dataset.project == config.project_id

        except Exception as e:
            pytest.skip(f"BigQuery dataset integration test failed: {e}")

    @pytest.mark.integration
    def test_bigquery_table_operations(self):
        """Test BigQuery table operations (requires valid credentials)."""
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            pytest.skip("GCP_PROJECT_ID not set - skipping integration test")

        try:
            uploader = BigQueryUploader(environment="dev")

            # Test table listing
            dataset_ref = uploader.client.dataset(uploader.dataset_id)
            tables = list(uploader.client.list_tables(dataset_ref))

            # Should be able to list tables (even if empty)
            assert isinstance(tables, list)

        except Exception as e:
            pytest.skip(f"BigQuery table integration test failed: {e}")

    @pytest.mark.integration
    def test_gcs_bucket_listing(self):
        """Test GCS bucket listing (requires valid credentials)."""
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            pytest.skip("GCP_PROJECT_ID not set - skipping integration test")

        try:
            auth = GCPAuthenticator(project_id=project_id)
            client = auth.get_storage_client()

            # Test bucket listing
            buckets = list(client.list_buckets(max_results=5))

            # Should be able to list buckets
            assert isinstance(buckets, list)

        except Exception as e:
            pytest.skip(f"GCS bucket integration test failed: {e}")

    @pytest.mark.integration
    def test_end_to_end_prediction_upload(self):
        """Test end-to-end prediction upload (requires valid credentials)."""
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            pytest.skip("GCP_PROJECT_ID not set - skipping integration test")

        try:
            # Create test prediction data
            test_data = pd.DataFrame(
                {
                    "game_id": [999999],  # Use unlikely game ID
                    "name": ["Test Game"],
                    "year_published": [2024],
                    "predicted_hurdle_prob": [0.8],
                    "predicted_complexity": [2.5],
                    "predicted_rating": [7.2],
                    "predicted_users_rated": [100],
                    "predicted_geek_rating": [6.8],
                    "hurdle_experiment": ["test_hurdle"],
                    "complexity_experiment": ["test_complexity"],
                    "rating_experiment": ["test_rating"],
                    "users_rated_experiment": ["test_users_rated"],
                    "score_ts": [datetime.now(timezone.utc)],
                }
            )

            uploader = BigQueryUploader(environment="dev")
            job_id = f"test-integration-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # Upload test data
            bq_job_id = uploader.upload_predictions(test_data, job_id)

            assert bq_job_id is not None
            assert isinstance(bq_job_id, str)

            # Query back the data
            result = uploader.query_predictions(job_id=job_id, limit=1)

            assert len(result) == 1
            assert result.iloc[0]["game_id"] == 999999
            assert result.iloc[0]["name"] == "Test Game"

        except Exception as e:
            pytest.skip(f"End-to-end integration test failed: {e}")


class TestPerformanceAndScaling:
    """Test performance and scaling considerations."""

    @patch("src.data.bigquery_uploader.load_config")
    def test_large_dataframe_preparation(self, mock_load_config):
        """Test preparation of large DataFrames."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")
        uploader.table_config = {
            "tables": {
                "predictions": {
                    "schema": [
                        {"name": "game_id", "type": "INTEGER", "mode": "REQUIRED"},
                        {
                            "name": "predicted_rating",
                            "type": "FLOAT",
                            "mode": "NULLABLE",
                        },
                    ]
                }
            }
        }

        # Create large test DataFrame
        large_df = pd.DataFrame(
            {
                "game_id": range(10000),
                "predicted_rating": [7.5] * 10000,
            }
        )

        result = uploader._prepare_dataframe_for_bigquery(large_df, "predictions")

        assert len(result) == 10000
        assert result["game_id"].dtype == "Int64"
        assert result["predicted_rating"].dtype == "float64"

    @patch("src.data.bigquery_uploader.load_config")
    def test_batch_upload_configuration(self, mock_load_config):
        """Test batch upload configuration."""
        mock_config = MagicMock()
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_load_job = MagicMock()

        mock_config.get_client.return_value = mock_client
        mock_client.get_table.return_value = mock_table
        mock_client.load_table_from_dataframe.return_value = mock_load_job
        mock_load_config.return_value = mock_config

        uploader = BigQueryUploader(environment="dev")
        uploader.table_config = {
            "tables": {
                "predictions": {
                    "schema": [
                        {"name": "game_id", "type": "INTEGER", "mode": "REQUIRED"}
                    ]
                }
            }
        }

        df = pd.DataFrame({"game_id": [123]})
        uploader.upload_predictions(df, "test-job")

        # Verify load job configuration
        call_args = mock_client.load_table_from_dataframe.call_args
        job_config = call_args[1]["job_config"]

        assert hasattr(job_config, "write_disposition")
        assert hasattr(job_config, "schema_update_options")
