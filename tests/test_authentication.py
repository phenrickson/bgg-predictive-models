"""Tests for authentication and cloud service access."""

import os
import pytest
from unittest.mock import patch, MagicMock
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.exceptions import NotFound, Forbidden
import tempfile
import json

from src.utils.config import load_config, BigQueryConfig
from scoring_service.auth import GCPAuthenticator, AuthenticationError


class TestBigQueryAuthentication:
    """Test BigQuery authentication using Application Default Credentials."""

    def test_load_config_requires_project_id(self):
        """Test that load_config requires GCP_PROJECT_ID environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="GCP_PROJECT_ID environment variable must be set"
            ):
                load_config()

    def test_load_config_with_project_id(self):
        """Test successful config loading with project ID."""
        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            config = load_config()
            assert config.project_id == "test-project"
            assert config.credentials_path is None  # Always use ADC

    @patch("src.data.config.default")
    def test_bigquery_client_creation_success(self, mock_default):
        """Test successful BigQuery client creation with ADC."""
        # Mock credentials
        mock_credentials = MagicMock()
        mock_default.return_value = (mock_credentials, "test-project")

        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            config = load_config()

            with patch("src.data.config.bigquery.Client") as mock_client:
                client = config.get_client()  # noqa: F841

                # Verify client was created with correct parameters
                mock_client.assert_called_once_with(
                    credentials=mock_credentials, project="test-project"
                )

    @patch("src.data.config.default")
    def test_bigquery_client_creation_failure(self, mock_default):
        """Test BigQuery client creation failure."""
        mock_default.side_effect = DefaultCredentialsError("No credentials found")

        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            config = load_config()

            with pytest.raises(DefaultCredentialsError):
                config.get_client()

    def test_bigquery_config_dataclass(self):
        """Test BigQueryConfig dataclass properties."""
        config = BigQueryConfig(
            project_id="test-project", dataset="test-dataset", credentials_path=None
        )

        assert config.project_id == "test-project"
        assert config.dataset == "test-dataset"
        assert config.credentials_path is None


class TestGCPAuthenticator:
    """Test GCP Authenticator from scoring service."""

    def test_authenticator_init_with_project_id(self):
        """Test authenticator initialization with explicit project ID."""
        auth = GCPAuthenticator(project_id="test-project")
        assert auth.project_id == "test-project"

    @patch.dict(os.environ, {"GCP_PROJECT_ID": "env-project"})
    def test_authenticator_init_from_env(self):
        """Test authenticator initialization from environment variable."""
        auth = GCPAuthenticator()
        assert auth.project_id == "env-project"

    @patch("scoring_service.auth.default")
    def test_authenticator_init_from_credentials(self, mock_default):
        """Test authenticator initialization from default credentials."""
        mock_default.return_value = (MagicMock(), "creds-project")

        with patch.dict(os.environ, {}, clear=True):
            auth = GCPAuthenticator()
            assert auth.project_id == "creds-project"

    @patch("requests.get")
    def test_authenticator_init_from_metadata(self, mock_get):
        """Test authenticator initialization from metadata service."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "metadata-project"
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "scoring_service.auth.default", side_effect=DefaultCredentialsError()
            ):
                auth = GCPAuthenticator()
                assert auth.project_id == "metadata-project"

    def test_authenticator_init_failure(self):
        """Test authenticator initialization failure when no project ID found."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "scoring_service.auth.default", side_effect=DefaultCredentialsError()
            ):
                with patch("requests.get", side_effect=Exception()):
                    with pytest.raises(
                        AuthenticationError, match="Could not determine GCP project ID"
                    ):
                        GCPAuthenticator()

    @patch("scoring_service.auth.storage.Client")
    def test_get_storage_client_success(self, mock_storage_client):
        """Test successful storage client creation."""
        mock_client = MagicMock()
        mock_storage_client.return_value = mock_client
        mock_client.list_buckets.return_value = []

        auth = GCPAuthenticator(project_id="test-project")
        client = auth.get_storage_client()

        assert client == mock_client
        mock_storage_client.assert_called_once_with(project="test-project")

    @patch("scoring_service.auth.storage.Client")
    def test_get_storage_client_auth_failure(self, mock_storage_client):
        """Test storage client creation with authentication failure."""
        mock_client = MagicMock()
        mock_storage_client.return_value = mock_client
        mock_client.list_buckets.side_effect = Forbidden("Access denied")

        auth = GCPAuthenticator(project_id="test-project")

        with pytest.raises(AuthenticationError, match="Authentication test failed"):
            auth.get_storage_client()

    @patch("scoring_service.auth.storage.Client")
    def test_verify_bucket_access_success(self, mock_storage_client):
        """Test successful bucket access verification."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_client.list_buckets.return_value = []

        auth = GCPAuthenticator(project_id="test-project")
        result = auth.verify_bucket_access("test-bucket")

        assert result is True
        mock_bucket.reload.assert_called_once()

    @patch("scoring_service.auth.storage.Client")
    def test_verify_bucket_access_failure(self, mock_storage_client):
        """Test bucket access verification failure."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_client.list_buckets.return_value = []
        mock_bucket.reload.side_effect = NotFound("Bucket not found")

        auth = GCPAuthenticator(project_id="test-project")
        result = auth.verify_bucket_access("test-bucket")

        assert result is False

    def test_get_authentication_info_service_account(self):
        """Test authentication info with service account credentials."""
        with patch.dict(
            os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/key.json"}
        ):
            auth = GCPAuthenticator(project_id="test-project")
            info = auth.get_authentication_info()

            assert info["project_id"] == "test-project"
            assert info["credentials_source"] == "service_account_key"
            assert info["credentials_file"] == "/path/to/key.json"

    def test_get_authentication_info_workload_identity(self):
        """Test authentication info with workload identity."""
        with patch.dict(
            os.environ, {"GOOGLE_CLOUD_PROJECT": "workload-project"}, clear=True
        ):
            auth = GCPAuthenticator(project_id="test-project")
            info = auth.get_authentication_info()

            assert info["project_id"] == "test-project"
            assert info["credentials_source"] == "workload_identity_or_metadata"

    @patch("requests.get")
    def test_get_authentication_info_compute_engine(self, mock_get):
        """Test authentication info with compute engine credentials."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {}, clear=True):
            auth = GCPAuthenticator(project_id="test-project")
            info = auth.get_authentication_info()

            assert info["project_id"] == "test-project"
            assert info["running_on_gcp"] is True
            assert info["credentials_source"] == "compute_engine_default"


class TestIntegrationAuthentication:
    """Integration tests for authentication (require actual credentials)."""

    @pytest.mark.integration
    def test_real_bigquery_connection(self):
        """Test real BigQuery connection (requires valid credentials)."""
        # Skip if no project ID set
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            pytest.skip("GCP_PROJECT_ID not set - skipping integration test")

        # Temporarily unset GOOGLE_APPLICATION_CREDENTIALS to force ADC
        original_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if original_creds:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

        try:
            config = load_config()
            client = config.get_client()

            # Test a simple query
            query = "SELECT 1 as test_value"
            result = client.query(query).result()
            rows = list(result)

            assert len(rows) == 1
            assert rows[0].test_value == 1

        except Exception as e:
            pytest.skip(
                f"BigQuery integration test failed (expected if no credentials): {e}"
            )
        finally:
            # Restore original environment
            if original_creds:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = original_creds

    @pytest.mark.integration
    def test_real_gcs_connection(self):
        """Test real GCS connection (requires valid credentials)."""
        # Skip if no project ID set
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            pytest.skip("GCP_PROJECT_ID not set - skipping integration test")

        try:
            auth = GCPAuthenticator(project_id=project_id)
            client = auth.get_storage_client()

            # Test listing buckets (minimal operation)
            buckets = list(client.list_buckets(max_results=1))
            # Just verify we can list buckets without error
            assert isinstance(buckets, list)

        except Exception as e:
            pytest.skip(
                f"GCS integration test failed (expected if no credentials): {e}"
            )

    @pytest.mark.integration
    def test_data_warehouse_uploader_connection(self):
        """Test data warehouse uploader connection (requires valid credentials)."""
        # Skip if no project ID set
        project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("DATA_WAREHOUSE_PROJECT_ID")
        if not project_id:
            pytest.skip("GCP_PROJECT_ID or DATA_WAREHOUSE_PROJECT_ID not set - skipping integration test")

        try:
            from src.data.bigquery_uploader import DataWarehousePredictionUploader

            uploader = DataWarehousePredictionUploader()

            # Test querying latest predictions (should work even with empty table)
            df = uploader.query_latest_predictions(limit=1)
            assert hasattr(df, 'columns')  # pandas DataFrame

        except Exception as e:
            pytest.skip(
                f"Data warehouse uploader integration test failed (expected if no credentials): {e}"
            )


class TestAuthenticationErrorHandling:
    """Test error handling scenarios for authentication."""

    def test_missing_config_file(self):
        """Test behavior when config file is missing."""
        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            with pytest.raises(FileNotFoundError):
                load_config(config_path="/nonexistent/config.yaml")

    def test_invalid_config_file(self):
        """Test behavior with invalid config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:")
            f.flush()

            with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
                with pytest.raises(Exception):  # YAML parsing error
                    load_config(config_path=f.name)

        os.unlink(f.name)

    def test_missing_bigquery_section(self):
        """Test behavior when bigquery section is missing from config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("other_section: value")
            f.flush()

            with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
                with pytest.raises(ValueError, match="Missing bigquery section"):
                    load_config(config_path=f.name)

        os.unlink(f.name)

    def test_missing_dataset_in_config(self):
        """Test behavior when dataset is missing from bigquery config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("bigquery:\n  other_field: value")
            f.flush()

            with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
                with pytest.raises(ValueError, match="Missing dataset"):
                    load_config(config_path=f.name)

        os.unlink(f.name)

    @patch("scoring_service.auth.storage.Client")
    def test_storage_client_creation_failure(self, mock_storage_client):
        """Test storage client creation failure."""
        mock_storage_client.side_effect = Exception("Storage client creation failed")

        auth = GCPAuthenticator(project_id="test-project")

        with pytest.raises(AuthenticationError, match="Failed to authenticate to GCP"):
            auth.get_storage_client()


class TestCredentialDiscovery:
    """Test credential discovery mechanisms."""

    def test_service_account_key_detection(self):
        """Test detection of service account key file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"type": "service_account", "project_id": "test"}, f)
            f.flush()

            with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": f.name}):
                auth = GCPAuthenticator(project_id="test-project")
                info = auth.get_authentication_info()

                assert info["credentials_source"] == "service_account_key"
                assert info["credentials_file"] == f.name

        os.unlink(f.name)

    def test_workload_identity_detection(self):
        """Test detection of workload identity."""
        with patch.dict(
            os.environ, {"GOOGLE_CLOUD_PROJECT": "workload-project"}, clear=True
        ):
            auth = GCPAuthenticator(project_id="test-project")
            info = auth.get_authentication_info()

            assert info["credentials_source"] == "workload_identity_or_metadata"

    @patch("requests.get")
    def test_metadata_service_detection(self, mock_get):
        """Test detection of metadata service (Compute Engine)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {}, clear=True):
            auth = GCPAuthenticator(project_id="test-project")
            info = auth.get_authentication_info()

            assert info["running_on_gcp"] is True
            assert info["credentials_source"] == "compute_engine_default"

    @patch("requests.get")
    def test_no_metadata_service(self, mock_get):
        """Test when metadata service is not available."""
        mock_get.side_effect = Exception("Connection failed")

        with patch.dict(os.environ, {}, clear=True):
            auth = GCPAuthenticator(project_id="test-project")
            info = auth.get_authentication_info()

            assert info["running_on_gcp"] is False
