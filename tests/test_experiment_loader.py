"""
Tests for the experiment loader functionality.
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.utils.experiment_loader import ExperimentLoader, get_experiment_loader


class TestExperimentLoader:
    """Test cases for ExperimentLoader class."""

    @pytest.fixture
    def mock_storage_client(self):
        """Mock Google Cloud Storage client."""
        with patch("src.utils.experiment_loader.storage.Client") as mock_client:
            mock_bucket = Mock()
            mock_client.return_value.bucket.return_value = mock_bucket
            yield mock_client, mock_bucket

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        with patch("src.utils.experiment_loader.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.get_bucket_name.return_value = "test-bucket"
            mock_load_config.return_value = mock_config
            yield mock_config

    def test_init_with_bucket_name(self, mock_storage_client):
        """Test initialization with explicit bucket name."""
        mock_client, mock_bucket = mock_storage_client

        loader = ExperimentLoader(bucket_name="test-bucket")

        assert loader.bucket_name == "test-bucket"
        mock_client.assert_called_once()
        mock_client.return_value.bucket.assert_called_once_with("test-bucket")

    def test_init_with_config(self, mock_storage_client, mock_config):
        """Test initialization using config file."""
        mock_client, mock_bucket = mock_storage_client

        loader = ExperimentLoader()

        assert loader.bucket_name == "test-bucket"
        mock_config.get_bucket_name.assert_called_once()

    def test_list_model_types(self, mock_storage_client):
        """Test listing model types."""
        mock_client, mock_bucket = mock_storage_client

        # Mock the list_blobs response
        mock_page = Mock()
        mock_page.prefixes = [
            "models/experiments/catboost-complexity/",
            "models/experiments/ridge-users_rated/",
            "models/experiments/predictions/",  # Should be filtered out
        ]
        mock_blobs = Mock()
        mock_blobs.pages = [mock_page]
        mock_bucket.list_blobs.return_value = mock_blobs

        loader = ExperimentLoader(bucket_name="test-bucket")
        model_types = loader.list_model_types()

        expected_types = ["catboost-complexity", "ridge-users_rated"]
        assert sorted(model_types) == sorted(expected_types)
        mock_bucket.list_blobs.assert_called_once_with(
            prefix="models/experiments/", delimiter="/"
        )

    def test_list_experiments(self, mock_storage_client):
        """Test listing experiments for a model type."""
        mock_client, mock_bucket = mock_storage_client

        # Mock the list_blobs response for experiment directories
        mock_page = Mock()
        mock_page.prefixes = [
            "models/experiments/catboost-complexity/experiment1/",
            "models/experiments/catboost-complexity/experiment2/",
        ]
        mock_blobs = Mock()
        mock_blobs.pages = [mock_page]
        mock_bucket.list_blobs.return_value = mock_blobs

        # Mock metadata loading
        mock_blob = Mock()
        mock_blob.download_as_text.return_value = json.dumps(
            {
                "name": "experiment1",
                "timestamp": "2023-01-01T00:00:00Z",
                "status": "completed",
            }
        )
        mock_bucket.blob.return_value = mock_blob

        loader = ExperimentLoader(bucket_name="test-bucket")
        experiments = loader.list_experiments("catboost-complexity")

        assert len(experiments) == 2
        assert all("full_name" in exp for exp in experiments)
        assert all("model_type" in exp for exp in experiments)

    def test_load_experiment_details(self, mock_storage_client):
        """Test loading detailed experiment information."""
        mock_client, mock_bucket = mock_storage_client

        # Mock file loading
        def mock_blob_side_effect(path):
            mock_blob = Mock()
            if "metadata.json" in path:
                mock_blob.download_as_text.return_value = json.dumps({"name": "test"})
            elif "metrics.json" in path:
                mock_blob.download_as_text.return_value = json.dumps({"accuracy": 0.95})
            elif "parameters.json" in path:
                mock_blob.download_as_text.return_value = json.dumps({"lr": 0.01})
            else:
                mock_blob.download_as_text.side_effect = Exception("Not found")
            return mock_blob

        mock_bucket.blob.side_effect = mock_blob_side_effect

        loader = ExperimentLoader(bucket_name="test-bucket")
        details = loader.load_experiment_details("catboost-complexity", "experiment1")

        assert "metadata" in details
        assert "metrics" in details
        assert "parameters" in details

    def test_load_predictions(self, mock_storage_client):
        """Test loading predictions data."""
        mock_client, mock_bucket = mock_storage_client

        # Mock parquet file download
        mock_blob = Mock()

        # Create a temporary parquet file for testing
        import pandas as pd

        test_df = pd.DataFrame(
            {"prediction": [1.0, 2.0, 3.0], "actual": [1.1, 1.9, 3.1]}
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            test_df.to_parquet(tmp_file.name)

            def mock_download_to_filename(filename):
                # Copy our test file to the requested filename
                import shutil

                shutil.copy2(tmp_file.name, filename)

            mock_blob.download_to_filename.side_effect = mock_download_to_filename
            mock_bucket.blob.return_value = mock_blob

            loader = ExperimentLoader(bucket_name="test-bucket")
            result_df = loader.load_predictions(
                "catboost-complexity", "experiment1", "test"
            )

            assert result_df is not None
            assert len(result_df) == 3
            assert "prediction" in result_df.columns
            assert "actual" in result_df.columns

            # Clean up
            Path(tmp_file.name).unlink()

    def test_load_feature_importance_json(self, mock_storage_client):
        """Test loading feature importance from JSON file."""
        mock_client, mock_bucket = mock_storage_client

        mock_blob = Mock()
        mock_blob.download_as_text.return_value = json.dumps(
            {"feature1": 0.5, "feature2": 0.3, "feature3": 0.2}
        )
        mock_bucket.blob.return_value = mock_blob

        loader = ExperimentLoader(bucket_name="test-bucket")
        importance = loader.load_feature_importance(
            "catboost-complexity", "experiment1"
        )

        assert importance is not None
        assert "feature1" in importance
        assert importance["feature1"] == 0.5

    def test_load_feature_importance_pickle_fallback(self, mock_storage_client):
        """Test loading feature importance from pickle file when JSON not found."""
        mock_client, mock_bucket = mock_storage_client

        # Mock JSON file not found, but pickle file exists
        def mock_blob_side_effect(path):
            mock_blob = Mock()
            if "feature_importance.json" in path:
                from google.cloud.exceptions import NotFound

                mock_blob.download_as_text.side_effect = NotFound("JSON not found")
            elif "feature_importance.pkl" in path:
                # Mock pickle file download
                import pickle

                test_data = {"feature1": 0.8, "feature2": 0.2}

                with tempfile.NamedTemporaryFile(
                    suffix=".pkl", delete=False
                ) as tmp_file:
                    pickle.dump(test_data, tmp_file)

                    def mock_download_to_filename(filename):
                        import shutil

                        shutil.copy2(tmp_file.name, filename)

                    mock_blob.download_to_filename.side_effect = (
                        mock_download_to_filename
                    )

                    # Clean up temp file after test
                    import atexit

                    atexit.register(lambda: Path(tmp_file.name).unlink())

            return mock_blob

        mock_bucket.blob.side_effect = mock_blob_side_effect

        loader = ExperimentLoader(bucket_name="test-bucket")
        importance = loader.load_feature_importance(
            "catboost-complexity", "experiment1"
        )

        assert importance is not None
        assert "feature1" in importance

    def test_clear_cache(self, mock_storage_client):
        """Test cache clearing functionality."""
        mock_client, mock_bucket = mock_storage_client

        loader = ExperimentLoader(bucket_name="test-bucket")

        # Add some data to cache
        loader._metadata_cache["test"] = {"data": "test"}
        loader._experiments_cache["test"] = [{"exp": "test"}]

        # Clear cache
        loader.clear_cache()

        assert len(loader._metadata_cache) == 0
        assert len(loader._experiments_cache) == 0

    def test_caching_behavior(self, mock_storage_client):
        """Test that caching works correctly."""
        mock_client, mock_bucket = mock_storage_client

        # Mock the list_blobs response
        mock_page = Mock()
        mock_page.prefixes = ["models/experiments/test-model/exp1/"]
        mock_blobs = Mock()
        mock_blobs.pages = [mock_page]
        mock_bucket.list_blobs.return_value = mock_blobs

        # Mock metadata loading
        mock_blob = Mock()
        mock_blob.download_as_text.return_value = json.dumps(
            {"name": "exp1", "timestamp": "2023-01-01T00:00:00Z"}
        )
        mock_bucket.blob.return_value = mock_blob

        loader = ExperimentLoader(bucket_name="test-bucket")

        # First call should hit the API
        experiments1 = loader.list_experiments("test-model")

        # Second call should use cache
        experiments2 = loader.list_experiments("test-model")

        assert experiments1 == experiments2
        # Should only call list_blobs once due to caching
        assert mock_bucket.list_blobs.call_count == 1


class TestGetExperimentLoader:
    """Test the global experiment loader function."""

    def setup_method(self):
        """Reset global loader before each test."""
        # Reset the global loader instance
        import src.utils.experiment_loader

        src.utils.experiment_loader._experiment_loader = None

    def test_get_experiment_loader_singleton(self):
        """Test that get_experiment_loader returns the same instance."""
        with patch("src.utils.experiment_loader.ExperimentLoader") as mock_loader_class:
            mock_instance = Mock()
            mock_loader_class.return_value = mock_instance

            # First call
            loader1 = get_experiment_loader()

            # Second call should return same instance
            loader2 = get_experiment_loader()

            assert loader1 is loader2
            # ExperimentLoader should only be instantiated once
            mock_loader_class.assert_called_once()

    def test_get_experiment_loader_with_params(self):
        """Test get_experiment_loader with parameters."""
        with patch("src.utils.experiment_loader.ExperimentLoader") as mock_loader_class:
            mock_instance = Mock()
            mock_loader_class.return_value = mock_instance

            loader = get_experiment_loader(
                bucket_name="test-bucket", config_path="test.yaml"
            )

            mock_loader_class.assert_called_once_with("test-bucket", "test.yaml")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
