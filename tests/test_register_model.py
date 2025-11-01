"""Tests for register_model.py functionality."""

import os
import pytest
from unittest.mock import patch, MagicMock

from scoring_service.register_model import validate_environment, register_model


class TestEnvironmentValidation:
    """Test environment validation functionality."""

    def test_validate_environment_success(self):
        """Test successful environment validation with required vars."""
        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            # Should not raise an exception
            validate_environment()

    def test_validate_environment_missing_required(self):
        """Test environment validation fails when required vars missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                validate_environment()
            assert "Missing required environment variables: GCP_PROJECT_ID" in str(
                exc_info.value
            )

    def test_validate_environment_with_optional(self):
        """Test environment validation succeeds with optional vars."""
        with patch.dict(
            os.environ,
            {"GCP_PROJECT_ID": "test-project", "ENVIRONMENT": "test"},  # Optional var
        ):
            # Should not raise an exception
            validate_environment()


@pytest.fixture
def mock_experiment():
    """Create a mock experiment for testing."""
    experiment = MagicMock()
    experiment.name = "test-experiment"
    experiment.metadata = {"test": "metadata"}
    experiment.load_pipeline.return_value = MagicMock()
    experiment.get_model_info.return_value = {"test": "info"}
    return experiment


class TestModelRegistration:
    """Test model registration functionality."""

    @patch("scoring_service.register_model.load_config")
    @patch("scoring_service.register_model.ExperimentTracker")
    @patch("scoring_service.register_model.RegisteredModel")
    def test_register_model_with_config_bucket(
        self, mock_registered_model_cls, mock_tracker_cls, mock_load_config
    ):
        """Test model registration using bucket from config."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.get_bucket_name.return_value = "test-bucket"
        mock_load_config.return_value = mock_config

        mock_tracker = MagicMock()
        mock_tracker.list_experiments.return_value = [
            {"name": "test-exp", "version": 1}
        ]
        mock_tracker.load_experiment.return_value = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        mock_registered_model = MagicMock()
        mock_registered_model.register.return_value = {
            "name": "test-model",
            "version": 1,
            "description": "test description",
            "registered_at": "2025-01-01T00:00:00",
        }
        mock_registered_model_cls.return_value = mock_registered_model

        # Test registration
        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            result = register_model(
                model_type="test",
                experiment_name="test-exp",
                registered_name="test-model",
                description="test description",
            )

        # Verify bucket name was obtained from config
        mock_config.get_bucket_name.assert_called_once()
        mock_registered_model_cls.assert_called_once_with(
            model_type="test", bucket_name="test-bucket", project_id="test-project"
        )

        assert result["name"] == "test-model"
        assert result["version"] == 1

    @patch("scoring_service.register_model.load_config")
    @patch("scoring_service.register_model.ExperimentTracker")
    @patch("scoring_service.register_model.RegisteredModel")
    def test_register_model_with_provided_bucket(
        self, mock_registered_model_cls, mock_tracker_cls, mock_load_config
    ):
        """Test model registration using provided bucket name."""
        # Setup mocks
        mock_tracker = MagicMock()
        mock_tracker.list_experiments.return_value = [
            {"name": "test-exp", "version": 1}
        ]
        mock_tracker.load_experiment.return_value = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        mock_registered_model = MagicMock()
        mock_registered_model.register.return_value = {
            "name": "test-model",
            "version": 1,
            "description": "test description",
            "registered_at": "2025-01-01T00:00:00",
        }
        mock_registered_model_cls.return_value = mock_registered_model

        # Test registration with explicit bucket
        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            result = register_model(
                model_type="test",
                experiment_name="test-exp",
                registered_name="test-model",
                description="test description",
                bucket_name="provided-bucket",
            )

        # Verify provided bucket was used
        mock_load_config.assert_not_called()  # Config not loaded since bucket provided
        mock_registered_model_cls.assert_called_once_with(
            model_type="test", bucket_name="provided-bucket", project_id="test-project"
        )

        assert result["name"] == "test-model"
        assert result["version"] == 1

    def test_register_model_missing_project_id(self):
        """Test model registration fails without project ID."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                register_model(
                    model_type="test",
                    experiment_name="test-exp",
                    registered_name="test-model",
                    description="test description",
                )
            assert "Missing required environment variables: GCP_PROJECT_ID" in str(
                exc_info.value
            )
