"""
Tests for train.py script functionality
"""

import pytest
import yaml
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path so we can import train
sys.path.insert(0, str(Path(__file__).parent.parent))
import train


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "current_year": 2025,
        "train_end_year": 2021,
        "tune_end_year": 2022,
        "test_start_year": 2023,
        "test_end_year": 2023,
        "models": {
            "hurdle": "lightgbm",
            "complexity": "catboost",
            "rating": "catboost",
            "users_rated": "ridge",
        },
        "experiments": {
            "hurdle": "lightgbm-hurdle",
            "complexity": "catboost-complexity",
            "rating": "catboost-rating",
            "users_rated": "ridge-users_rated",
        },
        "model_settings": {
            "complexity": {"use_sample_weights": True},
            "rating": {"use_sample_weights": True, "min_ratings": 5},
            "users_rated": {"min_ratings": 0},
        },
        "paths": {
            "complexity_predictions": "models/experiments/predictions/catboost-complexity.parquet"
        },
    }


@pytest.fixture
def mock_config_file(tmp_path, sample_config):
    """Create a temporary config file for testing"""
    config_file = tmp_path / "model_config.yml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    return config_file


class TestConfigLoading:
    """Test configuration loading functionality"""

    def test_load_training_config_returns_expected_structure(self):
        """Test that load_training_config returns expected structure"""
        # This tests the actual function with real config
        config = train.load_training_config()

        # Verify expected keys exist
        assert "current_year" in config
        assert "train_end_year" in config
        assert "tune_end_year" in config
        assert "test_start_year" in config
        assert "test_end_year" in config
        assert "models" in config
        assert "experiments" in config
        assert "model_settings" in config
        assert "paths" in config

        # Verify model types exist
        for model_type in ["hurdle", "complexity", "rating", "users_rated"]:
            assert model_type in config["models"]
            assert model_type in config["experiments"]


class TestCommandExecution:
    """Test command execution functionality"""

    @patch("subprocess.run")
    def test_run_command_success(self, mock_run):
        """Test successful command execution"""
        mock_run.return_value = MagicMock(returncode=0)

        # Should not raise an exception
        train.run_command(["echo", "test"], "Test command")

        mock_run.assert_called_once_with(
            ["echo", "test"], check=True, capture_output=False
        )

    @patch("subprocess.run")
    @patch("sys.exit")
    def test_run_command_failure(self, mock_exit, mock_run):
        """Test command execution failure"""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["false"])

        train.run_command(["false"], "Failing command")

        mock_exit.assert_called_once_with(1)


class TestModelTrainingFunctions:
    """Test individual model training functions"""

    @patch("train.run_command")
    def test_train_hurdle(self, mock_run_command, sample_config):
        """Test hurdle model training command generation"""
        train.train_hurdle(sample_config)

        expected_cmd = [
            "uv",
            "run",
            "-m",
            "src.models.hurdle",
            "--experiment",
            "lightgbm-hurdle",
            "--model",
            "lightgbm",
            "--train-end-year",
            "2021",
            "--tune-start-year",
            "2021",
            "--tune-end-year",
            "2022",
            "--test-start-year",
            "2023",
            "--test-end-year",
            "2023",
        ]

        mock_run_command.assert_called_once_with(expected_cmd, "Training hurdle model")

    @patch("train.run_command")
    def test_train_complexity_with_sample_weights(
        self, mock_run_command, sample_config
    ):
        """Test complexity model training with sample weights"""
        train.train_complexity(sample_config)

        # Check that the command includes --use-sample-weights
        call_args = mock_run_command.call_args[0]
        cmd = call_args[0]

        assert "--use-sample-weights" in cmd
        assert "--model" in cmd
        assert "catboost" in cmd

    @patch("train.run_command")
    def test_train_rating_with_complexity_path(self, mock_run_command, sample_config):
        """Test rating model training with complexity predictions path"""
        train.train_rating(sample_config)

        call_args = mock_run_command.call_args[0]
        cmd = call_args[0]

        assert "--complexity-experiment" in cmd
        assert "catboost-complexity" in cmd
        assert "--local-complexity-path" in cmd
        assert "models/experiments/predictions/catboost-complexity.parquet" in cmd

    @patch("train.run_command")
    def test_train_users_rated_with_min_ratings(self, mock_run_command, sample_config):
        """Test users_rated model training with min ratings"""
        train.train_users_rated(sample_config)

        call_args = mock_run_command.call_args[0]
        cmd = call_args[0]

        assert "--min-ratings" in cmd
        assert "0" in cmd  # min_ratings value from config


class TestMainPipeline:
    """Test the main training pipeline"""

    @patch("train.load_training_config")
    @patch("train.train_hurdle")
    @patch("train.finalize_hurdle")
    @patch("train.score_hurdle")
    @patch("train.train_complexity")
    @patch("train.finalize_complexity")
    @patch("train.score_complexity")
    @patch("train.train_rating")
    @patch("train.finalize_rating")
    @patch("train.score_rating")
    @patch("train.train_users_rated")
    @patch("train.finalize_users_rated")
    @patch("train.score_users_rated")
    @patch("train.train_geek_rating")
    def test_main_pipeline_success(
        self,
        mock_geek,
        mock_score_users,
        mock_finalize_users,
        mock_train_users,
        mock_score_rating,
        mock_finalize_rating,
        mock_train_rating,
        mock_score_complexity,
        mock_finalize_complexity,
        mock_train_complexity,
        mock_score_hurdle,
        mock_finalize_hurdle,
        mock_train_hurdle,
        mock_load_config,
        sample_config,
    ):
        """Test successful execution of main pipeline"""
        mock_load_config.return_value = sample_config

        train.main()

        # Verify all training functions were called
        mock_train_hurdle.assert_called_once_with(sample_config)
        mock_finalize_hurdle.assert_called_once_with(sample_config)
        mock_score_hurdle.assert_called_once_with(sample_config)

        mock_train_complexity.assert_called_once_with(sample_config)
        mock_finalize_complexity.assert_called_once_with(sample_config)
        mock_score_complexity.assert_called_once_with(sample_config)

        mock_train_rating.assert_called_once_with(sample_config)
        mock_finalize_rating.assert_called_once_with(sample_config)
        mock_score_rating.assert_called_once_with(sample_config)

        mock_train_users.assert_called_once_with(sample_config)
        mock_finalize_users.assert_called_once_with(sample_config)
        mock_score_users.assert_called_once_with(sample_config)

        mock_geek.assert_called_once_with(sample_config)

    @patch("train.load_training_config")
    @patch("sys.exit")
    def test_main_pipeline_config_error(self, mock_exit, mock_load_config):
        """Test main pipeline with config loading error"""
        mock_load_config.side_effect = Exception("Config error")

        train.main()

        mock_exit.assert_called_once_with(1)


class TestIntegration:
    """Integration tests"""

    def test_config_file_structure(self):
        """Test that the actual config file has the expected structure"""
        config_path = Path("model_config.yml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check required keys exist
            required_keys = [
                "current_year",
                "train_end_year",
                "tune_end_year",
                "test_start_year",
                "test_end_year",
                "models",
                "experiments",
                "model_settings",
                "paths",
            ]

            for key in required_keys:
                assert key in config, f"Missing required key: {key}"

            # Check model types
            required_models = ["hurdle", "complexity", "rating", "users_rated"]
            for model in required_models:
                assert model in config["models"], f"Missing model: {model}"
                assert model in config["experiments"], f"Missing experiment: {model}"

    @patch("subprocess.run")
    def test_script_can_be_executed(self, mock_run):
        """Test that the script can be executed without syntax errors"""
        mock_run.return_value = MagicMock(returncode=0)

        # This should not raise a syntax error
        result = subprocess.run(
            ["python", "-c", "import train; print('Script imports successfully')"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
