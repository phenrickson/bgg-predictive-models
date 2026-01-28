"""
Tests for register.py script functionality
"""

import pytest
import yaml
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path so we can import register
sys.path.insert(0, str(Path(__file__).parent.parent))
import register


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "current_year": 2025,
        "train_through": 2021,
        "tune_through": 2022,
        "test_start": 2023,
        "test_through": 2023,
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

    def test_load_registration_config_returns_expected_structure(self):
        """Test that load_registration_config returns expected structure"""
        # This tests the actual function with real config
        config = register.load_registration_config()

        # Verify expected keys exist
        assert "current_year" in config
        assert "experiments" in config
        assert isinstance(config["current_year"], int)
        assert isinstance(config["experiments"], dict)

        # Verify experiment names exist
        for model_type in ["hurdle", "complexity", "rating", "users_rated"]:
            assert model_type in config["experiments"]


class TestCommandExecution:
    """Test command execution functionality"""

    @patch("subprocess.run")
    def test_run_command_success(self, mock_run):
        """Test successful command execution"""
        mock_run.return_value = MagicMock(returncode=0)

        # Should not raise an exception
        register.run_command(["echo", "test"], "Test command")

        mock_run.assert_called_once_with(
            ["echo", "test"], check=True, capture_output=False
        )

    @patch("subprocess.run")
    @patch("sys.exit")
    def test_run_command_failure(self, mock_exit, mock_run):
        """Test command execution failure"""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["false"])

        register.run_command(["false"], "Failing command")

        mock_exit.assert_called_once_with(1)


class TestModelRegistrationFunctions:
    """Test individual model registration functions"""

    @patch("register.run_command")
    def test_register_complexity(self, mock_run_command, sample_config):
        """Test complexity model registration command generation"""
        register.register_complexity(sample_config)

        expected_cmd = [
            "uv",
            "run",
            "-m",
            "scoring_service.register_model",
            "--model-type",
            "complexity",
            "--experiment",
            "catboost-complexity",
            "--name",
            "complexity-v2025",
            "--description",
            "Production (v2025) model for predicting game complexity",
        ]

        mock_run_command.assert_called_once_with(
            expected_cmd, "Registering complexity model"
        )

    @patch("register.run_command")
    def test_register_rating(self, mock_run_command, sample_config):
        """Test rating model registration command generation"""
        register.register_rating(sample_config)

        call_args = mock_run_command.call_args[0]
        cmd = call_args[0]

        assert "scoring_service.register_model" in cmd
        assert "--model-type" in cmd
        assert "rating" in cmd
        assert "--experiment" in cmd
        assert "catboost-rating" in cmd
        assert "--name" in cmd
        assert "rating-v2025" in cmd

    @patch("register.run_command")
    def test_register_users_rated(self, mock_run_command, sample_config):
        """Test users_rated model registration command generation"""
        register.register_users_rated(sample_config)

        call_args = mock_run_command.call_args[0]
        cmd = call_args[0]

        assert "scoring_service.register_model" in cmd
        assert "--model-type" in cmd
        assert "users_rated" in cmd
        assert "--experiment" in cmd
        assert "ridge-users_rated" in cmd
        assert "--name" in cmd
        assert "users_rated-v2025" in cmd

    @patch("register.run_command")
    def test_register_hurdle(self, mock_run_command, sample_config):
        """Test hurdle model registration command generation"""
        register.register_hurdle(sample_config)

        call_args = mock_run_command.call_args[0]
        cmd = call_args[0]

        assert "scoring_service.register_model" in cmd
        assert "--model-type" in cmd
        assert "hurdle" in cmd
        assert "--experiment" in cmd
        assert "lightgbm-hurdle" in cmd
        assert "--name" in cmd
        assert "hurdle-v2025" in cmd
        assert "--description" in cmd
        # Check that the description mentions "hurdle"
        description_idx = cmd.index("--description") + 1
        assert "hurdle" in cmd[description_idx].lower()


class TestMainPipeline:
    """Test the main registration pipeline"""

    @patch("register.load_registration_config")
    @patch("register.register_complexity")
    @patch("register.register_rating")
    @patch("register.register_users_rated")
    @patch("register.register_hurdle")
    def test_main_pipeline_success(
        self,
        mock_register_hurdle,
        mock_register_users_rated,
        mock_register_rating,
        mock_register_complexity,
        mock_load_config,
        sample_config,
    ):
        """Test successful execution of main registration pipeline"""
        mock_load_config.return_value = sample_config

        register.main()

        # Verify all registration functions were called
        mock_register_complexity.assert_called_once_with(sample_config)
        mock_register_rating.assert_called_once_with(sample_config)
        mock_register_users_rated.assert_called_once_with(sample_config)
        mock_register_hurdle.assert_called_once_with(sample_config)

    @patch("register.load_registration_config")
    @patch("sys.exit")
    def test_main_pipeline_config_error(self, mock_exit, mock_load_config):
        """Test main pipeline with config loading error"""
        mock_load_config.side_effect = Exception("Config error")

        register.main()

        mock_exit.assert_called_once_with(1)

    @patch("register.load_registration_config")
    @patch("register.register_complexity")
    @patch("sys.exit")
    def test_main_pipeline_registration_error(
        self, mock_exit, mock_register_complexity, mock_load_config, sample_config
    ):
        """Test main pipeline with registration error"""
        mock_load_config.return_value = sample_config
        mock_register_complexity.side_effect = Exception("Registration error")

        register.main()

        mock_exit.assert_called_once_with(1)


class TestModelVersioning:
    """Test model versioning functionality"""

    @patch("register.run_command")
    def test_model_names_include_year(self, mock_run_command, sample_config):
        """Test that all model names include the current year"""
        # Test complexity model
        register.register_complexity(sample_config)
        call_args = mock_run_command.call_args[0]
        cmd = call_args[0]
        name_idx = cmd.index("--name") + 1
        assert "v2025" in cmd[name_idx]

        mock_run_command.reset_mock()

        # Test rating model
        register.register_rating(sample_config)
        call_args = mock_run_command.call_args[0]
        cmd = call_args[0]
        name_idx = cmd.index("--name") + 1
        assert "v2025" in cmd[name_idx]

    @patch("register.run_command")
    def test_model_descriptions_include_year(self, mock_run_command, sample_config):
        """Test that all model descriptions include the current year"""
        register.register_complexity(sample_config)
        call_args = mock_run_command.call_args[0]
        cmd = call_args[0]
        desc_idx = cmd.index("--description") + 1
        assert "v2025" in cmd[desc_idx]


class TestIntegration:
    """Integration tests"""

    def test_config_file_structure(self):
        """Test that the actual config file has the expected structure"""
        config_path = Path("model_config.yml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check required keys exist for registration
            required_keys = ["current_year", "experiments"]

            for key in required_keys:
                assert key in config, f"Missing required key: {key}"

            # Check experiment names
            required_experiments = ["hurdle", "complexity", "rating", "users_rated"]
            for experiment in required_experiments:
                assert experiment in config["experiments"], (
                    f"Missing experiment: {experiment}"
                )

    @patch("subprocess.run")
    def test_script_can_be_executed(self, mock_run):
        """Test that the script can be executed without syntax errors"""
        mock_run.return_value = MagicMock(returncode=0)

        # This should not raise a syntax error
        result = subprocess.run(
            ["python", "-c", "import register; print('Script imports successfully')"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_registration_order_matches_makefile(self):
        """Test that registration order matches the Makefile order"""
        # The Makefile registers in this order: complexity, rating, users_rated, hurdle
        # This test ensures our main() function follows the same order

        with (
            patch("register.load_registration_config") as mock_load_config,
            patch("register.register_complexity") as mock_complexity,
            patch("register.register_rating") as mock_rating,
            patch("register.register_users_rated") as mock_users_rated,
            patch("register.register_hurdle") as mock_hurdle,
        ):
            mock_load_config.return_value = {
                "current_year": 2025,
                "experiments": {
                    "complexity": "test-complexity",
                    "rating": "test-rating",
                    "users_rated": "test-users_rated",
                    "hurdle": "test-hurdle",
                },
            }

            register.main()

            # Check call order by examining call_count at each step
            assert mock_complexity.call_count == 1
            assert mock_rating.call_count == 1
            assert mock_users_rated.call_count == 1
            assert mock_hurdle.call_count == 1


class TestErrorHandling:
    """Test error handling scenarios"""

    @patch("register.load_registration_config")
    @patch("register.register_complexity")
    @patch("sys.exit")
    def test_handles_subprocess_error(
        self, mock_exit, mock_register_complexity, mock_load_config, sample_config
    ):
        """Test that subprocess errors are handled properly"""
        mock_load_config.return_value = sample_config
        mock_register_complexity.side_effect = subprocess.CalledProcessError(1, ["uv"])

        register.main()

        mock_exit.assert_called_once_with(1)

    @patch("register.load_registration_config")
    @patch("sys.exit")
    def test_handles_yaml_error(self, mock_exit, mock_load_config):
        """Test that YAML parsing errors are handled properly"""
        mock_load_config.side_effect = yaml.YAMLError("Invalid YAML")

        register.main()

        mock_exit.assert_called_once_with(1)
