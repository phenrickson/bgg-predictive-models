import os
import sys
import subprocess
import dotenv
import pytest
from pathlib import Path
import logging

from src.utils.logging import setup_logging

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    dotenv.load_dotenv(env_path)

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging for tests
logger = setup_logging()


@pytest.fixture(scope="session")
def raw_data_path():
    """
    Ensure raw data exists, generating it if missing.

    This fixture checks for the existence of game_features.parquet in data/raw.
    If the file doesn't exist, it runs the data generation script.

    Scope is 'session' to run only once per test session.
    """
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "raw" / "game_features.parquet"

    # Check if the raw data file exists
    if not data_file.exists():
        try:
            logger.info(f"Raw data file {data_file} not found. Loading raw data...")
            # Run the data generation script
            result = subprocess.run(
                [sys.executable, "-m", "src.data.get_raw_data"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Raw data loaded successfully.")
            if result.stdout.strip():
                logger.debug(f"Output: {result.stdout}")

            # Verify the file was created
            if not data_file.exists():
                pytest.fail(
                    f"Data generation script completed but {data_file} was not created"
                )

        except subprocess.CalledProcessError as e:
            pytest.fail(f"Failed to generate raw data: {e.stderr}")
        except Exception as e:
            pytest.fail(f"Unexpected error during data generation: {str(e)}")

    return data_file.parent


@pytest.fixture
def raw_data(raw_data_path):
    """
    Provides access to raw data for tests.
    Depends on raw_data_path to ensure data is generated first.

    Returns the path to the raw data directory.
    """
    return raw_data_path


def check_experiments_exist():
    """
    Check if required experiments exist for geek rating tests.

    Looks for any experiment with a finalized model in each of the required
    model type directories: hurdle, complexity, rating, users_rated.

    Returns:
        bool: True if all required model types have at least one experiment, False otherwise
    """
    project_root = Path(__file__).parent.parent
    experiments_dir = project_root / "models" / "experiments"

    # Required model types for geek rating tests
    required_model_types = ["hurdle", "complexity", "rating", "users_rated"]

    for model_type in required_model_types:
        model_type_dir = experiments_dir / model_type

        if not model_type_dir.exists():
            return False

        # Look for any experiment directory in this model type
        experiment_found = False
        for exp_dir in model_type_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            # Check for finalized model in this experiment
            finalized_model = exp_dir / "finalized" / "pipeline.pkl"
            if finalized_model.exists():
                experiment_found = True
                break

            # Check for versioned finalized models
            version_dirs = [
                d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("v")
            ]
            if version_dirs:
                latest_version = max(version_dirs, key=lambda x: int(x.name[1:]))
                versioned_finalized = latest_version / "finalized" / "pipeline.pkl"
                if versioned_finalized.exists():
                    experiment_found = True
                    break

        # If no experiment found for this model type, return False
        if not experiment_found:
            return False

    # All required model types have at least one experiment
    return True
