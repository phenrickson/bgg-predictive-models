import os
import sys
import dotenv
import pytest
from pathlib import Path

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
def test_fixtures_path():
    """
    Return path to test fixtures directory.

    The fixtures directory contains sample_games.parquet with real BGG data
    pulled from BigQuery for testing purposes.
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_games_path(test_fixtures_path):
    """
    Return path to sample games parquet file.

    This fixture provides the path to sample_games.parquet containing
    real BGG game data for testing transformers and preprocessors.
    """
    data_file = test_fixtures_path / "sample_games.parquet"
    if not data_file.exists():
        pytest.skip(
            f"Test fixture {data_file} not found. "
            "Run 'python tests/fixtures/generate_test_data.py' to create it."
        )
    return data_file


@pytest.fixture(scope="session")
def raw_data_path():
    """
    Legacy fixture - returns path to data/raw directory.

    Note: Most tests should use sample_games_path instead for the test fixture.
    This fixture is kept for backwards compatibility with integration tests
    that need the full raw data.
    """
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    return data_dir


@pytest.fixture
def raw_data(raw_data_path):
    """
    Provides access to raw data for tests.

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
