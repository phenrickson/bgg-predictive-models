"""Tests for the games_features view and data loading."""

import os
import yaml
from pathlib import Path

import pytest
import polars as pl

from src.data.config import BigQueryConfig
from src.data.loader import BGGDataLoader


@pytest.fixture
def bigquery_config():
    """Create a BigQuery configuration for testing."""
    # Load dataset from config/bigquery.yaml
    config_path = Path(__file__).parent.parent / "config" / "bigquery.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    default_env = config.get("default_environment", "dev")
    env_config = config.get("environments", {}).get(default_env, {})

    # Create BigQuery config
    return BigQueryConfig(
        project_id=os.getenv("GCP_PROJECT_ID"),
        dataset=env_config.get("dataset"),
        credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    )


@pytest.fixture
def data_loader(bigquery_config):
    """Create a BGGDataLoader for testing."""
    return BGGDataLoader(bigquery_config)


def test_games_features_view_exists(bigquery_config):
    """Test that the games_features view exists and is accessible."""
    client = bigquery_config.get_client()

    # Query to check if view exists
    query = f"""
    SELECT table_name
    FROM `{bigquery_config.project_id}.{bigquery_config.dataset}.__TABLES_SUMMARY__`
    WHERE table_name = 'games_features'
    """

    # Convert pandas DataFrame to Polars DataFrame
    result = pl.from_pandas(client.query(query).to_dataframe())
    assert len(result) == 1, "games_features view does not exist"


def test_games_features_view_structure(bigquery_config):
    """Test the structure of the games_features view."""
    client = bigquery_config.get_client()

    # Query to get a sample row
    query = f"""
    SELECT *
    FROM `{bigquery_config.project_id}.{bigquery_config.dataset}.games_features`
    LIMIT 1
    """

    # Convert pandas DataFrame to Polars DataFrame
    result = pl.from_pandas(client.query(query).to_dataframe())

    # Check that the view has the expected columns
    expected_columns = [
        "game_id",
        "year_published",
        "average_rating",
        "average_weight",
        "users_rated",
        "min_players",
        "max_players",
        "min_playtime",
        "max_playtime",
        "min_age",
        "image",
        "thumbnail",
        "description",
        "categories",
        "mechanics",
        "publishers",
        "designers",
        "artists",
        "families",
    ]

    for col in expected_columns:
        assert col in result.columns, f"Column {col} missing from games_features view"

    # Check that array columns are actually arrays
    array_columns = [
        "categories",
        "mechanics",
        "publishers",
        "designers",
        "artists",
        "families",
    ]
    for col in array_columns:
        assert isinstance(result[col][0].to_list(), list), (
            f"Column {col} is not an array"
        )


def test_data_loader_training_data(data_loader):
    """Test loading training data from the view."""
    # Load a small sample of training data
    features, targets = data_loader.load_training_data(
        end_train_year=2022, min_ratings=1000
    )

    # Check that we got some data
    assert len(features) > 0, "No training data loaded"

    # Check that targets have the expected keys
    expected_targets = ["hurdle", "complexity", "rating", "users_rated"]
    for target in expected_targets:
        assert target in targets, f"Target {target} missing from targets"

    # Check that features include one-hot encoded categories and mechanics
    assert any(col.startswith("category_") for col in features.columns), (
        "No category features found"
    )
    assert any(col.startswith("mechanic_") for col in features.columns), (
        "No mechanic features found"
    )


def test_data_loader_prediction_data(data_loader):
    """Test loading prediction data for specific games."""
    # Load prediction data for Gloomhaven (ID: 174430)
    features = data_loader.load_prediction_data(game_ids=[174430])

    # Check that we got exactly one game
    assert len(features) == 1, "Expected exactly one game in prediction data"

    # Check that features include derived features
    assert "player_range" in features.columns, "Derived feature 'player_range' missing"
    assert "playtime_range" in features.columns, (
        "Derived feature 'playtime_range' missing"
    )
