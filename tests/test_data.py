"""Tests for the games_features view and data loading."""

import os
import yaml
from pathlib import Path
from unittest.mock import patch

import pytest
import polars as pl

from src.data.config import load_config
from src.data.loader import BGGDataLoader


@pytest.fixture
def bigquery_config():
    """Create a BigQuery configuration for testing using load_config()."""
    # Temporarily unset GOOGLE_APPLICATION_CREDENTIALS to force ADC
    original_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if original_creds:
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    try:
        return load_config()
    finally:
        # Restore original environment
        if original_creds:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = original_creds


@pytest.fixture
def data_loader(bigquery_config):
    """Create a BGGDataLoader for testing."""
    return BGGDataLoader(bigquery_config)


def test_games_features_view_exists(bigquery_config):
    """Test that the games_features_materialized view exists and is accessible."""
    client = bigquery_config.get_client()

    # Query to check if view exists using proper information schema
    query = f"""
    SELECT table_name
    FROM `{bigquery_config.project_id}.{bigquery_config.dataset}.INFORMATION_SCHEMA.TABLES`
    WHERE table_name = 'games_features_materialized'
    """

    # Convert pandas DataFrame to Polars DataFrame
    result = pl.from_pandas(client.query(query).to_dataframe())
    assert len(result) == 1, "games_features_materialized view does not exist"


def test_games_features_view_structure(bigquery_config):
    """Test the structure of the games_features view."""
    client = bigquery_config.get_client()

    # Query to get a sample row
    query = f"""
    SELECT *
    FROM `{bigquery_config.project_id}.{bigquery_config.dataset}.games_features_materialized`
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
        assert isinstance(
            result[col][0].to_list(), list
        ), f"Column {col} is not an array"


def test_data_loader_training_data(data_loader):
    """Test loading training data from the view."""
    # Load a small sample of training data (returns single DataFrame when no preprocessor)
    data = data_loader.load_training_data(end_train_year=2022, min_ratings=1000)

    # Check that we got some data
    assert len(data) > 0, "No training data loaded"

    # Check that the data has the expected target columns
    expected_targets = ["hurdle", "complexity", "rating", "users_rated"]
    for target in expected_targets:
        assert target in data.columns, f"Target {target} missing from data"

    # Check that the data has the expected feature columns
    expected_features = [
        "game_id",
        "year_published",
        "average_rating",
        "average_weight",
        "min_players",
        "max_players",
        "categories",
        "mechanics",
    ]
    for feature in expected_features:
        assert feature in data.columns, f"Feature {feature} missing from data"


def test_data_loader_prediction_data(data_loader):
    """Test loading prediction data for specific games."""
    # Load prediction data for Gloomhaven (ID: 174430)
    features = data_loader.load_prediction_data(game_ids=[174430])

    # Check that we got exactly one game
    assert len(features) == 1, "Expected exactly one game in prediction data"
