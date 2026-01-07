"""Test the test fixtures functionality."""

from pathlib import Path
import pandas as pd


def test_fixtures_path_provides_directory(test_fixtures_path):
    """Test that the test_fixtures_path fixture provides a valid directory."""
    assert isinstance(test_fixtures_path, Path)
    assert test_fixtures_path.exists()
    assert test_fixtures_path.is_dir()


def test_sample_games_path_provides_file(sample_games_path):
    """Test that the sample_games_path fixture provides a valid parquet file."""
    assert isinstance(sample_games_path, Path)
    assert sample_games_path.exists()
    assert sample_games_path.is_file()
    assert sample_games_path.suffix == ".parquet"


def test_sample_games_has_expected_columns(sample_games_path):
    """Test that the sample games fixture has expected columns for testing."""
    df = pd.read_parquet(sample_games_path)

    # Core columns needed for transformers
    required_columns = [
        "game_id",
        "name",
        "year_published",
        "min_players",
        "max_players",
        "min_playtime",
        "max_playtime",
        "min_age",
        "mechanics",
        "categories",
        "designers",
        "publishers",
        "description",
    ]

    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"


def test_sample_games_has_reasonable_size(sample_games_path):
    """Test that the sample games fixture has a reasonable number of rows."""
    df = pd.read_parquet(sample_games_path)
    # Should have at least 100 rows for meaningful tests
    assert len(df) >= 100, f"Sample games has too few rows: {len(df)}"
    # Should not be too large (we limited to 1000 in generation)
    assert len(df) <= 2000, f"Sample games has too many rows: {len(df)}"
