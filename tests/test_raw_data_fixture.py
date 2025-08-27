"""Test the raw data fixture functionality."""

from pathlib import Path


def test_raw_data_fixture_provides_path(raw_data):
    """Test that the raw_data fixture provides a valid path."""
    assert isinstance(raw_data, Path)
    assert raw_data.exists()
    assert raw_data.is_dir()


def test_raw_data_contains_parquet_file(raw_data):
    """Test that the raw data directory contains the expected parquet file."""
    parquet_file = raw_data / "game_features.parquet"
    assert parquet_file.exists()
    assert parquet_file.is_file()
    assert parquet_file.suffix == ".parquet"


def test_raw_data_path_fixture_provides_path(raw_data_path):
    """Test that the raw_data_path fixture provides a valid path."""
    assert isinstance(raw_data_path, Path)
    assert raw_data_path.exists()
    assert raw_data_path.is_dir()

    # Verify the expected file exists
    parquet_file = raw_data_path / "game_features.parquet"
    assert parquet_file.exists()
