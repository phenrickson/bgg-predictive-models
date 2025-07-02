"""Script to fetch and save raw data from the materialized view."""
import argparse
from pathlib import Path

import polars as pl
from google.cloud import bigquery

from .config import load_config
from .loader import BGGDataLoader

# Load data
def get_raw_data(output_dir = 'data/raw') -> None:
    """Fetch raw data from the materialized view and save locally.
    
    Args:
        output_dir: Directory to save data files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize BigQuery client from config
    config = load_config()
    loader = BGGDataLoader(config)
    
    df = loader.load_training_data(end_train_year = 2022, min_ratings = 0)
    df.write_parquet('data/raw/game_features.parquet')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch raw data from the materialized view and save locally"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save data files (default: data/raw)",
    )
    
    get_raw_data()


if __name__ == "__main__":
    main()
