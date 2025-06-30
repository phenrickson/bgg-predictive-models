"""Script to fetch and save raw data from the materialized view."""
import argparse
from pathlib import Path

import polars as pl
from google.cloud import bigquery

from .config import load_config


def get_raw_data(output_dir: str) -> None:
    """Fetch raw data from the materialized view and save locally.
    
    Args:
        output_dir: Directory to save data files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize BigQuery client from config
    config = load_config()
    client = config.get_client()
    
    # Query to get all data from the materialized view
    query = f"""
    SELECT *
    FROM `{config.project_id}.{config.dataset}.games_features_materialized`
    """
    
    print("Fetching data from BigQuery...")
    
    # Execute query and convert result to polars DataFrame
    pandas_df = client.query(query).to_dataframe()
    df = pl.from_pandas(pandas_df)
    
    print(f"Retrieved {len(df)} rows with {len(df.columns)} columns")
    
    # Save raw data as parquet
    output_file = output_path / "games_features_raw.parquet"
    df.write_parquet(output_file)
    print(f"Saved raw data to {output_file}")
    
    # Print column information
    print("\nColumn information:")
    for col in df.columns:
        col_type = df[col].dtype
        print(f"  {col}: {col_type}")


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
    
    args = parser.parse_args()
    get_raw_data(args.output_dir)


if __name__ == "__main__":
    main()
