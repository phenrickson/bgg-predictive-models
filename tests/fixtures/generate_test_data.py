"""Pull a small sample of real data from BigQuery for test fixtures.

This script queries the data warehouse and saves a small sample of real BGG data
that can be used for testing transformers, preprocessors, and pipelines.

Run this script to regenerate the test data fixture:
    python tests/fixtures/generate_test_data.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config


def main():
    """Pull sample data from BigQuery and save as fixture."""
    fixtures_dir = Path(__file__).parent
    output_path = fixtures_dir / "sample_games.parquet"

    print("Loading config...")
    config = load_config()

    print("Connecting to BigQuery...")
    client = config.data_warehouse.get_client()

    # Query a random sample of games with good coverage of features
    # Filter for complete data needed by transformers
    dw = config.data_warehouse
    query = f"""
    SELECT *
    FROM `{dw.project_id}.{dw.features_dataset}.{dw.features_table}`
    WHERE users_rated >= 100
      AND min_players IS NOT NULL AND min_players > 0
      AND max_players IS NOT NULL AND max_players > 0
      AND min_playtime IS NOT NULL
      AND max_playtime IS NOT NULL
      AND year_published IS NOT NULL
    ORDER BY RAND()
    LIMIT 1000
    """

    print("Querying data warehouse for sample games...")
    df = client.query(query).to_dataframe()

    print(f"Retrieved {len(df)} games")
    print(f"Columns: {list(df.columns)}")

    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False)

    print(f"\nTest fixture saved: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
