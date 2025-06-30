"""Script to create the games_features view in BigQuery."""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.config import get_config_from_env


def read_sql_file(file_path: str) -> str:
    """Read SQL from a file.
    
    Args:
        file_path: Path to SQL file
        
    Returns:
        SQL content as string
    """
    with open(file_path, "r") as f:
        return f.read()


def create_view():
    """Create the games_features view in BigQuery."""
    # Get BigQuery configuration
    try:
        config = get_config_from_env()
    except ValueError as e:
        print(f"Error getting configuration: {e}")
        sys.exit(1)
    
    # Get BigQuery client
    client = config.get_client()
    
    # Read SQL file
    sql_path = Path(__file__).parent / "games_features_view.sql"
    sql = read_sql_file(sql_path)
    
    # Execute SQL to create view
    print(f"Creating view in project {config.project_id}, dataset {config.dataset}...")
    try:
        query_job = client.query(sql)
        query_job.result()  # Wait for query to complete
        print("View created successfully!")
    except Exception as e:
        print(f"Error creating view: {e}")
        sys.exit(1)


if __name__ == "__main__":
    create_view()
