"""Script to create the games_features view in BigQuery."""
import os
import sys
import yaml
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import config, but we'll create our own client with available env vars
from src.data.config import BigQueryConfig


def read_sql_file(file_path: str) -> str:
    """Read SQL from a file.
    
    Args:
        file_path: Path to SQL file
        
    Returns:
        SQL content as string
    """
    with open(file_path, "r") as f:
        return f.read()


def load_bigquery_config():
    """Load BigQuery configuration from YAML file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "bigquery.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Get the default environment
        default_env = config.get("default_environment", "dev")
        env_config = config.get("environments", {}).get(default_env, {})
        
        return {
            "project_id": env_config.get("project_id"),
            "dataset": env_config.get("dataset"),
            "location": env_config.get("location")
        }
    except Exception as e:
        print(f"Error loading BigQuery config: {e}")
        return {"project_id": None, "dataset": "bgg_data_dev", "location": "US"}


def create_view():
    """Create the games_features view in BigQuery."""
    # Get environment variables
    env_project_id = os.getenv("GCP_PROJECT_ID")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Load config from YAML
    bq_config = load_bigquery_config()
    
    # Use environment variable for project_id if available, otherwise use config
    project_id = env_project_id or bq_config["project_id"]
    dataset = bq_config["dataset"]
    
    if not project_id:
        print("Error: GCP_PROJECT_ID environment variable not set")
        sys.exit(1)
    
    print(f"Using project: {project_id}")
    print(f"Using dataset: {dataset}")
    
    # Create BigQuery client
    try:
        if credentials_path and os.path.exists(credentials_path):
            print(f"Using credentials from: {credentials_path}")
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            client = bigquery.Client(
                project=project_id,
                credentials=credentials,
            )
        else:
            print("Using default credentials")
            client = bigquery.Client(project=project_id)
    except Exception as e:
        print(f"Error creating BigQuery client: {e}")
        sys.exit(1)
    
    # Read SQL file and replace dataset placeholder
    sql_path = Path(__file__).parent / "games_features_view.sql"
    sql = read_sql_file(sql_path)
    sql = sql.format(dataset=dataset)
    
    # Execute SQL to create view
    print(f"Creating view in project {project_id}, dataset {dataset}...")
    try:
        query_job = client.query(sql)
        query_job.result()  # Wait for query to complete
        print("View created successfully!")
    except Exception as e:
        print(f"Error creating view: {e}")
        sys.exit(1)


if __name__ == "__main__":
    create_view()
