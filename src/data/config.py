"""Configuration for data loading and database connections."""
import os
from dataclasses import dataclass
from typing import Optional

from google.cloud import bigquery
from google.oauth2 import service_account


@dataclass
class BigQueryConfig:
    """Configuration for BigQuery connection."""
    
    project_id: str
    dataset: str
    credentials_path: Optional[str] = None
    
    def get_client(self) -> bigquery.Client:
        """Get authenticated BigQuery client.
        
        Returns:
            Authenticated BigQuery client
            
        Raises:
            ValueError: If credentials not found
        """
        if self.credentials_path:
            if not os.path.exists(self.credentials_path):
                raise ValueError(
                    f"Credentials file not found at: {self.credentials_path}"
                )
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            return bigquery.Client(
                project=self.project_id,
                credentials=credentials,
            )
        
        # Try default credentials
        return bigquery.Client(project=self.project_id)


def get_config_from_env() -> BigQueryConfig:
    """Get BigQuery config from environment variables.
    
    Required env vars:
    - BGG_PROJECT_ID: GCP project ID
    - BGG_DATASET: BigQuery dataset name
    - BGG_CREDENTIALS_PATH: Path to service account key (optional)
    
    Returns:
        BigQuery configuration
        
    Raises:
        ValueError: If required env vars missing
    """
    project_id = os.getenv("BGG_PROJECT_ID")
    dataset = os.getenv("BGG_DATASET")
    
    if not project_id or not dataset:
        raise ValueError(
            "Required environment variables not set. "
            "Please set BGG_PROJECT_ID and BGG_DATASET."
        )
    
    return BigQueryConfig(
        project_id=project_id,
        dataset=dataset,
        credentials_path=os.getenv("BGG_CREDENTIALS_PATH"),
    )
