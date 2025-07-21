"""Configuration for data loading and database connections."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
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
        try:
            if self.credentials_path:
                if not os.path.exists(self.credentials_path):
                    raise ValueError(
                        f"Credentials file not found at: {self.credentials_path}"
                    )

                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

                return bigquery.Client(
                    project=self.project_id,
                    credentials=credentials,
                )

            # Try default credentials
            return bigquery.Client(project=self.project_id)

        except Exception as e:
            raise


def load_config(config_path: Optional[str] = None) -> BigQueryConfig:
    """Load BigQuery configuration from YAML file.

    Args:
        config_path: Path to config YAML file. If not provided,
            defaults to src/data/config.yaml

    Returns:
        BigQuery configuration

    Raises:
        ValueError: If required config values missing
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "bigquery" not in config:
        raise ValueError("Missing bigquery section in config")

    bq_config = config["bigquery"]
    if "dataset" not in bq_config:
        raise ValueError("Missing dataset in bigquery config")

    # Get project ID from environment or default
    project_id = os.getenv("GCP_PROJECT_ID", "gcp-demos-411520")

    # Explicitly set credentials path from .env
    credentials_path = os.path.join(
        Path(__file__).parent.parent.parent, "credentials", "service-account-key.json"
    )

    return BigQueryConfig(
        project_id=project_id,
        dataset=bq_config["dataset"],
        credentials_path=credentials_path,
    )
