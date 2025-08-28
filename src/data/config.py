"""Configuration for data loading and database connections."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from google.auth import default
from google.cloud import bigquery


@dataclass
class BigQueryConfig:
    """Configuration for BigQuery connection."""

    project_id: str
    dataset: str
    credentials_path: Optional[str] = None

    def get_client(self) -> bigquery.Client:
        """Get authenticated BigQuery client using Google Application Default Credentials.

        Uses google.auth.default() to automatically discover credentials in the following order:
        1. Service account key file specified in GOOGLE_APPLICATION_CREDENTIALS environment variable
        2. Google Cloud SDK credentials (for local development)
        3. Compute Engine/Cloud Run service account (for cloud deployment)

        Returns:
            Authenticated BigQuery client

        Raises:
            Exception: If credentials cannot be found or authenticated
        """
        try:
            # Get credentials using google.auth.default()
            credentials, _ = default()

            # Create BigQuery client with explicit credentials and project
            return bigquery.Client(credentials=credentials, project=self.project_id)

        except Exception:
            raise


def load_config(config_path: Optional[str] = None) -> BigQueryConfig:
    """Load BigQuery configuration from YAML file.

    Uses Google Application Default Credentials (ADC) for authentication.
    Project ID is read from GCP_PROJECT_ID environment variable.

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

    # Get project ID from environment variable
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable must be set")

    return BigQueryConfig(
        project_id=project_id,
        dataset=bq_config["dataset"],
        credentials_path=None,  # Always use ADC
    )
