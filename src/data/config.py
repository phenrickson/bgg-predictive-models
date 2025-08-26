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

        except Exception:
            raise


def load_config(
    config_path: Optional[str] = None, use_service_account: Optional[bool] = None
) -> BigQueryConfig:
    """Load BigQuery configuration from YAML file.

    Args:
        config_path: Path to config YAML file. If not provided,
            defaults to src/data/config.yaml
        use_service_account: If True, use service account file. If False, use default credentials.
            If None, auto-detect based on environment and file existence.

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

    # Determine credentials approach
    credentials_path = None

    if use_service_account is True:
        # Explicitly requested service account
        credentials_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "credentials",
            "service-account-key.json",
        )
    elif use_service_account is False:
        # Explicitly requested default credentials
        credentials_path = None
    else:
        # Auto-detect: only use service account if file exists AND not in Cloud Run
        potential_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "credentials",
            "service-account-key.json",
        )

        # Check if we're in Cloud Run (common environment variables)
        in_cloud_run = (
            os.getenv("K_SERVICE") is not None  # Cloud Run service name
            or os.getenv("GOOGLE_CLOUD_PROJECT") is not None  # GCP project in Cloud Run
            or os.getenv("GAE_ENV") is not None  # App Engine (similar environment)
        )

        if not in_cloud_run and os.path.exists(potential_path):
            credentials_path = potential_path
        # Otherwise, credentials_path stays None (use default credentials)

    return BigQueryConfig(
        project_id=project_id,
        dataset=bq_config["dataset"],
        credentials_path=credentials_path,
    )
