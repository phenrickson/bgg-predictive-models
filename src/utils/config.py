"""Configuration management for BGG predictive models."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml
from dotenv import load_dotenv
from google.auth import default
from google.cloud import bigquery

# Load environment variables from .env file
load_dotenv()


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


@dataclass
class ModelConfig:
    """Configuration for model settings."""

    type: str
    experiment_name: str
    use_sample_weights: bool = False
    min_ratings: int = 0
    predictions_path: Optional[str] = None


@dataclass
class YearConfig:
    """Configuration for year settings."""

    current: int
    train_end: int
    tune_end: int
    test_start: int
    test_end: int
    eval_start: int
    eval_end: int
    score_start: int
    score_end: int


@dataclass
class Config:
    """Main configuration class."""

    bigquery: BigQueryConfig
    years: YearConfig
    models: Dict[str, ModelConfig]


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file. If not provided,
            defaults to config.yaml in project root

    Returns:
        Complete configuration object

    Raises:
        ValueError: If required config values missing
    """
    if config_path is None:
        # Look for config.yaml in project root
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required sections
    if not all(section in config for section in ["data", "years", "models"]):
        raise ValueError("Missing required config sections: data, years, models")

    # Get project ID from environment variable
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable must be set")

    # Create BigQuery config
    bigquery_config = BigQueryConfig(
        project_id=project_id,
        dataset=config["data"]["bigquery"]["dataset"],
    )

    # Create years config
    years_config = YearConfig(
        current=config["years"]["current"],
        train_end=config["years"]["train_end"],
        tune_end=config["years"]["tune_end"],
        test_start=config["years"]["test_start"],
        test_end=config["years"]["test_end"],
        eval_start=config["years"]["eval"]["start"],
        eval_end=config["years"]["eval"]["end"],
        score_start=config["years"]["score"]["start"],
        score_end=config["years"]["score"]["end"],
    )

    # Create model configs
    model_configs = {}
    for model_name, model_config in config["models"].items():
        model_configs[model_name] = ModelConfig(
            type=model_config["type"],
            experiment_name=model_config["experiment_name"],
            use_sample_weights=model_config.get("use_sample_weights", False),
            min_ratings=model_config.get("min_ratings", 0),
            predictions_path=model_config.get("predictions_path"),
        )

    return Config(
        bigquery=bigquery_config,
        years=years_config,
        models=model_configs,
    )
