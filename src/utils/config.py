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
class DataWarehouseConfig:
    """Configuration for reading from bgg-data-warehouse."""

    project_id: str
    location: str = "US"
    features_dataset: str = "core"
    features_table: str = "games_features_materialized"
    datasets: Optional[Dict[str, str]] = None

    def get_client(self) -> bigquery.Client:
        """Get authenticated BigQuery client for data warehouse."""
        credentials, _ = default()
        return bigquery.Client(credentials=credentials, project=self.project_id)


@dataclass
class PredictionsDestinationConfig:
    """Configuration for writing predictions to data warehouse landing table."""

    project_id: str
    dataset: str
    table: str

    def get_client(self) -> bigquery.Client:
        """Get authenticated BigQuery client for predictions destination."""
        credentials, _ = default()
        return bigquery.Client(credentials=credentials, project=self.project_id)

    def get_table_id(self) -> str:
        """Get fully qualified table ID."""
        return f"{self.project_id}.{self.dataset}.{self.table}"


@dataclass
class BigQueryConfig:
    """Configuration for BigQuery connection (legacy, for backwards compatibility)."""

    project_id: str
    dataset: str
    credentials_path: Optional[str] = None
    location: str = "US"
    datasets: Optional[Dict[str, str]] = None

    def get_client(self) -> bigquery.Client:
        """Get authenticated BigQuery client using Google Application Default Credentials."""
        try:
            credentials, _ = default()
            return bigquery.Client(credentials=credentials, project=self.project_id)
        except Exception:
            raise


@dataclass
class EnvironmentConfig:
    """Configuration for environment-specific ML artifact settings."""

    bucket_name: str


@dataclass
class ScoringConfig:
    """Configuration for scoring settings."""

    models: Dict[str, str]  # Maps model type to registered model name
    parameters: Dict[str, float]  # Scoring parameters like prior_rating
    output: Dict[str, str]  # Output settings like predictions_path


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

    environment: Dict[str, EnvironmentConfig]
    default_environment: str
    years: YearConfig
    models: Dict[str, ModelConfig]
    data_warehouse: DataWarehouseConfig
    predictions: PredictionsDestinationConfig
    ml_project_id: str
    scoring: Optional[ScoringConfig] = None

    def get_current_environment(self) -> str:
        """Get the current environment name based on ENVIRONMENT variable or default."""
        return os.getenv("ENVIRONMENT", self.default_environment)

    def get_bucket_name(self) -> str:
        """Get the bucket name for the current environment."""
        env_name = self.get_current_environment()
        if env_name not in self.environment:
            raise ValueError(f"Unknown environment: {env_name}")
        return self.environment[env_name].bucket_name

    def get_data_warehouse_config(self) -> DataWarehouseConfig:
        """Get the data warehouse configuration for reading data."""
        return self.data_warehouse

    def get_predictions_destination(self) -> PredictionsDestinationConfig:
        """Get the predictions destination configuration."""
        return self.predictions

    def get_bigquery_config(self) -> BigQueryConfig:
        """Get BigQuery configuration (legacy, maps to data warehouse).

        This method is kept for backwards compatibility with existing code.
        New code should use get_data_warehouse_config() instead.
        """
        return BigQueryConfig(
            project_id=self.data_warehouse.project_id,
            dataset=self.data_warehouse.features_dataset,
            location=self.data_warehouse.location,
        )


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
    required_sections = ["environment", "years", "models", "data_warehouse", "predictions"]
    if not all(section in config for section in required_sections):
        raise ValueError(f"Missing required config sections: {required_sections}")

    # Get project IDs from environment variables
    # Support both new env vars and legacy GCP_PROJECT_ID for backwards compatibility
    data_warehouse_project_id = os.getenv(
        "DATA_WAREHOUSE_PROJECT_ID",
        os.getenv("GCP_PROJECT_ID"),  # Fallback for backwards compatibility
    )
    ml_project_id = os.getenv(
        "ML_PROJECT_ID",
        os.getenv("GCP_PROJECT_ID"),  # Fallback for backwards compatibility
    )

    if not data_warehouse_project_id:
        raise ValueError(
            "DATA_WAREHOUSE_PROJECT_ID (or GCP_PROJECT_ID) environment variable must be set"
        )
    if not ml_project_id:
        raise ValueError(
            "ML_PROJECT_ID (or GCP_PROJECT_ID) environment variable must be set"
        )

    # Create data warehouse config
    dw_config = config["data_warehouse"]
    data_warehouse_config = DataWarehouseConfig(
        project_id=dw_config.get("project_id", data_warehouse_project_id),
        location=dw_config.get("location", "US"),
        features_dataset=dw_config.get("features_dataset", "core"),
        features_table=dw_config.get("features_table", "games_features_materialized"),
        datasets=dw_config.get("datasets"),
    )

    # Create predictions destination config
    pred_config = config["predictions"]
    predictions_config = PredictionsDestinationConfig(
        project_id=pred_config.get("project_id", data_warehouse_project_id),
        dataset=pred_config.get("dataset", "raw"),
        table=pred_config.get("table", "ml_predictions_landing"),
    )

    # Create environment configs (for ML artifacts - buckets only)
    environment_configs = {}
    for env_name, env_config in config["environment"].items():
        environment_configs[env_name] = EnvironmentConfig(
            bucket_name=env_config["bucket_name"],
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

    # Create scoring config if present
    scoring_config = None
    if "scoring" in config:
        scoring_config = ScoringConfig(
            models=config["scoring"]["models"],
            parameters=config["scoring"]["parameters"],
            output=config["scoring"]["output"],
        )

    return Config(
        environment=environment_configs,
        default_environment=config.get("default_environment", "dev"),
        years=years_config,
        models=model_configs,
        data_warehouse=data_warehouse_config,
        predictions=predictions_config,
        ml_project_id=ml_project_id,
        scoring=scoring_config,
    )
