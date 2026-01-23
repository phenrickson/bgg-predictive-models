"""Configuration management for BGG predictive models."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
    features_dataset: str = "analytics"
    features_table: str = "games_features"
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
    """Configuration for BigQuery connection."""

    project_id: str
    dataset: str
    table: str
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
class ScoringConfig:
    """Configuration for scoring settings."""

    models: Dict[str, str]  # Maps model type to registered model name
    parameters: Dict[str, float]  # Scoring parameters like prior_rating
    output: Dict[str, str]  # Output settings like predictions_path


@dataclass
class EmbeddingAlgorithmConfig:
    """Configuration for a specific embedding algorithm."""

    pca: Optional[Dict[str, Any]] = None
    svd: Optional[Dict[str, Any]] = None
    umap: Optional[Dict[str, Any]] = None
    autoencoder: Optional[Dict[str, Any]] = None
    vae: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingUploadConfig:
    """Configuration for uploading embeddings to BigQuery (raw table)."""

    dataset: str = "raw"
    table: str = "game_embeddings"


@dataclass
class EmbeddingVectorSearchConfig:
    """Configuration for BigQuery Vector Search (curated table for similarity search)."""

    project: Optional[str] = None  # If None, uses ml_project_id
    dataset: str = "predictions"
    table: str = "bgg_game_embeddings"


@dataclass
class EmbeddingSearchConfig:
    """Configuration for embedding similarity search."""

    default_distance_type: str = "cosine"  # cosine, euclidean, dot_product
    default_top_k: int = 10


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    algorithm: str  # pca, svd, umap, autoencoder
    embedding_dim: int
    experiment_name: str
    algorithms: EmbeddingAlgorithmConfig
    upload: EmbeddingUploadConfig  # Raw table for writing new embeddings
    vector_search: EmbeddingVectorSearchConfig  # Curated table for similarity search
    search: EmbeddingSearchConfig
    min_ratings: int = 25  # Minimum users_rated for training data

    def get_algorithm_params(self, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Get parameters for a specific algorithm.

        Args:
            algorithm: Algorithm name. If None, uses self.algorithm.

        Returns:
            Dictionary of algorithm parameters.
        """
        algo = algorithm or self.algorithm
        params = getattr(self.algorithms, algo, None)
        return params if params is not None else {}


@dataclass
class TextEmbeddingAlgorithmConfig:
    """Configuration for text embedding algorithms."""

    pmi: Optional[Dict[str, Any]] = None
    word2vec: Optional[Dict[str, Any]] = None


@dataclass
class TextEmbeddingUploadConfig:
    """Configuration for uploading text embeddings to BigQuery."""

    dataset: str = "raw"
    table: str = "description_embeddings"


@dataclass
class TextEmbeddingConfig:
    """Configuration for text embeddings from descriptions."""

    algorithm: str  # pmi, word2vec
    embedding_dim: int
    experiment_name: str
    document_method: str  # mean, tfidf, sif
    algorithms: TextEmbeddingAlgorithmConfig
    upload: Optional[TextEmbeddingUploadConfig] = None

    def get_algorithm_params(self, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Get parameters for a specific algorithm."""
        algo = algorithm or self.algorithm
        params = getattr(self.algorithms, algo, None)
        return params if params is not None else {}


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

    bucket_name: str
    default_environment: str
    years: YearConfig
    models: Dict[str, ModelConfig]
    data_warehouse: DataWarehouseConfig
    predictions: PredictionsDestinationConfig
    ml_project_id: str
    scoring: Optional[ScoringConfig] = None
    embeddings: Optional[EmbeddingConfig] = None
    text_embeddings: Optional[TextEmbeddingConfig] = None

    def get_current_environment(self) -> str:
        """Get the current environment name based on ENVIRONMENT variable or default."""
        return os.getenv("ENVIRONMENT", self.default_environment)

    def get_bucket_name(self) -> str:
        """Get the bucket name (single bucket for all environments)."""
        return self.bucket_name

    def get_environment_prefix(self) -> str:
        """Get the environment prefix for GCS paths (e.g., 'dev', 'prod')."""
        return self.get_current_environment()

    def get_gcs_path(self, *path_parts: str) -> str:
        """Get a GCS path with environment prefix.

        Example: get_gcs_path("models", "hurdle-v2025") returns
        "gs://bgg-predictive-models/dev/models/hurdle-v2025"
        """
        env = self.get_environment_prefix()
        path = "/".join([env] + list(path_parts))
        return f"gs://{self.bucket_name}/{path}"

    def get_data_warehouse_config(self) -> DataWarehouseConfig:
        """Get the data warehouse configuration for reading data."""
        return self.data_warehouse

    def get_predictions_destination(self) -> PredictionsDestinationConfig:
        """Get the predictions destination configuration."""
        return self.predictions

    def get_bigquery_config(self) -> BigQueryConfig:
        """Get BigQuery configuration for data warehouse access."""
        return BigQueryConfig(
            project_id=self.data_warehouse.project_id,
            dataset=self.data_warehouse.features_dataset,
            table=self.data_warehouse.features_table,
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
    required_sections = ["ml_project", "years", "models", "data_warehouse", "predictions"]
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
        features_dataset=dw_config.get("features_dataset", "analytics"),
        features_table=dw_config.get("features_table", "games_features"),
        datasets=dw_config.get("datasets"),
    )

    # Create predictions destination config
    pred_config = config["predictions"]
    predictions_config = PredictionsDestinationConfig(
        project_id=pred_config.get("project_id", data_warehouse_project_id),
        dataset=pred_config.get("dataset", "raw"),
        table=pred_config.get("table", "ml_predictions_landing"),
    )

    # Get bucket name from ml_project config
    ml_project_config = config["ml_project"]
    bucket_name = ml_project_config.get("bucket_name", "bgg-predictive-models")

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

    # Create embeddings config if present
    embeddings_config = None
    if "embeddings" in config:
        emb = config["embeddings"]
        algorithms_config = EmbeddingAlgorithmConfig(
            pca=emb.get("algorithms", {}).get("pca"),
            svd=emb.get("algorithms", {}).get("svd"),
            umap=emb.get("algorithms", {}).get("umap"),
            autoencoder=emb.get("algorithms", {}).get("autoencoder"),
            vae=emb.get("algorithms", {}).get("vae"),
        )
        upload_config = EmbeddingUploadConfig(
            dataset=emb.get("upload", {}).get("dataset", "raw"),
            table=emb.get("upload", {}).get("table", "game_embeddings"),
        )
        vector_search_config = EmbeddingVectorSearchConfig(
            project=emb.get("vector_search", {}).get("project"),
            dataset=emb.get("vector_search", {}).get("dataset", "predictions"),
            table=emb.get("vector_search", {}).get("table", "bgg_game_embeddings"),
        )
        search_config = EmbeddingSearchConfig(
            default_distance_type=emb.get("search", {}).get("default_distance_type", "cosine"),
            default_top_k=emb.get("search", {}).get("default_top_k", 10),
        )
        embeddings_config = EmbeddingConfig(
            algorithm=emb["algorithm"],
            embedding_dim=emb["embedding_dim"],
            experiment_name=emb["experiment_name"],
            algorithms=algorithms_config,
            upload=upload_config,
            vector_search=vector_search_config,
            search=search_config,
            min_ratings=emb.get("min_ratings", 25),
        )

    # Create text embeddings config if present
    text_embeddings_config = None
    if "text_embeddings" in config:
        te = config["text_embeddings"]
        te_algorithms_config = TextEmbeddingAlgorithmConfig(
            pmi=te.get("algorithms", {}).get("pmi"),
            word2vec=te.get("algorithms", {}).get("word2vec"),
        )
        te_upload_config = None
        if "upload" in te:
            te_upload_config = TextEmbeddingUploadConfig(
                dataset=te["upload"].get("dataset", "raw"),
                table=te["upload"].get("table", "description_embeddings"),
            )
        text_embeddings_config = TextEmbeddingConfig(
            algorithm=te.get("algorithm", "pmi"),
            embedding_dim=te.get("embedding_dim", 100),
            experiment_name=te.get("experiment_name", "text-embeddings"),
            document_method=te.get("document_method", "mean"),
            algorithms=te_algorithms_config,
            upload=te_upload_config,
        )

    return Config(
        bucket_name=bucket_name,
        default_environment=config.get("default_environment", "dev"),
        years=years_config,
        models=model_configs,
        data_warehouse=data_warehouse_config,
        predictions=predictions_config,
        ml_project_id=ml_project_id,
        scoring=scoring_config,
        embeddings=embeddings_config,
        text_embeddings=text_embeddings_config,
    )
