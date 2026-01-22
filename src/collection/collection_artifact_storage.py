"""GCS storage layer for user collection artifacts (models, predictions, analysis)."""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from google.cloud import storage

from src.utils.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class ArtifactStorageConfig:
    """Configuration for collection artifact storage."""

    bucket_name: Optional[str] = None
    environment: Optional[str] = None
    base_prefix: str = "collections"


class CollectionArtifactStorage:
    """Handles GCS storage for user collection artifacts.

    Stores model pipelines, predictions, and analysis artifacts in GCS buckets
    organized by username. Each user gets their own prefix in the bucket:

        gs://{bucket}/{environment}/collections/{username}/
            ├── metadata.json
            ├── collection/latest.parquet
            ├── splits/{train,validation,test}.parquet
            ├── models/ownership/v{version}/
            ├── predictions/latest/
            └── analysis/
    """

    def __init__(
        self,
        username: str,
        config: Optional[ArtifactStorageConfig] = None,
    ):
        """Initialize artifact storage for a specific user.

        Args:
            username: BGG username
            config: Optional storage configuration
        """
        self.username = username
        self.config = config or ArtifactStorageConfig()

        # Load project config for defaults
        project_config = load_config()

        # Get bucket name from config or project config
        bucket_name = self.config.bucket_name or project_config.get_bucket_name()

        # Get environment prefix
        environment = self.config.environment or project_config.get_environment_prefix()

        # Initialize GCS client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

        # Build base prefix for this user
        self.base_prefix = f"{environment}/{self.config.base_prefix}/{username}"

        logger.info(f"Initialized artifact storage for user '{username}'")
        logger.info(f"GCS path: gs://{bucket_name}/{self.base_prefix}/")

    def _get_blob_path(self, *parts: str) -> str:
        """Build a blob path from parts."""
        return "/".join([self.base_prefix] + list(parts))

    def _upload_json(self, blob_path: str, data: Dict[str, Any]) -> str:
        """Upload JSON data to GCS."""
        blob = self.bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(data, indent=2, default=str), content_type="application/json"
        )
        return f"gs://{self.bucket.name}/{blob_path}"

    def _download_json(self, blob_path: str) -> Optional[Dict[str, Any]]:
        """Download JSON data from GCS."""
        blob = self.bucket.blob(blob_path)
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())

    def _upload_parquet(self, blob_path: str, df: pl.DataFrame) -> str:
        """Upload Polars DataFrame as parquet to GCS."""
        blob = self.bucket.blob(blob_path)
        buffer = BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type="application/octet-stream")
        return f"gs://{self.bucket.name}/{blob_path}"

    def _download_parquet(self, blob_path: str) -> Optional[pl.DataFrame]:
        """Download parquet file from GCS as Polars DataFrame."""
        blob = self.bucket.blob(blob_path)
        if not blob.exists():
            return None
        buffer = BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        return pl.read_parquet(buffer)

    def _upload_pickle(self, blob_path: str, obj: Any) -> str:
        """Upload pickled object to GCS."""
        blob = self.bucket.blob(blob_path)
        blob.upload_from_string(
            pickle.dumps(obj), content_type="application/octet-stream"
        )
        return f"gs://{self.bucket.name}/{blob_path}"

    def _download_pickle(self, blob_path: str) -> Optional[Any]:
        """Download pickled object from GCS."""
        blob = self.bucket.blob(blob_path)
        if not blob.exists():
            return None
        return pickle.loads(blob.download_as_bytes())

    # --- Collection Operations ---

    def save_collection(self, collection_df: pl.DataFrame) -> str:
        """Save collection snapshot to GCS.

        Args:
            collection_df: Collection DataFrame with features

        Returns:
            GCS path where collection was saved
        """
        blob_path = self._get_blob_path("collection", "latest.parquet")
        path = self._upload_parquet(blob_path, collection_df)
        logger.info(f"Saved collection ({len(collection_df)} games) to {path}")
        return path

    def load_collection(self) -> Optional[pl.DataFrame]:
        """Load latest collection from GCS.

        Returns:
            Collection DataFrame or None if not found
        """
        blob_path = self._get_blob_path("collection", "latest.parquet")
        df = self._download_parquet(blob_path)
        if df is not None:
            logger.info(f"Loaded collection ({len(df)} games) from GCS")
        return df

    # --- Split Operations ---

    def save_splits(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> Dict[str, str]:
        """Save train/val/test splits to GCS.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame

        Returns:
            Dictionary mapping split names to GCS paths
        """
        paths = {}
        for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            blob_path = self._get_blob_path("splits", f"{name}.parquet")
            paths[name] = self._upload_parquet(blob_path, df)
            logger.info(f"Saved {name} split ({len(df)} rows) to {paths[name]}")
        return paths

    def load_splits(self) -> Optional[Dict[str, pl.DataFrame]]:
        """Load splits from GCS.

        Returns:
            Dictionary with train/validation/test DataFrames or None if not found
        """
        splits = {}
        for name in ["train", "validation", "test"]:
            blob_path = self._get_blob_path("splits", f"{name}.parquet")
            df = self._download_parquet(blob_path)
            if df is None:
                logger.warning(f"Split '{name}' not found in GCS")
                return None
            splits[name] = df
            logger.info(f"Loaded {name} split ({len(df)} rows)")
        return splits

    # --- Model Operations ---

    def save_model(
        self,
        pipeline: Any,
        metadata: Dict[str, Any],
        threshold: float,
        version: Optional[int] = None,
    ) -> str:
        """Save model pipeline and metadata to GCS.

        Args:
            pipeline: Trained sklearn pipeline
            metadata: Model metadata (metrics, params, etc.)
            threshold: Optimal classification threshold
            version: Model version (auto-incremented if not provided)

        Returns:
            GCS path where model was saved
        """
        # Auto-increment version if not provided
        if version is None:
            existing_versions = self.list_model_versions()
            version = max([v["version"] for v in existing_versions], default=0) + 1

        version_prefix = self._get_blob_path("models", "ownership", f"v{version}")

        # Save pipeline
        pipeline_path = f"{version_prefix}/pipeline.pkl"
        self._upload_pickle(pipeline_path, pipeline)

        # Save threshold
        threshold_path = f"{version_prefix}/threshold.json"
        self._upload_json(threshold_path, {"threshold": threshold})

        # Build registration metadata
        registration = {
            "username": self.username,
            "version": version,
            "threshold": threshold,
            "created_at": datetime.now().isoformat(),
            **metadata,
        }

        # Save registration
        registration_path = f"{version_prefix}/registration.json"
        self._upload_json(registration_path, registration)

        logger.info(f"Saved model v{version} to gs://{self.bucket.name}/{version_prefix}/")
        return f"gs://{self.bucket.name}/{version_prefix}/"

    def load_model(
        self, version: Optional[int] = None
    ) -> Tuple[Any, Dict[str, Any], float]:
        """Load model from GCS.

        Args:
            version: Model version (latest if not specified)

        Returns:
            Tuple of (pipeline, metadata, threshold)
        """
        # Get latest version if not specified
        if version is None:
            versions = self.list_model_versions()
            if not versions:
                raise ValueError(f"No models found for user '{self.username}'")
            version = max(v["version"] for v in versions)

        version_prefix = self._get_blob_path("models", "ownership", f"v{version}")

        # Load pipeline
        pipeline = self._download_pickle(f"{version_prefix}/pipeline.pkl")
        if pipeline is None:
            raise ValueError(f"Model v{version} not found for user '{self.username}'")

        # Load metadata
        metadata = self._download_json(f"{version_prefix}/registration.json") or {}

        # Load threshold
        threshold_data = self._download_json(f"{version_prefix}/threshold.json") or {}
        threshold = threshold_data.get("threshold", 0.5)

        logger.info(f"Loaded model v{version} for user '{self.username}'")
        return pipeline, metadata, threshold

    def list_model_versions(self) -> List[Dict[str, Any]]:
        """List all model versions for this user.

        Returns:
            List of version metadata dictionaries
        """
        versions = []
        prefix = self._get_blob_path("models", "ownership")
        blobs = self.bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            if blob.name.endswith("registration.json"):
                try:
                    registration = json.loads(blob.download_as_text())
                    versions.append(registration)
                except Exception as e:
                    logger.warning(f"Error loading registration from {blob.name}: {e}")

        return sorted(versions, key=lambda x: x.get("version", 0))

    # --- Predictions Operations ---

    def save_predictions(
        self,
        predictions_df: pl.DataFrame,
        top_recommendations: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Save predictions and recommendations to GCS.

        Args:
            predictions_df: DataFrame with all game predictions
            top_recommendations: List of top recommended games

        Returns:
            Dictionary mapping artifact names to GCS paths
        """
        paths = {}

        # Save full predictions
        pred_path = self._get_blob_path("predictions", "latest", "predictions.parquet")
        paths["predictions"] = self._upload_parquet(pred_path, predictions_df)
        logger.info(f"Saved predictions ({len(predictions_df)} games)")

        # Save top recommendations
        rec_path = self._get_blob_path("predictions", "latest", "top_recommendations.json")
        rec_data = {
            "username": self.username,
            "generated_at": datetime.now().isoformat(),
            "total_games_scored": len(predictions_df),
            "recommendations": top_recommendations,
        }
        paths["recommendations"] = self._upload_json(rec_path, rec_data)
        logger.info(f"Saved top {len(top_recommendations)} recommendations")

        return paths

    def load_predictions(self) -> Optional[pl.DataFrame]:
        """Load latest predictions from GCS.

        Returns:
            Predictions DataFrame or None if not found
        """
        blob_path = self._get_blob_path("predictions", "latest", "predictions.parquet")
        return self._download_parquet(blob_path)

    def load_recommendations(self) -> Optional[Dict[str, Any]]:
        """Load top recommendations from GCS.

        Returns:
            Recommendations dictionary or None if not found
        """
        blob_path = self._get_blob_path("predictions", "latest", "top_recommendations.json")
        return self._download_json(blob_path)

    # --- Analysis Operations ---

    def save_analysis_artifacts(
        self,
        summary_stats: Dict[str, Any],
        feature_importance: pl.DataFrame,
        category_affinity: Dict[str, Any],
    ) -> Dict[str, str]:
        """Save analysis artifacts to GCS.

        Args:
            summary_stats: Collection summary statistics
            feature_importance: Feature importance DataFrame
            category_affinity: Category/mechanic preferences

        Returns:
            Dictionary mapping artifact names to GCS paths
        """
        paths = {}

        # Summary stats
        stats_path = self._get_blob_path("analysis", "summary_stats.json")
        paths["summary_stats"] = self._upload_json(stats_path, summary_stats)

        # Feature importance
        fi_path = self._get_blob_path("analysis", "feature_importance.parquet")
        paths["feature_importance"] = self._upload_parquet(fi_path, feature_importance)

        # Category affinity
        affinity_path = self._get_blob_path("analysis", "category_affinity.json")
        paths["category_affinity"] = self._upload_json(affinity_path, category_affinity)

        logger.info(f"Saved analysis artifacts for user '{self.username}'")
        return paths

    def load_summary_stats(self) -> Optional[Dict[str, Any]]:
        """Load summary statistics from GCS."""
        blob_path = self._get_blob_path("analysis", "summary_stats.json")
        return self._download_json(blob_path)

    def load_feature_importance(self) -> Optional[pl.DataFrame]:
        """Load feature importance from GCS."""
        blob_path = self._get_blob_path("analysis", "feature_importance.parquet")
        return self._download_parquet(blob_path)

    def load_category_affinity(self) -> Optional[Dict[str, Any]]:
        """Load category affinity from GCS."""
        blob_path = self._get_blob_path("analysis", "category_affinity.json")
        return self._download_json(blob_path)

    # --- Metadata Operations ---

    def save_user_metadata(self, metadata: Dict[str, Any]) -> str:
        """Save user metadata to GCS.

        Args:
            metadata: User metadata dictionary

        Returns:
            GCS path where metadata was saved
        """
        blob_path = self._get_blob_path("metadata.json")
        metadata["username"] = self.username
        metadata["updated_at"] = datetime.now().isoformat()
        return self._upload_json(blob_path, metadata)

    def load_user_metadata(self) -> Optional[Dict[str, Any]]:
        """Load user metadata from GCS.

        Returns:
            User metadata dictionary or None if not found
        """
        blob_path = self._get_blob_path("metadata.json")
        return self._download_json(blob_path)

    def get_artifact_status(self) -> Dict[str, Any]:
        """Get status of all artifacts for this user.

        Returns:
            Dictionary with artifact existence and metadata
        """
        status = {
            "username": self.username,
            "base_path": f"gs://{self.bucket.name}/{self.base_prefix}/",
            "artifacts": {},
        }

        # Check each artifact type
        checks = [
            ("metadata", self._get_blob_path("metadata.json")),
            ("collection", self._get_blob_path("collection", "latest.parquet")),
            ("train_split", self._get_blob_path("splits", "train.parquet")),
            ("validation_split", self._get_blob_path("splits", "validation.parquet")),
            ("test_split", self._get_blob_path("splits", "test.parquet")),
            ("predictions", self._get_blob_path("predictions", "latest", "predictions.parquet")),
            ("recommendations", self._get_blob_path("predictions", "latest", "top_recommendations.json")),
            ("summary_stats", self._get_blob_path("analysis", "summary_stats.json")),
        ]

        for name, path in checks:
            blob = self.bucket.blob(path)
            status["artifacts"][name] = {
                "exists": blob.exists(),
                "path": f"gs://{self.bucket.name}/{path}",
            }

        # Check model versions
        model_versions = self.list_model_versions()
        status["model_versions"] = len(model_versions)
        if model_versions:
            status["latest_model_version"] = max(v["version"] for v in model_versions)

        return status
