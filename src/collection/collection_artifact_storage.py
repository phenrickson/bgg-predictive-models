"""GCS storage layer for user collection artifacts (models, predictions, analysis).

Path layout per user:
    gs://{bucket}/{env}/collections/{username}/
        metadata.json                           # global user metadata, outcome-agnostic
        collection/latest.parquet               # raw snapshot, outcome-agnostic
        {outcome}/v{N}/
            model.pkl
            threshold.json                      # classification only
            registration.json
            splits/train.parquet
            splits/validation.parquet
            splits/test.parquet
            predictions/predictions.parquet
            predictions/top_recommendations.json
            analysis/summary_stats.json
            analysis/feature_importance.parquet
            analysis/category_affinity.json
"""

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
    organized by username and outcome. Each user+outcome pair gets its own
    versioned prefix:

        gs://{bucket}/{environment}/collections/{username}/
            ├── metadata.json
            ├── collection/latest.parquet
            ├── {outcome}/v{N}/
            │   ├── model.pkl
            │   ├── threshold.json          (classification only)
            │   ├── registration.json
            │   ├── splits/{train,validation,test}.parquet
            │   ├── predictions/{predictions.parquet,top_recommendations.json}
            │   └── analysis/{summary_stats.json,feature_importance.parquet,category_affinity.json}
            └── ...
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

    # --- Internal path / upload / download helpers ---

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

    # --- Version helpers ---

    def latest_version(self, outcome: str) -> Optional[int]:
        """Return the highest version number under {outcome}/, or None if none exist.

        Args:
            outcome: Outcome name (e.g. "own", "love", "rating")

        Returns:
            Highest version number found, or None
        """
        prefix = self._get_blob_path(outcome, "v")
        blobs = self.bucket.list_blobs(prefix=prefix)

        versions = []
        for blob in blobs:
            # Path relative to base_prefix: outcome/v{N}/...
            relative = blob.name[len(self.base_prefix) + 1:]  # strip "base_prefix/"
            parts = relative.split("/")
            # parts[0] = outcome, parts[1] = "v{N}"
            if len(parts) >= 2 and parts[1].startswith("v"):
                try:
                    versions.append(int(parts[1][1:]))
                except ValueError:
                    pass

        return max(versions) if versions else None

    def _next_version(self, outcome: str) -> int:
        """Return the next version number for a given outcome.

        Args:
            outcome: Outcome name

        Returns:
            Next version number (latest + 1, or 1 if no versions exist)
        """
        return (self.latest_version(outcome) or 0) + 1

    # --- Collection Operations (outcome-agnostic) ---

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
        outcome: str,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Save train/val/test splits to GCS under the given outcome + version.

        Args:
            outcome: Outcome name (e.g. "own", "love", "rating")
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            version: Version number; auto-incremented via _next_version if None

        Returns:
            Dictionary with "version" and per-split GCS paths
        """
        if version is None:
            version = self._next_version(outcome)

        paths: Dict[str, Any] = {"version": version}
        for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            blob_path = self._get_blob_path(outcome, f"v{version}", "splits", f"{name}.parquet")
            paths[name] = self._upload_parquet(blob_path, df)
            logger.info(f"Saved {outcome}/v{version} {name} split ({len(df)} rows) to {paths[name]}")
        return paths

    def load_splits(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load splits from GCS.

        Args:
            outcome: Outcome name
            version: Version number; defaults to latest for the outcome

        Returns:
            Dictionary with "version" and train/validation/test DataFrames, or None if not found
        """
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                logger.warning(f"No splits found for outcome '{outcome}'")
                return None

        splits: Dict[str, Any] = {"version": version}
        for name in ["train", "validation", "test"]:
            blob_path = self._get_blob_path(outcome, f"v{version}", "splits", f"{name}.parquet")
            df = self._download_parquet(blob_path)
            if df is None:
                logger.warning(f"Split '{name}' not found for {outcome}/v{version}")
                return None
            splits[name] = df
            logger.info(f"Loaded {outcome}/v{version} {name} split ({len(df)} rows)")
        return splits

    # --- Model Operations ---

    def save_model(
        self,
        outcome: str,
        pipeline: Any,
        metadata: Dict[str, Any],
        threshold: Optional[float] = None,
        version: Optional[int] = None,
    ) -> str:
        """Save model pipeline and metadata to GCS.

        Args:
            outcome: Outcome name (e.g. "own", "love", "rating")
            pipeline: Trained sklearn pipeline
            metadata: Model metadata (metrics, params, etc.)
            threshold: Optimal classification threshold; None for regression outcomes
            version: Model version (auto-incremented if not provided)

        Returns:
            GCS path prefix where model artifacts were saved
        """
        if version is None:
            version = self._next_version(outcome)

        version_prefix = self._get_blob_path(outcome, f"v{version}")

        # Save pipeline
        pipeline_path = f"{version_prefix}/model.pkl"
        self._upload_pickle(pipeline_path, pipeline)

        # Save threshold (classification only)
        if threshold is not None:
            threshold_path = f"{version_prefix}/threshold.json"
            self._upload_json(threshold_path, {"threshold": threshold})

        # Build and save registration metadata
        registration = {
            "username": self.username,
            "outcome": outcome,
            "version": version,
            "created_at": datetime.now().isoformat(),
            **metadata,
        }
        if threshold is not None:
            registration["threshold"] = threshold

        registration_path = f"{version_prefix}/registration.json"
        self._upload_json(registration_path, registration)

        logger.info(f"Saved {outcome} model v{version} to gs://{self.bucket.name}/{version_prefix}/")
        return f"gs://{self.bucket.name}/{version_prefix}/"

    def load_model(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any], Optional[float]]:
        """Load model from GCS.

        Args:
            outcome: Outcome name
            version: Model version (latest if not specified)

        Returns:
            Tuple of (pipeline, metadata, threshold); threshold is None for regression
        """
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                raise ValueError(f"No models found for user '{self.username}', outcome '{outcome}'")

        version_prefix = self._get_blob_path(outcome, f"v{version}")

        # Load pipeline
        pipeline = self._download_pickle(f"{version_prefix}/model.pkl")
        if pipeline is None:
            raise ValueError(
                f"Model not found for user '{self.username}', outcome '{outcome}', version {version}"
            )

        # Load metadata
        metadata = self._download_json(f"{version_prefix}/registration.json") or {}

        # Load threshold (None if not present — regression outcome)
        threshold_data = self._download_json(f"{version_prefix}/threshold.json")
        threshold = threshold_data.get("threshold") if threshold_data else None

        logger.info(f"Loaded {outcome} model v{version} for user '{self.username}'")
        return pipeline, metadata, threshold

    def list_model_versions(self, outcome: str) -> List[Dict[str, Any]]:
        """List all model versions for a given outcome.

        Args:
            outcome: Outcome name

        Returns:
            List of registration metadata dictionaries, sorted by version
        """
        versions = []
        prefix = self._get_blob_path(outcome)
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
        outcome: str,
        version: int,
        predictions_df: pl.DataFrame,
        top_recommendations: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Save predictions and recommendations to GCS.

        Args:
            outcome: Outcome name
            version: Explicit version (predictions belong to a specific model version)
            predictions_df: DataFrame with all game predictions
            top_recommendations: List of top recommended games

        Returns:
            Dictionary mapping artifact names to GCS paths
        """
        paths: Dict[str, str] = {}

        # Save full predictions
        pred_path = self._get_blob_path(outcome, f"v{version}", "predictions", "predictions.parquet")
        paths["predictions"] = self._upload_parquet(pred_path, predictions_df)
        logger.info(f"Saved {outcome}/v{version} predictions ({len(predictions_df)} games)")

        # Save top recommendations
        rec_path = self._get_blob_path(
            outcome, f"v{version}", "predictions", "top_recommendations.json"
        )
        rec_data = {
            "username": self.username,
            "outcome": outcome,
            "version": version,
            "generated_at": datetime.now().isoformat(),
            "total_games_scored": len(predictions_df),
            "recommendations": top_recommendations,
        }
        paths["recommendations"] = self._upload_json(rec_path, rec_data)
        logger.info(f"Saved {outcome}/v{version} top {len(top_recommendations)} recommendations")

        return paths

    def load_predictions(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[pl.DataFrame]:
        """Load predictions from GCS.

        Args:
            outcome: Outcome name
            version: Version number; defaults to latest for the outcome

        Returns:
            Predictions DataFrame or None if not found
        """
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                return None

        blob_path = self._get_blob_path(outcome, f"v{version}", "predictions", "predictions.parquet")
        return self._download_parquet(blob_path)

    def load_recommendations(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load top recommendations from GCS.

        Args:
            outcome: Outcome name
            version: Version number; defaults to latest for the outcome

        Returns:
            Recommendations dictionary or None if not found
        """
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                return None

        blob_path = self._get_blob_path(
            outcome, f"v{version}", "predictions", "top_recommendations.json"
        )
        return self._download_json(blob_path)

    # --- Analysis Operations ---

    def save_analysis_artifacts(
        self,
        outcome: str,
        version: int,
        summary_stats: Dict[str, Any],
        feature_importance: pl.DataFrame,
        category_affinity: Dict[str, Any],
    ) -> Dict[str, str]:
        """Save analysis artifacts to GCS.

        Args:
            outcome: Outcome name
            version: Explicit version (analysis belongs to a specific model version)
            summary_stats: Collection summary statistics
            feature_importance: Feature importance DataFrame
            category_affinity: Category/mechanic preferences

        Returns:
            Dictionary mapping artifact names to GCS paths
        """
        paths: Dict[str, str] = {}

        # Summary stats
        stats_path = self._get_blob_path(outcome, f"v{version}", "analysis", "summary_stats.json")
        paths["summary_stats"] = self._upload_json(stats_path, summary_stats)

        # Feature importance
        fi_path = self._get_blob_path(
            outcome, f"v{version}", "analysis", "feature_importance.parquet"
        )
        paths["feature_importance"] = self._upload_parquet(fi_path, feature_importance)

        # Category affinity
        affinity_path = self._get_blob_path(
            outcome, f"v{version}", "analysis", "category_affinity.json"
        )
        paths["category_affinity"] = self._upload_json(affinity_path, category_affinity)

        logger.info(f"Saved {outcome}/v{version} analysis artifacts for user '{self.username}'")
        return paths

    def load_summary_stats(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load summary statistics from GCS."""
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                return None
        blob_path = self._get_blob_path(outcome, f"v{version}", "analysis", "summary_stats.json")
        return self._download_json(blob_path)

    def load_feature_importance(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[pl.DataFrame]:
        """Load feature importance from GCS."""
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                return None
        blob_path = self._get_blob_path(
            outcome, f"v{version}", "analysis", "feature_importance.parquet"
        )
        return self._download_parquet(blob_path)

    def load_category_affinity(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load category affinity from GCS."""
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                return None
        blob_path = self._get_blob_path(
            outcome, f"v{version}", "analysis", "category_affinity.json"
        )
        return self._download_json(blob_path)

    # --- Metadata Operations (outcome-agnostic) ---

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

    # --- Status ---

    def get_artifact_status(self) -> Dict[str, Any]:
        """Get status of all artifacts for this user.

        Enumerates top-level subdirectories under the user prefix (excluding
        "collection" and "metadata.json") and treats each as an outcome.
        For each outcome, lists its versioned subdirectories.

        Returns:
            Dictionary with per-outcome version info::

                {
                    "username": str,
                    "base_path": str,
                    "collection_exists": bool,
                    "outcomes": {
                        "own": {"latest_version": int | None, "versions": [1, 2, ...]},
                        ...
                    },
                }
        """
        base_path = f"gs://{self.bucket.name}/{self.base_prefix}/"

        # Check whether the raw collection snapshot exists
        collection_blob = self.bucket.blob(self._get_blob_path("collection", "latest.parquet"))
        collection_exists = collection_blob.exists()

        # Discover outcomes by listing all blobs under the user prefix and
        # picking out the top-level directory names that are not reserved names.
        reserved = {"collection", "metadata.json"}
        outcome_versions: Dict[str, set] = {}

        blobs = self.bucket.list_blobs(prefix=self.base_prefix + "/")
        for blob in blobs:
            relative = blob.name[len(self.base_prefix) + 1:]  # strip "base_prefix/"
            parts = relative.split("/")
            if not parts:
                continue
            top = parts[0]
            if top in reserved or top == "":
                continue
            # A valid outcome dir has a v{N} sub-directory
            if len(parts) >= 2 and parts[1].startswith("v"):
                try:
                    ver = int(parts[1][1:])
                except ValueError:
                    continue
                outcome_versions.setdefault(top, set()).add(ver)

        outcomes: Dict[str, Dict[str, Any]] = {}
        for outcome, ver_set in sorted(outcome_versions.items()):
            sorted_versions = sorted(ver_set)
            outcomes[outcome] = {
                "latest_version": max(sorted_versions),
                "versions": sorted_versions,
            }

        return {
            "username": self.username,
            "base_path": base_path,
            "collection_exists": collection_exists,
            "outcomes": outcomes,
        }
