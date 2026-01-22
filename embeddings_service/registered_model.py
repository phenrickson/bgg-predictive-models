"""Manages registered embedding models for production deployment."""

import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.models.experiments import Experiment, ExperimentTracker  # noqa: E402
from src.utils.config import load_config  # noqa: E402

try:
    from .auth import (
        get_authenticated_storage_client,
        AuthenticationError,
    )
except ImportError:
    from auth import get_authenticated_storage_client, AuthenticationError  # noqa: E402


class RegisteredEmbeddingModel(ExperimentTracker):
    """Manages registered embedding models for production deployment."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        project_id: Optional[str] = None,
        base_prefix: str = "models/registered",
    ):
        """Initialize registered embedding model manager.

        Args:
            bucket_name: Google Cloud Storage bucket name. If not provided, uses config.
            project_id: Optional Google Cloud project ID.
            base_prefix: Base prefix for registered model storage.
        """
        # Initialize base tracker with embeddings model type
        super().__init__("embeddings")

        # Get bucket name and environment prefix from config if not provided
        config = load_config()
        if bucket_name is None:
            bucket_name = config.get_bucket_name()

        # Get environment prefix for path construction
        environment_prefix = config.get_environment_prefix()

        # Initialize GCS client
        try:
            self.storage_client = get_authenticated_storage_client(project_id)
            self.bucket = self.storage_client.bucket(bucket_name)
            self.base_prefix = f"{environment_prefix}/{base_prefix}/embeddings"
        except AuthenticationError as e:
            raise ValueError(f"Authentication failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to initialize GCS client: {str(e)}")

        self.STATUSES = {
            "DRAFT": "draft",
            "REGISTERED": "registered",
            "ARCHIVED": "archived",
        }

    def register(
        self,
        experiment: Experiment,
        name: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register an embedding model for production use.

        Args:
            experiment: Experiment to register.
            name: Name for the registered model.
            description: Description of the model.
            metadata: Optional additional metadata.

        Returns:
            Dictionary containing registration details.
        """
        # Load experiment pipeline and info
        pipeline = experiment.load_pipeline()
        model_info = experiment.get_model_info()

        # Get next version number
        existing_versions = self.list_model_versions(name)
        next_version = max([v["version"] for v in existing_versions], default=0) + 1

        # Add environment to metadata
        if metadata is None:
            metadata = {}
        metadata["environment"] = os.getenv("ENVIRONMENT", "unknown")

        # Prepare registration metadata
        registration = {
            "name": name,
            "version": next_version,
            "status": self.STATUSES["REGISTERED"],
            "description": description,
            "original_experiment": {
                "name": experiment.name,
                "metadata": experiment.metadata,
            },
            "model_info": model_info,
            "registered_at": datetime.now().isoformat(),
            "registered_by": metadata.get("registered_by"),
            "environment": metadata["environment"],
            "metadata": metadata,
        }

        # Create version directory in GCS
        version_prefix = f"{self.base_prefix}/{name}/v{next_version}"

        # Save registration metadata
        metadata_blob = self.bucket.blob(f"{version_prefix}/registration.json")
        metadata_blob.upload_from_string(
            json.dumps(registration, indent=2), content_type="application/json"
        )

        # Save model pipeline
        pipeline_blob = self.bucket.blob(f"{version_prefix}/pipeline.pkl")
        pipeline_blob.upload_from_string(
            pickle.dumps(pipeline), content_type="application/octet-stream"
        )

        # Save UMAP model if it exists in the experiment directory
        umap_model_path = experiment.exp_dir / "umap_2d_model.pkl"
        if umap_model_path.exists():
            with open(umap_model_path, "rb") as f:
                umap_model_bytes = f.read()
            umap_blob = self.bucket.blob(f"{version_prefix}/umap_2d_model.pkl")
            umap_blob.upload_from_string(
                umap_model_bytes, content_type="application/octet-stream"
            )
            registration["has_umap_model"] = True
        else:
            registration["has_umap_model"] = False

        # Save PCA model if it exists in the experiment directory
        pca_model_path = experiment.exp_dir / "pca_2d_model.pkl"
        if pca_model_path.exists():
            with open(pca_model_path, "rb") as f:
                pca_model_bytes = f.read()
            pca_blob = self.bucket.blob(f"{version_prefix}/pca_2d_model.pkl")
            pca_blob.upload_from_string(
                pca_model_bytes, content_type="application/octet-stream"
            )
            registration["has_pca_model"] = True
        else:
            registration["has_pca_model"] = False

        return registration

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """List all registered embedding models.

        Returns:
            List of registered model metadata.
        """
        models = []
        blobs = self.bucket.list_blobs(prefix=self.base_prefix)

        for blob in blobs:
            if blob.name.endswith("registration.json"):
                registration = json.loads(blob.download_as_text())
                models.append(registration)

        return sorted(models, key=lambda x: (x["name"], x["version"]))

    def list_model_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a specific registered model.

        Args:
            name: Name of the registered model.

        Returns:
            List of version metadata.
        """
        versions = []
        prefix = f"{self.base_prefix}/{name}"
        blobs = self.bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            if blob.name.endswith("registration.json"):
                registration = json.loads(blob.download_as_text())
                versions.append(registration)

        return sorted(versions, key=lambda x: x["version"])

    def load_registered_model(
        self, name: str, version: Optional[int] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load a registered embedding model pipeline and metadata.

        Args:
            name: Name of the registered model.
            version: Optional specific version (latest if None).

        Returns:
            Tuple of (model pipeline, registration metadata).
        """
        versions = self.list_model_versions(name)
        if not versions:
            raise ValueError(f"No registered model found with name '{name}'")

        if version is None:
            version = max(v["version"] for v in versions)

        if not any(v["version"] == version for v in versions):
            raise ValueError(f"Version {version} not found for model '{name}'")

        version_prefix = f"{self.base_prefix}/{name}/v{version}"
        metadata_blob = self.bucket.blob(f"{version_prefix}/registration.json")
        registration = json.loads(metadata_blob.download_as_text())

        pipeline_blob = self.bucket.blob(f"{version_prefix}/pipeline.pkl")
        pipeline = pickle.loads(pipeline_blob.download_as_string())

        return pipeline, registration

    def load_umap_model(
        self, name: str, version: Optional[int] = None
    ) -> Optional[Any]:
        """Load the UMAP model for a registered embedding model.

        Args:
            name: Name of the registered model.
            version: Optional specific version (latest if None).

        Returns:
            Fitted UMAP model, or None if not available.
        """
        versions = self.list_model_versions(name)
        if not versions:
            return None

        if version is None:
            version = max(v["version"] for v in versions)

        version_prefix = f"{self.base_prefix}/{name}/v{version}"
        umap_blob = self.bucket.blob(f"{version_prefix}/umap_2d_model.pkl")

        if not umap_blob.exists():
            return None

        try:
            umap_model = pickle.loads(umap_blob.download_as_string())
            return umap_model
        except Exception:
            return None

    def load_pca_model(
        self, name: str, version: Optional[int] = None
    ) -> Optional[Any]:
        """Load the PCA model for a registered embedding model.

        Args:
            name: Name of the registered model.
            version: Optional specific version (latest if None).

        Returns:
            Fitted PCA model, or None if not available.
        """
        versions = self.list_model_versions(name)
        if not versions:
            return None

        if version is None:
            version = max(v["version"] for v in versions)

        version_prefix = f"{self.base_prefix}/{name}/v{version}"
        pca_blob = self.bucket.blob(f"{version_prefix}/pca_2d_model.pkl")

        if not pca_blob.exists():
            return None

        try:
            pca_model = pickle.loads(pca_blob.download_as_string())
            return pca_model
        except Exception:
            return None

    def archive_model(
        self, name: str, version: Optional[int] = None, reason: str = ""
    ) -> Dict[str, Any]:
        """Archive a registered model version.

        Args:
            name: Name of the registered model.
            version: Optional specific version (latest if None).
            reason: Reason for archiving.

        Returns:
            Updated registration metadata.
        """
        pipeline, registration = self.load_registered_model(name, version)

        registration["status"] = self.STATUSES["ARCHIVED"]
        registration["archived_at"] = datetime.now().isoformat()
        registration["archive_reason"] = reason

        version = registration["version"]
        version_prefix = f"{self.base_prefix}/{name}/v{version}"
        metadata_blob = self.bucket.blob(f"{version_prefix}/registration.json")
        metadata_blob.upload_from_string(
            json.dumps(registration, indent=2), content_type="application/json"
        )

        return registration
