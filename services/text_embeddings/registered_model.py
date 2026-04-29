"""Manages registered text embedding models for production deployment."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
import sys

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


class RegisteredTextEmbeddingModel(ExperimentTracker):
    """Manages registered text embedding models for production deployment."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        project_id: Optional[str] = None,
        base_prefix: str = "models/registered",
    ):
        """Initialize registered text embedding model manager.

        Args:
            bucket_name: Google Cloud Storage bucket name. If not provided, uses config.
            project_id: Optional Google Cloud project ID.
            base_prefix: Base prefix for registered model storage.
        """
        # Initialize base tracker with text_embeddings model type
        super().__init__("text_embeddings")

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
            self.base_prefix = f"{environment_prefix}/{base_prefix}/text_embeddings"
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
        """Register a text embedding model for production use.

        Args:
            experiment: Experiment to register.
            name: Name for the registered model.
            description: Description of the model.
            metadata: Optional additional metadata.

        Returns:
            Dictionary containing registration details.
        """
        # Get next version number
        existing_versions = self.list_model_versions(name)
        next_version = max([v["version"] for v in existing_versions], default=0) + 1

        # Add environment to metadata
        if metadata is None:
            metadata = {}
        metadata["environment"] = os.getenv("ENVIRONMENT", "unknown")

        # Load experiment metadata
        exp_metadata = experiment.metadata

        # Prepare registration metadata
        registration = {
            "name": name,
            "version": next_version,
            "status": self.STATUSES["REGISTERED"],
            "description": description,
            "original_experiment": {
                "name": experiment.name,
                "metadata": exp_metadata,
            },
            "model_info": {
                "algorithm": exp_metadata.get("algorithm"),
                "embedding_dim": exp_metadata.get("embedding_dim"),
                "document_method": exp_metadata.get("document_method"),
                "vocab_size": exp_metadata.get("vocab_size"),
                "n_documents": exp_metadata.get("n_documents"),
            },
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

        # Save word embedding model
        word_model_path = experiment.exp_dir / "word_embedding.pkl"
        if word_model_path.exists():
            with open(word_model_path, "rb") as f:
                word_model_bytes = f.read()
            word_blob = self.bucket.blob(f"{version_prefix}/word_embedding.pkl")
            word_blob.upload_from_string(
                word_model_bytes, content_type="application/octet-stream"
            )
            registration["has_word_model"] = True
        else:
            registration["has_word_model"] = False

        # Save document embedding model
        doc_model_path = experiment.exp_dir / "document_embedding.pkl"
        if doc_model_path.exists():
            with open(doc_model_path, "rb") as f:
                doc_model_bytes = f.read()
            doc_blob = self.bucket.blob(f"{version_prefix}/document_embedding.pkl")
            doc_blob.upload_from_string(
                doc_model_bytes, content_type="application/octet-stream"
            )
            registration["has_document_model"] = True
        else:
            registration["has_document_model"] = False

        return registration

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """List all registered text embedding models.

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
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load a registered text embedding model.

        Args:
            name: Name of the registered model.
            version: Optional specific version (latest if None).

        Returns:
            Tuple of (word_model, doc_model, registration metadata).
        """
        versions = self.list_model_versions(name)
        if not versions:
            raise ValueError(f"No registered model found with name '{name}'")

        if version is None:
            version = max(v["version"] for v in versions)

        if not any(v["version"] == version for v in versions):
            raise ValueError(f"Version {version} not found for model '{name}'")

        version_prefix = f"{self.base_prefix}/{name}/v{version}"

        # Load registration metadata
        metadata_blob = self.bucket.blob(f"{version_prefix}/registration.json")
        registration = json.loads(metadata_blob.download_as_text())

        # Load word embedding model
        word_blob = self.bucket.blob(f"{version_prefix}/word_embedding.pkl")
        word_model = pickle.loads(word_blob.download_as_string())

        # Load document embedding model
        doc_blob = self.bucket.blob(f"{version_prefix}/document_embedding.pkl")
        doc_model = pickle.loads(doc_blob.download_as_string())

        return word_model, doc_model, registration

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
        word_model, doc_model, registration = self.load_registered_model(name, version)

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
