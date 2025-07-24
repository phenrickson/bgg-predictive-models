"""Manages registered models for production deployment."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from google.cloud import storage
from src.models.experiments import Experiment, ExperimentTracker


class ModelValidationError(Exception):
    """Raised when model validation fails."""

    pass


class RegisteredModel(ExperimentTracker):
    """Manages registered models for production deployment."""

    def __init__(
        self,
        model_type: str,
        bucket_name: str,
        project_id: Optional[str] = None,
        base_prefix: str = "models/registered",
    ):
        """Initialize registered model manager.

        Args:
            model_type: Type of model (hurdle, complexity, etc.)
            bucket_name: Google Cloud Storage bucket name
            project_id: Optional Google Cloud project ID (uses environment default if not provided)
            base_prefix: Base prefix for registered model storage
        """
        # Initialize base tracker with local directory
        super().__init__(model_type)

        # Initialize GCS client
        try:
            if project_id:
                self.storage_client = storage.Client(project=project_id)
            else:
                # Try to get project ID from environment
                import os

                project_id = os.getenv("GCP_PROJECT_ID")
                if not project_id:
                    raise ValueError(
                        "Project ID not provided and GCP_PROJECT_ID environment variable not set"
                    )
                self.storage_client = storage.Client(project=project_id)

            self.bucket = self.storage_client.bucket(bucket_name)
            self.base_prefix = f"{base_prefix}/{model_type}"

            # Verify bucket exists and we have access
            if not self.bucket.exists():
                raise ValueError(f"Bucket {bucket_name} does not exist")

        except Exception as e:
            raise ValueError(f"Failed to initialize GCS client: {str(e)}")

        # Define model statuses
        self.STATUSES = {
            "DRAFT": "draft",
            "REGISTERED": "registered",
            "ARCHIVED": "archived",
        }

    def validate_model(
        self, experiment: Experiment, validation_metrics: Dict[str, float]
    ) -> bool:
        """Validate model meets requirements for registration.

        Args:
            experiment: Experiment to validate
            validation_metrics: Dictionary of metric thresholds
                e.g. {"min_r2": 0.7, "max_rmse": 1.0}

        Returns:
            True if validation passes

        Raises:
            ModelValidationError if validation fails
        """
        # Get test metrics
        test_metrics = experiment.get_metrics("test")

        # Validate each metric
        for metric_name, threshold in validation_metrics.items():
            if metric_name.startswith("min_"):
                metric = metric_name[4:]  # Remove "min_"
                if test_metrics[metric] < threshold:
                    raise ModelValidationError(
                        f"{metric} value {test_metrics[metric]} below minimum threshold {threshold}"
                    )
            elif metric_name.startswith("max_"):
                metric = metric_name[4:]  # Remove "max_"
                if test_metrics[metric] > threshold:
                    raise ModelValidationError(
                        f"{metric} value {test_metrics[metric]} above maximum threshold {threshold}"
                    )

        return True

    def register(
        self,
        experiment: Experiment,
        name: str,
        description: str,
        validation_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a model for production use.

        Args:
            experiment: Experiment to register
            name: Name for the registered model
            description: Description of the model
            validation_metrics: Optional metric thresholds for validation
            metadata: Optional additional metadata

        Returns:
            Dictionary containing registration details
        """
        # Validate model if metrics provided
        if validation_metrics:
            self.validate_model(experiment, validation_metrics)

        # Load experiment pipeline and info
        pipeline = experiment.load_pipeline()
        model_info = experiment.get_model_info()

        # Get next version number
        existing_versions = self.list_model_versions(name)
        next_version = max([v["version"] for v in existing_versions], default=0) + 1

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
            "validation_metrics": validation_metrics,
            "registered_at": datetime.now().isoformat(),
            "registered_by": metadata.get("registered_by") if metadata else None,
            "metadata": metadata or {},
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

        return registration

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """List all registered models for this model type.

        Returns:
            List of registered model metadata
        """
        models = []
        blobs = self.bucket.list_blobs(prefix=self.base_prefix)

        for blob in blobs:
            if blob.name.endswith("registration.json"):
                # Load registration metadata
                registration = json.loads(blob.download_as_text())
                models.append(registration)

        return sorted(models, key=lambda x: (x["name"], x["version"]))

    def list_model_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a specific registered model.

        Args:
            name: Name of the registered model

        Returns:
            List of version metadata
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
        """Load a registered model pipeline and metadata.

        Args:
            name: Name of the registered model
            version: Optional specific version (latest if None)

        Returns:
            Tuple of (model pipeline, registration metadata)
        """
        # Get versions
        versions = self.list_model_versions(name)
        if not versions:
            raise ValueError(f"No registered model found with name '{name}'")

        # Use latest version if not specified
        if version is None:
            version = max(v["version"] for v in versions)

        # Verify version exists
        if not any(v["version"] == version for v in versions):
            raise ValueError(f"Version {version} not found for model '{name}'")

        # Load registration metadata
        version_prefix = f"{self.base_prefix}/{name}/v{version}"
        metadata_blob = self.bucket.blob(f"{version_prefix}/registration.json")
        registration = json.loads(metadata_blob.download_as_text())

        # Load pipeline
        pipeline_blob = self.bucket.blob(f"{version_prefix}/pipeline.pkl")
        pipeline = pickle.loads(pipeline_blob.download_as_string())

        return pipeline, registration

    def archive_model(
        self, name: str, version: Optional[int] = None, reason: str = ""
    ) -> Dict[str, Any]:
        """Archive a registered model version.

        Args:
            name: Name of the registered model
            version: Optional specific version (latest if None)
            reason: Reason for archiving

        Returns:
            Updated registration metadata
        """
        # Load current registration
        pipeline, registration = self.load_registered_model(name, version)

        # Update status
        registration["status"] = self.STATUSES["ARCHIVED"]
        registration["archived_at"] = datetime.now().isoformat()
        registration["archive_reason"] = reason

        # Save updated registration
        version = registration["version"]
        version_prefix = f"{self.base_prefix}/{name}/v{version}"
        metadata_blob = self.bucket.blob(f"{version_prefix}/registration.json")
        metadata_blob.upload_from_string(
            json.dumps(registration, indent=2), content_type="application/json"
        )

        return registration
