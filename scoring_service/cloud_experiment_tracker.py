import json
from typing import Dict, Any, Optional, List
from pathlib import Path

from google.cloud import storage
from src.models.experiments import ExperimentTracker, Experiment


class CloudExperimentTracker(ExperimentTracker):
    """
    Extended ExperimentTracker that supports retrieving models from Google Cloud Storage
    """

    def __init__(
        self, model_type: str, bucket_name: str, base_prefix: str = "models/experiments"
    ):
        """
        Initialize Cloud Experiment Tracker

        Args:
            model_type: Type of model (hurdle, complexity, etc.)
            bucket_name: Google Cloud Storage bucket name
            base_prefix: Base prefix for model storage in bucket
        """
        super().__init__(model_type)

        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.base_prefix = f"{base_prefix}/{model_type}"

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List experiments from Google Cloud Storage

        Returns:
            List of experiment metadata
        """
        experiments = []
        blobs = self.bucket.list_blobs(prefix=self.base_prefix)

        # Track unique experiment names and their versions
        experiment_versions = {}  # noqa: F841

        for blob in blobs:
            # Extract experiment name and version from blob path
            relative_path = blob.name.replace(f"{self.base_prefix}/", "")
            parts = relative_path.split("/")

            if len(parts) >= 2 and parts[1].startswith("v"):
                experiment_name = parts[0]
                version_str = parts[1]

                if version_str.startswith("v") and version_str[1:].isdigit():
                    version = int(version_str[1:])

                    # Try to load metadata
                    metadata_blob = self.bucket.blob(
                        f"{self.base_prefix}/{experiment_name}/{version_str}/metadata.json"
                    )
                    metadata = {}
                    if metadata_blob.exists():
                        metadata_content = metadata_blob.download_as_text()
                        metadata = json.loads(metadata_content)

                    experiments.append(
                        {
                            "name": experiment_name,
                            "version": version,
                            "full_name": f"{experiment_name}/v{version}",
                            "description": metadata.get("description"),
                            "timestamp": metadata.get("timestamp"),
                        }
                    )

        return sorted(experiments, key=lambda x: (x["name"], x["version"]))

    def load_experiment(self, name: str, version: Optional[int] = None) -> Experiment:
        """
        Load an experiment from Google Cloud Storage

        Args:
            name: Experiment name
            version: Specific version (latest if None)

        Returns:
            Experiment object
        """
        experiments = self.list_experiments()
        matching_experiments = [exp for exp in experiments if exp["name"] == name]

        if not matching_experiments:
            raise ValueError(f"No experiments found matching '{name}'")

        # Use latest version if not specified
        if version is None:
            version = max(exp["version"] for exp in matching_experiments)

        # Verify version exists
        if not any(exp["version"] == version for exp in matching_experiments):
            raise ValueError(f"Version {version} not found for experiment '{name}'")

        # Construct version directory prefix
        version_prefix = f"{self.base_prefix}/{name}/v{version}"

        # Download metadata
        metadata_blob = self.bucket.blob(f"{version_prefix}/metadata.json")
        metadata_content = metadata_blob.download_as_text()
        metadata = json.loads(metadata_content)

        # Create a temporary local directory for experiment
        local_exp_dir = Path(f"/tmp/experiments/{name}/v{version}")
        local_exp_dir.mkdir(parents=True, exist_ok=True)

        # Download relevant files
        files_to_download = [
            "metadata.json",
            "parameters.json",
            "model_info.json",
            "pipeline.pkl",
        ]

        for filename in files_to_download:
            blob = self.bucket.blob(f"{version_prefix}/{filename}")
            if blob.exists():
                blob.download_to_filename(local_exp_dir / filename)

        # Create and return Experiment object
        return Experiment(
            name=name,
            base_dir=local_exp_dir,
            description=metadata.get("description"),
            metadata=metadata.get("metadata", {}),
        )
