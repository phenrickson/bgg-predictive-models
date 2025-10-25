"""
Efficient experiment loading for Streamlit dashboard.
Loads only essential experiment data from GCS without mounting entire bucket.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from google.cloud import storage
import google.cloud.exceptions
from dotenv import load_dotenv

from .config import load_config

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ExperimentLoader:
    """Efficient loader for experiment data from GCS."""

    def __init__(
        self, bucket_name: Optional[str] = None, config_path: Optional[str] = None
    ):
        """Initialize the experiment loader.

        Args:
            bucket_name: GCS bucket name. If None, uses config.
            config_path: Path to config file.
        """
        # Get bucket name from config if not provided
        if bucket_name is None:
            try:
                config = load_config(config_path)
                bucket_name = config.get_bucket_name()
                logger.info(f"Using bucket from config: {bucket_name}")
            except Exception as e:
                logger.error(f"Failed to load bucket name from config: {e}")
                raise

        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

        # Cache for experiment metadata
        self._metadata_cache = {}
        self._experiments_cache = {}

    def list_model_types(self) -> List[str]:
        """List available model types in the experiments bucket.

        Returns:
            List of model type names.
        """
        try:
            # List all prefixes under models/experiments/
            blobs = self.bucket.list_blobs(prefix="models/experiments/", delimiter="/")

            # Get the prefixes (which represent model types)
            model_types = []
            for page in blobs.pages:
                model_types.extend(
                    [
                        prefix.rstrip("/").split("/")[-1]
                        for prefix in page.prefixes
                        if not prefix.endswith("predictions/")
                    ]
                )

            return sorted(model_types)
        except Exception as e:
            logger.error(f"Error listing model types: {e}")
            return []

    def list_experiments(self, model_type: str) -> List[Dict[str, Any]]:
        """List experiments for a given model type with basic metadata.

        Args:
            model_type: The model type (e.g., 'catboost-complexity').

        Returns:
            List of experiment dictionaries with metadata.
        """
        cache_key = f"experiments_{model_type}"
        if cache_key in self._experiments_cache:
            return self._experiments_cache[cache_key]

        try:
            experiments = []
            prefix = f"models/experiments/{model_type}/"

            # List all experiment directories
            blobs = self.bucket.list_blobs(prefix=prefix, delimiter="/")

            experiment_dirs = []
            for page in blobs.pages:
                experiment_dirs.extend(
                    [prefix.rstrip("/").split("/")[-1] for prefix in page.prefixes]
                )

            # Load metadata for each experiment in parallel
            def load_experiment_metadata(exp_name):
                try:
                    return self._load_single_experiment_metadata(model_type, exp_name)
                except Exception as e:
                    logger.warning(
                        f"Failed to load metadata for {model_type}/{exp_name}: {e}"
                    )
                    return None

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(load_experiment_metadata, exp_name)
                    for exp_name in experiment_dirs
                ]

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        experiments.append(result)

            # Sort by timestamp (newest first)
            experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            # Cache the results
            self._experiments_cache[cache_key] = experiments

            return experiments

        except Exception as e:
            logger.error(f"Error listing experiments for {model_type}: {e}")
            return []

    def _load_single_experiment_metadata(
        self, model_type: str, exp_name: str
    ) -> Dict[str, Any]:
        """Load metadata for a single experiment.

        Args:
            model_type: The model type.
            exp_name: The experiment name.

        Returns:
            Dictionary with experiment metadata.
        """
        # Try to load the metadata.json file
        metadata_path = f"models/experiments/{model_type}/{exp_name}/metadata.json"

        try:
            blob = self.bucket.blob(metadata_path)
            metadata_content = blob.download_as_text()
            metadata = json.loads(metadata_content)

            # Add computed fields
            metadata["full_name"] = f"{exp_name}"
            metadata["model_type"] = model_type
            metadata["experiment_name"] = exp_name

            return metadata

        except google.cloud.exceptions.NotFound:
            # If no metadata.json, create basic metadata from directory structure
            logger.warning(f"No metadata.json found for {model_type}/{exp_name}")
            return {
                "full_name": f"{exp_name}",
                "model_type": model_type,
                "experiment_name": exp_name,
                "timestamp": "",
                "status": "unknown",
            }

    def load_experiment_details(self, model_type: str, exp_name: str) -> Dict[str, Any]:
        """Load detailed experiment information including metrics and parameters.

        Args:
            model_type: The model type.
            exp_name: The experiment name.

        Returns:
            Dictionary with detailed experiment information.
        """
        cache_key = f"details_{model_type}_{exp_name}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        try:
            details = {}
            base_path = f"models/experiments/{model_type}/{exp_name}"

            # Load various experiment files
            files_to_load = {
                "metadata": "metadata.json",
                "metrics": "metrics.json",
                "parameters": "parameters.json",
                "feature_importance": "feature_importance.json",
            }

            def load_file(file_info):
                file_key, filename = file_info
                file_path = f"{base_path}/{filename}"
                try:
                    blob = self.bucket.blob(file_path)
                    content = blob.download_as_text()
                    return file_key, json.loads(content)
                except google.cloud.exceptions.NotFound:
                    logger.debug(f"File not found: {file_path}")
                    return file_key, None
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    return file_key, None

            # Load files in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(load_file, item) for item in files_to_load.items()
                ]

                for future in as_completed(futures):
                    file_key, content = future.result()
                    if content:
                        details[file_key] = content

            # Cache the results
            self._metadata_cache[cache_key] = details

            return details

        except Exception as e:
            logger.error(
                f"Error loading experiment details for {model_type}/{exp_name}: {e}"
            )
            return {}

    def load_predictions(
        self, model_type: str, exp_name: str, dataset: str = "test"
    ) -> Optional[pd.DataFrame]:
        """Load predictions for an experiment.

        Args:
            model_type: The model type.
            exp_name: The experiment name.
            dataset: Dataset name ('train', 'tune', 'test').

        Returns:
            DataFrame with predictions or None if not found.
        """
        try:
            predictions_path = f"models/experiments/{model_type}/{exp_name}/predictions_{dataset}.parquet"

            blob = self.bucket.blob(predictions_path)

            # Download to temporary file and load with pandas
            import tempfile

            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as tmp_file:
                blob.download_to_filename(tmp_file.name)
                df = pd.read_parquet(tmp_file.name)

                # Clean up temp file
                os.unlink(tmp_file.name)

                return df

        except google.cloud.exceptions.NotFound:
            logger.debug(f"Predictions file not found: {predictions_path}")
            return None
        except Exception as e:
            logger.error(
                f"Error loading predictions for {model_type}/{exp_name}/{dataset}: {e}"
            )
            return None

    def load_feature_importance(
        self, model_type: str, exp_name: str
    ) -> Optional[Dict[str, Any]]:
        """Load feature importance data for an experiment.

        Args:
            model_type: The model type.
            exp_name: The experiment name.

        Returns:
            Dictionary with feature importance data or None if not found.
        """
        try:
            # Try JSON first (newer format)
            importance_path = (
                f"models/experiments/{model_type}/{exp_name}/feature_importance.json"
            )

            blob = self.bucket.blob(importance_path)
            content = blob.download_as_text()
            return json.loads(content)

        except google.cloud.exceptions.NotFound:
            # Try pickle format (older experiments)
            try:
                importance_path = (
                    f"models/experiments/{model_type}/{exp_name}/feature_importance.pkl"
                )
                blob = self.bucket.blob(importance_path)

                import tempfile

                with tempfile.NamedTemporaryFile(
                    suffix=".pkl", delete=False
                ) as tmp_file:
                    blob.download_to_filename(tmp_file.name)
                    with open(tmp_file.name, "rb") as f:
                        data = pickle.load(f)

                    # Clean up temp file
                    os.unlink(tmp_file.name)

                    return data

            except google.cloud.exceptions.NotFound:
                logger.debug(
                    f"Feature importance not found for {model_type}/{exp_name}"
                )
                return None
        except Exception as e:
            logger.error(
                f"Error loading feature importance for {model_type}/{exp_name}: {e}"
            )
            return None

    def clear_cache(self):
        """Clear all cached data."""
        self._metadata_cache.clear()
        self._experiments_cache.clear()
        logger.info("Experiment loader cache cleared")


# Global instance for use in Streamlit
_experiment_loader = None


def get_experiment_loader(
    bucket_name: Optional[str] = None, config_path: Optional[str] = None
) -> ExperimentLoader:
    """Get a global experiment loader instance.

    Args:
        bucket_name: GCS bucket name. If None, uses config.
        config_path: Path to config file.

    Returns:
        ExperimentLoader instance.
    """
    global _experiment_loader
    if _experiment_loader is None:
        _experiment_loader = ExperimentLoader(bucket_name, config_path)
    return _experiment_loader
