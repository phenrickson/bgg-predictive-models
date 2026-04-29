"""Authentication utilities for the scoring service."""

import os
import logging
from typing import Optional
from google.cloud import storage
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


class GCPAuthenticator:
    """Handles Google Cloud Platform authentication for the scoring service."""

    def __init__(self, project_id: Optional[str] = None):
        """Initialize the authenticator.

        Args:
            project_id: Optional GCP project ID. If not provided, will try to get from environment.
        """
        self.project_id = project_id or self._get_project_id()
        self._storage_client = None

    def _get_project_id(self) -> str:
        """Get project ID from environment or metadata service."""
        # Try environment variable first
        project_id = os.getenv("GCP_PROJECT_ID")
        if project_id:
            return project_id

        # Try to get from default credentials
        try:
            _, project_id = default()
            if project_id:
                return project_id
        except DefaultCredentialsError:
            pass

        # Try Google Cloud metadata service (when running on GCP)
        try:
            import requests

            metadata_url = (
                "http://metadata.google.internal/computeMetadata/v1/project/project-id"
            )
            headers = {"Metadata-Flavor": "Google"}
            response = requests.get(metadata_url, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.text
        except Exception:
            pass

        raise AuthenticationError(
            "Could not determine GCP project ID. Set GCP_PROJECT_ID environment variable "
            "or ensure the service is running with proper GCP credentials."
        )

    def get_storage_client(self) -> storage.Client:
        """Get authenticated Google Cloud Storage client.

        Returns:
            Authenticated storage client

        Raises:
            AuthenticationError: If authentication fails
        """
        if self._storage_client is None:
            try:
                # Use Application Default Credentials (ADC)
                # This works with:
                # - Service account keys (GOOGLE_APPLICATION_CREDENTIALS)
                # - Workload Identity Federation
                # - Compute Engine default service account
                # - Cloud Shell credentials
                # - gcloud user credentials
                self._storage_client = storage.Client(project=self.project_id)
                logger.info(
                    f"Created storage client for GCP project: {self.project_id}"
                )

            except Exception as e:
                logger.error(f"Failed to create authenticated storage client: {str(e)}")
                raise AuthenticationError(f"Failed to authenticate to GCP: {str(e)}")

        return self._storage_client

    def verify_bucket_access(self, bucket_name: str) -> bool:
        """Verify access to a specific bucket.

        Args:
            bucket_name: Name of the bucket to verify access to

        Returns:
            True if bucket is accessible, False otherwise
        """
        try:
            client = self.get_storage_client()
            bucket = client.bucket(bucket_name)

            # Test bucket access by listing objects (only requires object-level permissions)
            # This works with roles/storage.objectAdmin without needing storage.buckets.get
            list(bucket.list_blobs(max_results=1))
            logger.info(f"Successfully verified access to bucket: {bucket_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to access bucket {bucket_name}: {str(e)}")
            return False

    def get_authentication_info(self) -> dict:
        """Get information about the current authentication setup.

        Returns:
            Dictionary with authentication details
        """
        info = {"project_id": self.project_id, "credentials_source": "unknown"}

        # Check for service account key file
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            info["credentials_source"] = "service_account_key"
            info["credentials_file"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # Check for workload identity
        elif os.getenv("GOOGLE_CLOUD_PROJECT"):
            info["credentials_source"] = "workload_identity_or_metadata"

        # Check if running on GCP
        try:
            import requests

            metadata_url = "http://metadata.google.internal/computeMetadata/v1/"
            headers = {"Metadata-Flavor": "Google"}
            response = requests.get(metadata_url, headers=headers, timeout=2)
            if response.status_code == 200:
                info["running_on_gcp"] = True
                info["credentials_source"] = "compute_engine_default"
        except Exception:
            info["running_on_gcp"] = False

        return info


def get_authenticated_storage_client(
    project_id: Optional[str] = None,
) -> storage.Client:
    """Get an authenticated Google Cloud Storage client.

    This is a convenience function that handles the authentication setup.

    Args:
        project_id: Optional GCP project ID

    Returns:
        Authenticated storage client

    Raises:
        AuthenticationError: If authentication fails
    """
    authenticator = GCPAuthenticator(project_id)
    return authenticator.get_storage_client()


def verify_authentication(
    project_id: Optional[str] = None, bucket_name: Optional[str] = None
) -> dict:
    """Verify authentication setup and return status information.

    Args:
        project_id: Optional GCP project ID
        bucket_name: Optional bucket name to test access

    Returns:
        Dictionary with authentication status and details
    """
    try:
        authenticator = GCPAuthenticator(project_id)
        auth_info = authenticator.get_authentication_info()

        # Test storage client creation
        client = authenticator.get_storage_client()  # noqa
        auth_info["storage_client_created"] = True

        # Test bucket access if provided
        if bucket_name:
            auth_info["bucket_accessible"] = authenticator.verify_bucket_access(
                bucket_name
            )

        auth_info["status"] = "success"

    except Exception as e:
        auth_info = {"status": "error", "error": str(e), "project_id": project_id}

    return auth_info
