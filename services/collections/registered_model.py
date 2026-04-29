"""GCS-backed registration for trained collection models."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from services.scoring.auth import (
    AuthenticationError,
    get_authenticated_storage_client,
)
from src.utils.config import load_config


class RegisteredCollectionModel:
    """Per-user collection model registration in GCS.

    Layout: {env}/services/collections/{username}/{outcome}/v{N}/
            {pipeline.pkl, threshold.json, registration.json}
    """

    def __init__(
        self,
        username: str,
        outcome: str,
        bucket_name: Optional[str] = None,
        environment_prefix: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        self.username = username
        self.outcome = outcome

        if bucket_name is None or environment_prefix is None:
            cfg = load_config()
            bucket_name = bucket_name or cfg.get_bucket_name()
            environment_prefix = environment_prefix or cfg.get_environment_prefix()

        try:
            self.storage_client = get_authenticated_storage_client(project_id)
        except AuthenticationError as e:
            raise ValueError(f"Authentication failed: {e}")

        self.bucket = self.storage_client.bucket(bucket_name)
        self.bucket_name = bucket_name
        self.environment_prefix = environment_prefix
        self.base_prefix = f"{environment_prefix}/services/collections/{username}/{outcome}"

    def list_versions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for blob in self.bucket.list_blobs(prefix=self.base_prefix):
            if blob.name.endswith("/registration.json"):
                out.append(json.loads(blob.download_as_text()))
        return sorted(out, key=lambda v: v["version"])

    def register(
        self,
        pipeline: Any,
        threshold: Optional[float],
        source_metadata: Dict[str, Any],
        description: str,
    ) -> Dict[str, Any]:
        version = max((v["version"] for v in self.list_versions()), default=0) + 1
        prefix = f"{self.base_prefix}/v{version}"

        registration = {
            "username": self.username,
            "outcome": self.outcome,
            "version": version,
            "description": description,
            "source": source_metadata,
            "threshold": threshold,
            "registered_at": datetime.now().isoformat(),
        }

        self.bucket.blob(f"{prefix}/pipeline.pkl").upload_from_string(
            pickle.dumps(pipeline), content_type="application/octet-stream"
        )
        self.bucket.blob(f"{prefix}/threshold.json").upload_from_string(
            json.dumps({"threshold": threshold}), content_type="application/json"
        )
        self.bucket.blob(f"{prefix}/registration.json").upload_from_string(
            json.dumps(registration, indent=2), content_type="application/json"
        )
        return registration

    def load(
        self, version: Optional[int] = None
    ) -> Tuple[Any, Optional[float], Dict[str, Any]]:
        versions = self.list_versions()
        if not versions:
            raise ValueError(f"No registered versions for {self.username}/{self.outcome}")
        if version is None:
            version = max(v["version"] for v in versions)
        elif not any(v["version"] == version for v in versions):
            raise ValueError(f"Version {version} not registered")

        prefix = f"{self.base_prefix}/v{version}"
        pipeline = pickle.loads(
            self.bucket.blob(f"{prefix}/pipeline.pkl").download_as_string()
        )
        threshold = json.loads(
            self.bucket.blob(f"{prefix}/threshold.json").download_as_text()
        ).get("threshold")
        registration = json.loads(
            self.bucket.blob(f"{prefix}/registration.json").download_as_text()
        )
        return pipeline, threshold, registration
