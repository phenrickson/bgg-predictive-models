"""FastAPI app for the collection scoring service.

Endpoints:
- GET  /health
- GET  /models
- GET  /model/{username}/{outcome}/info
- POST /predict_own   (Task 9)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.cloud import bigquery

# Make project root importable when running from services/collections/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from services.collections.auth import GCPAuthenticator, AuthenticationError  # noqa: E402
from services.collections.registry import CollectionRegistry  # noqa: E402
from services.collections.registered_model import RegisteredCollectionModel  # noqa: E402
from src.utils.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize once
try:
    authenticator = GCPAuthenticator()
    GCP_PROJECT_ID = authenticator.project_id
    config = load_config()
    BUCKET_NAME = config.get_bucket_name()
    ENVIRONMENT_PREFIX = config.get_environment_prefix()
    REGISTRY_TABLE = config.get_collection_registry_table()
    LANDING_TABLE = config.get_collection_landing_table()
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    registry = CollectionRegistry(REGISTRY_TABLE, bq_client)
except AuthenticationError as e:
    logger.error(f"Auth failed: {e}")
    raise

app = FastAPI(title="BGG Collection Scoring", version="0.1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "project_id": GCP_PROJECT_ID,
        "bucket": BUCKET_NAME,
        "environment": ENVIRONMENT_PREFIX,
        "registry_table": REGISTRY_TABLE,
        "landing_table": LANDING_TABLE,
    }


@app.get("/models")
def list_models(outcome: Optional[str] = None):
    entries = registry.list_active(outcome=outcome)
    return {
        "count": len(entries),
        "models": [
            {
                "username": e.username,
                "outcome": e.outcome,
                "model_version": e.model_version,
                "gcs_path": e.gcs_path,
                "finalize_through_year": e.finalize_through_year,
            }
            for e in entries
        ],
    }


@app.get("/model/{username}/{outcome}/info")
def model_info(username: str, outcome: str):
    entry = registry.lookup_latest(username, outcome)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active registry entry for ({username!r}, {outcome!r})",
        )
    return {
        "username": entry.username,
        "outcome": entry.outcome,
        "model_version": entry.model_version,
        "gcs_path": entry.gcs_path,
        "finalize_through_year": entry.finalize_through_year,
        "status": entry.status,
    }
