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


import uuid  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import List  # noqa: E402

import polars as pl  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from services.collections.change_detection import find_unscored  # noqa: E402
from services.collections.landing_uploader import (  # noqa: E402
    CollectionPredictionRow, CollectionPredictionsUploader,
)
from src.data.loader import BGGDataLoader  # noqa: E402


class PredictOwnRequest(BaseModel):
    username: str
    game_ids: Optional[List[int]] = None
    use_change_detection: bool = False
    upload_to_data_warehouse: bool = True
    model_version: Optional[int] = None  # None = latest active


class PredictOwnPrediction(BaseModel):
    game_id: int
    predicted_prob: float
    predicted_label: bool


class PredictOwnResponse(BaseModel):
    job_id: str
    username: str
    outcome: str
    model_version: int
    n_scored: int
    score_ts: datetime
    predictions: List[PredictOwnPrediction]


# Cap per-request work to bound Cloud Run memory. The daily GHA loops calls
# until n_scored==0, mirroring the run-complexity-scoring workflow.
MAX_GAMES_PER_REQUEST = 25000


# Reuse a single BGGDataLoader for feature pulls
_loader: Optional[BGGDataLoader] = None


def _get_loader() -> BGGDataLoader:
    global _loader
    if _loader is None:
        _loader = BGGDataLoader(config.get_bigquery_config())
    return _loader


# Cache loaded pipelines: (username, outcome, version) -> (pipeline, threshold)
_PIPELINE_CACHE: dict = {}


def _load_pipeline(username: str, outcome: str, version: int):
    key = (username, outcome, version)
    if key not in _PIPELINE_CACHE:
        rcm = RegisteredCollectionModel(
            username=username,
            outcome=outcome,
            bucket_name=BUCKET_NAME,
            environment_prefix=ENVIRONMENT_PREFIX,
            project_id=GCP_PROJECT_ID,
        )
        pipeline, threshold, _ = rcm.load(version=version)
        _PIPELINE_CACHE[key] = (pipeline, threshold)
    return _PIPELINE_CACHE[key]


@app.post("/predict_own", response_model=PredictOwnResponse)
def predict_own(req: PredictOwnRequest):
    job_id = str(uuid.uuid4())

    # 1. Resolve registry entry
    entry = registry.lookup_latest(req.username, "own")
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active 'own' model for user {req.username!r}",
        )
    version = req.model_version if req.model_version is not None else entry.model_version

    # 2. Determine target game_ids
    if req.use_change_detection and req.game_ids:
        raise HTTPException(
            status_code=400,
            detail="Pass either game_ids or use_change_detection=true, not both",
        )
    if req.use_change_detection:
        game_ids = find_unscored(
            username=req.username,
            outcome="own",
            model_version=version,
            landing_table=LANDING_TABLE,
            candidate_table="bgg-data-warehouse.analytics.games_features",
            bq_client=bq_client,
            limit=MAX_GAMES_PER_REQUEST,
        )
    elif req.game_ids:
        game_ids = list(req.game_ids)
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide game_ids or use_change_detection=true",
        )

    score_ts = datetime.now(timezone.utc)

    if not game_ids:
        return PredictOwnResponse(
            job_id=job_id,
            username=req.username,
            outcome="own",
            model_version=version,
            n_scored=0,
            score_ts=score_ts,
            predictions=[],
        )

    # 3. Load pipeline + threshold
    try:
        pipeline, threshold = _load_pipeline(req.username, "own", version)
    except Exception as e:  # GCS load / pickle errors
        logger.exception("Failed loading pipeline")
        raise HTTPException(status_code=502, detail=f"Pipeline load failed: {e}")

    # 4. Pull features
    try:
        # Match the training-side and Streamlit feature loading: include
        # predicted_complexity. Pipelines were fit with this column.
        features_df = (
            _get_loader()
            .load_features(use_predicted_complexity=True, use_embeddings=False)
            .filter(pl.col("game_id").is_in(game_ids))
        )
        X = features_df.to_pandas()
    except Exception as e:
        logger.exception("Feature load failed")
        raise HTTPException(status_code=502, detail=f"Feature load failed: {e}")

    # Defense: change-detection can return game_ids that don't materialize in
    # the joined features view (e.g. streaming-buffer lag, stale lookup).
    # Treat zero-row feature frames as "nothing to score this batch" rather
    # than letting sklearn raise an unhelpful column-name error.
    if X.empty:
        logger.warning(
            "find_unscored returned %d game_ids but features filter yielded 0 rows; "
            "returning n_scored=0 for this batch",
            len(game_ids),
        )
        return PredictOwnResponse(
            job_id=job_id,
            username=req.username,
            outcome="own",
            model_version=version,
            n_scored=0,
            score_ts=score_ts,
            predictions=[],
        )

    # 5. Score
    proba = pipeline.predict_proba(X)[:, 1]
    thr = threshold if threshold is not None else 0.5
    labels = (proba >= thr)

    predictions = [
        PredictOwnPrediction(
            game_id=int(gid),
            predicted_prob=float(p),
            predicted_label=bool(lbl),
        )
        for gid, p, lbl in zip(X["game_id"].tolist(), proba, labels)
    ]

    # 6. Upload
    if req.upload_to_data_warehouse and predictions:
        rows = [
            CollectionPredictionRow(
                job_id=job_id,
                username=req.username,
                game_id=p.game_id,
                outcome="own",
                predicted_prob=p.predicted_prob,
                predicted_label=p.predicted_label,
                threshold=threshold,
                model_name=f"collection_own_{req.username}",
                model_version=version,
                score_ts=score_ts,
            )
            for p in predictions
        ]
        CollectionPredictionsUploader(LANDING_TABLE, bq_client).upload(rows)

    return PredictOwnResponse(
        job_id=job_id,
        username=req.username,
        outcome="own",
        model_version=version,
        n_scored=len(predictions),
        score_ts=score_ts,
        predictions=predictions,
    )
