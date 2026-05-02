"""Append-only uploader for collection_predictions_landing."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from google.cloud import bigquery
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CollectionPredictionRow(BaseModel):
    job_id: str
    username: str
    game_id: int
    outcome: str
    predicted_prob: float
    predicted_label: bool
    threshold: float | None
    model_name: str
    model_version: int
    score_ts: datetime


class CollectionPredictionsUploader:
    """Append rows to raw.collection_predictions_landing."""

    def __init__(self, table_id: str, client: bigquery.Client | None = None):
        self.table_id = table_id
        self.client = client or bigquery.Client()

    def upload(self, rows: List[CollectionPredictionRow]) -> int:
        """Insert rows, return count inserted. Raises on any insertion error."""
        if not rows:
            return 0
        payload = [
            {
                "job_id": r.job_id,
                "username": r.username,
                "game_id": r.game_id,
                "outcome": r.outcome,
                "predicted_prob": r.predicted_prob,
                "predicted_label": r.predicted_label,
                "threshold": r.threshold,
                "model_name": r.model_name,
                "model_version": r.model_version,
                "score_ts": r.score_ts.astimezone(timezone.utc).isoformat(),
            }
            for r in rows
        ]
        errors = self.client.insert_rows_json(self.table_id, payload)
        if errors:
            raise RuntimeError(f"BQ insert errors: {errors}")
        logger.info(f"Inserted {len(payload)} rows into {self.table_id}")
        return len(payload)
