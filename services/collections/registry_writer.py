"""Write-side BigQuery client for the collection_models_registry table.

Sibling of services.collections.registry (the read side). Inserts a new
active row and flips any prior active row for (username, outcome) to inactive.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from google.cloud import bigquery

logger = logging.getLogger(__name__)


class RegistryWriter:
    """Append a new active deployment row, demoting any prior active row."""

    def __init__(self, table_id: str, client: Optional[bigquery.Client] = None):
        self.table_id = table_id
        self.client = client or bigquery.Client()

    def register_deployment(
        self,
        *,
        username: str,
        outcome: str,
        model_version: int,
        gcs_path: str,
        finalize_through_year: Optional[int],
        registered_at: datetime,
    ) -> None:
        """Flip prior active row(s) to inactive, then insert the new active row.

        Two sequential BigQuery statements (not a transaction). The reader's
        lookup_latest uses ORDER BY model_version DESC LIMIT 1, so the
        intermediate state — zero active rows for ~50ms — is still correct.
        """
        self._deactivate_prior(username, outcome)
        try:
            self._insert_active(
                username=username,
                outcome=outcome,
                model_version=model_version,
                gcs_path=gcs_path,
                finalize_through_year=finalize_through_year,
                registered_at=registered_at,
            )
        except Exception:
            logger.warning(
                "registry insert FAILED after GCS upload — "
                "GCS artifact at %s is orphaned. Retry promote to recover.",
                gcs_path,
            )
            raise

    def _deactivate_prior(self, username: str, outcome: str) -> None:
        sql = f"""
            UPDATE `{self.table_id}`
            SET status = 'inactive'
            WHERE username = @username
              AND outcome = @outcome
              AND status = 'active'
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
                bigquery.ScalarQueryParameter("outcome", "STRING", outcome),
            ]
        )
        self.client.query(sql, job_config=job_config).result()

    def _insert_active(
        self,
        *,
        username: str,
        outcome: str,
        model_version: int,
        gcs_path: str,
        finalize_through_year: Optional[int],
        registered_at: datetime,
    ) -> None:
        sql = f"""
            INSERT INTO `{self.table_id}`
                (username, outcome, model_version, finalize_through_year,
                 gcs_path, registered_at, status)
            VALUES
                (@username, @outcome, @model_version, @finalize_through_year,
                 @gcs_path, @registered_at, 'active')
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
                bigquery.ScalarQueryParameter("outcome", "STRING", outcome),
                bigquery.ScalarQueryParameter("model_version", "INT64", model_version),
                bigquery.ScalarQueryParameter(
                    "finalize_through_year", "INT64", finalize_through_year
                ),
                bigquery.ScalarQueryParameter("gcs_path", "STRING", gcs_path),
                bigquery.ScalarQueryParameter("registered_at", "TIMESTAMP", registered_at),
            ]
        )
        self.client.query(sql, job_config=job_config).result()
