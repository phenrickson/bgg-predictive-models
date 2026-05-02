"""BigQuery client for the collection_models_registry table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from google.cloud import bigquery


@dataclass(frozen=True)
class RegistryEntry:
    username: str
    outcome: str
    model_version: int
    finalize_through_year: Optional[int]
    gcs_path: str
    status: str


class CollectionRegistry:
    """Read access to the collection_models_registry table."""

    def __init__(self, table_id: str, client: Optional[bigquery.Client] = None):
        self.table_id = table_id
        self.client = client or bigquery.Client()

    def lookup_latest(self, username: str, outcome: str) -> Optional[RegistryEntry]:
        """Return the highest-version active row for (username, outcome), or None."""
        sql = f"""
            SELECT username, outcome, model_version, finalize_through_year,
                   gcs_path, status
            FROM `{self.table_id}`
            WHERE username = @username
              AND outcome = @outcome
              AND status = 'active'
            ORDER BY model_version DESC
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
                bigquery.ScalarQueryParameter("outcome", "STRING", outcome),
            ]
        )
        rows = list(self.client.query(sql, job_config=job_config).result())
        if not rows:
            return None
        r = rows[0]
        return RegistryEntry(
            username=r.username,
            outcome=r.outcome,
            model_version=r.model_version,
            finalize_through_year=r.finalize_through_year,
            gcs_path=r.gcs_path,
            status=r.status,
        )

    def list_active(self, outcome: Optional[str] = None) -> List[RegistryEntry]:
        """Return all active rows, optionally filtered by outcome."""
        where_outcome = "AND outcome = @outcome" if outcome else ""
        sql = f"""
            SELECT username, outcome, model_version, finalize_through_year,
                   gcs_path, status
            FROM `{self.table_id}`
            WHERE status = 'active'
              {where_outcome}
        """
        params = []
        if outcome:
            params.append(bigquery.ScalarQueryParameter("outcome", "STRING", outcome))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        rows = self.client.query(sql, job_config=job_config).result()
        return [
            RegistryEntry(
                username=r.username,
                outcome=r.outcome,
                model_version=r.model_version,
                finalize_through_year=r.finalize_through_year,
                gcs_path=r.gcs_path,
                status=r.status,
            )
            for r in rows
        ]
