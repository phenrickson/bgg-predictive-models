"""SQL helpers for finding game_ids not yet scored under a given user/version."""

from __future__ import annotations

from typing import List, Optional

from google.cloud import bigquery


def build_unscored_query(landing_table: str, candidate_table: str) -> str:
    """Build the LEFT ANTI JOIN SQL.

    Returns game_ids present in the candidate table that have no row in the
    landing table for the given (username, outcome, model_version).
    """
    return f"""
        SELECT gf.game_id
        FROM `{candidate_table}` gf
        LEFT JOIN `{landing_table}` lp
          ON lp.game_id = gf.game_id
         AND lp.username = @username
         AND lp.outcome = @outcome
         AND lp.model_version = @model_version
        WHERE lp.game_id IS NULL
    """


def find_unscored(
    username: str,
    outcome: str,
    model_version: int,
    landing_table: str,
    candidate_table: str,
    bq_client: Optional[bigquery.Client] = None,
    limit: Optional[int] = None,
) -> List[int]:
    """Return the list of game_ids not yet scored for this user/version."""
    client = bq_client or bigquery.Client()
    sql = build_unscored_query(landing_table, candidate_table)
    if limit is not None:
        sql = sql + f"\n        LIMIT {int(limit)}"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("username", "STRING", username),
            bigquery.ScalarQueryParameter("outcome", "STRING", outcome),
            bigquery.ScalarQueryParameter("model_version", "INT64", model_version),
        ]
    )
    rows = client.query(sql, job_config=job_config).result()
    return [r.game_id for r in rows]
