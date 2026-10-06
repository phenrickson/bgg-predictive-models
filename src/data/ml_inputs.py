"""Latest-per-game ML outputs, read straight from the ML project's raw tables.

These reproduce the warehouse Dataform models bgg_description_embeddings and
bgg_complexity_predictions, so ML stages can read the previous stage's output
mid-chain without a Dataform run in between.

Description embeddings keep only the latest embedding_version (one consistent
vector space): after a version bump, games not yet re-embedded drop out until
they are, which differs deliberately from the incremental Dataform copy.

Complexity history holds superseded model versions in old score_ts partitions.
complexity_min_score_ts finds the earliest score_ts among each game's latest
row; filtering on it gives identical results while pruning those partitions.
"""

from typing import Optional

ML_PROJECT_ID = "bgg-predictive-models"


def latest_description_embeddings_sql(ml_project: str = ML_PROJECT_ID) -> str:
    table = f"`{ml_project}.raw.description_embeddings`"
    return f"""(
  SELECT game_id, embedding, embedding_version, created_ts, job_id
  FROM (
    SELECT game_id, embedding, embedding_version, created_ts, job_id,
      ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY created_ts DESC, job_id DESC) AS rn
    FROM {table}
    WHERE embedding_version = (SELECT MAX(embedding_version) FROM {table})
  )
  WHERE rn = 1
)"""


def complexity_min_score_ts(client, ml_project: str = ML_PROJECT_ID) -> Optional[str]:
    sql = f"""
SELECT CAST(MIN(latest_ts) AS STRING) AS bound
FROM (
  SELECT MAX(score_ts) AS latest_ts
  FROM `{ml_project}.raw.complexity_predictions`
  GROUP BY game_id
)"""
    rows = list(client.query(sql).result())
    return rows[0]["bound"] if rows else None


def latest_complexity_sql(min_score_ts: Optional[str], ml_project: str = ML_PROJECT_ID) -> str:
    bound = f"WHERE score_ts >= TIMESTAMP('{min_score_ts}')" if min_score_ts else ""
    return f"""(
  SELECT game_id, predicted_complexity, score_ts
  FROM (
    SELECT game_id, predicted_complexity, score_ts,
      ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY score_ts DESC, job_id DESC) AS rn
    FROM `{ml_project}.raw.complexity_predictions`
    {bound}
  )
  WHERE rn = 1
)"""
