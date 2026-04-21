"""Storage layer for BGG user collections in BigQuery.

One row per (username, game_id) in `collections.user_collections`. Writes go
through a single MERGE that inserts new rows, updates changed rows, and
soft-deletes rows that are no longer present in the source. Schema is managed
by Terraform — this module does not create tables.
"""

import logging
from typing import Optional

import pandas as pd
import polars as pl
from google.cloud import bigquery

from src.utils.config import load_config

logger = logging.getLogger(__name__)


TABLE_COLUMNS = [
    "game_id",
    "game_name",
    "subtype",
    "collection_id",
    "owned",
    "previously_owned",
    "for_trade",
    "want",
    "want_to_play",
    "want_to_buy",
    "wishlist",
    "wishlist_priority",
    "preordered",
    "user_rating",
    "user_comment",
    "last_modified",
]


class CollectionStorage:
    """Upsert and read user collections in `collections.user_collections`."""

    def __init__(self, environment: str = "dev"):
        self.environment = environment
        config = load_config()
        bq_config = config.get_bigquery_config()

        self.project_id = config.ml_project_id
        self.dataset_id = bq_config.collections_dataset
        self.table_id = "user_collections"
        self.location = bq_config.location

        self.client = bigquery.Client(project=self.project_id)
        self.fq_table = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

        logger.info(
            f"CollectionStorage initialized: {self.fq_table} (env={environment})"
        )

    def _prepare_rows(self, username: str, df: pl.DataFrame) -> list[dict]:
        """Validate input and return a list of JSON-safe dicts for staging.

        - Filters to subtype == 'boardgame'.
        - Rejects empty input.
        - Rejects duplicate (username, game_id) after filtering.
        - Casts `last_modified` from string to ISO-8601 timestamp (or None).
        """
        if df.height == 0:
            raise ValueError(
                f"Cannot save empty collection for user '{username}'. "
                "Refusing to soft-delete every row in this user's collection."
            )

        if "subtype" in df.columns:
            df = df.filter(pl.col("subtype") == "boardgame")

        if df.height == 0:
            raise ValueError(
                f"Collection for user '{username}' is empty after filtering to "
                "boardgame subtype. Refusing to soft-delete every row."
            )

        dupes = df.group_by("game_id").len().filter(pl.col("len") > 1)
        if dupes.height > 0:
            raise ValueError(
                f"duplicate game_id rows in collection for '{username}': "
                f"{dupes['game_id'].to_list()}"
            )

        missing = [c for c in TABLE_COLUMNS if c not in df.columns]
        for col in missing:
            df = df.with_columns(pl.lit(None).alias(col))

        df = df.select(TABLE_COLUMNS)
        pdf = df.to_pandas()

        if "last_modified" in pdf.columns:
            pdf["last_modified"] = pd.to_datetime(
                pdf["last_modified"], errors="coerce"
            )
            pdf["last_modified"] = pdf["last_modified"].apply(
                lambda ts: None if pd.isna(ts) else ts.isoformat()
            )

        # Convert pandas NaN/NaT/numpy types to JSON-safe Python values.
        pdf = pdf.astype(object).where(pd.notnull(pdf), None)
        return pdf.to_dict(orient="records")

    def save_collection(self, username: str, collection_df: pl.DataFrame) -> None:
        """Upsert one user's collection via MERGE.

        Soft-deletes rows present in the table but not in `collection_df`.
        Raises ValueError on empty input or duplicate (username, game_id).
        """
        rows = self._prepare_rows(username, collection_df)

        # Stage to a temp table — MERGE's USING clause needs a table, not params.
        # BigQuery table IDs disallow dashes; sanitize only the username portion.
        safe_user = username.replace("-", "_")
        staging_table = (
            f"{self.project_id}.{self.dataset_id}._staging_{safe_user}"
        )

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            schema=[
                bigquery.SchemaField("game_id", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("game_name", "STRING"),
                bigquery.SchemaField("subtype", "STRING"),
                bigquery.SchemaField("collection_id", "INTEGER"),
                bigquery.SchemaField("owned", "BOOL"),
                bigquery.SchemaField("previously_owned", "BOOL"),
                bigquery.SchemaField("for_trade", "BOOL"),
                bigquery.SchemaField("want", "BOOL"),
                bigquery.SchemaField("want_to_play", "BOOL"),
                bigquery.SchemaField("want_to_buy", "BOOL"),
                bigquery.SchemaField("wishlist", "BOOL"),
                bigquery.SchemaField("wishlist_priority", "INTEGER"),
                bigquery.SchemaField("preordered", "BOOL"),
                bigquery.SchemaField("user_rating", "FLOAT"),
                bigquery.SchemaField("user_comment", "STRING"),
                bigquery.SchemaField("last_modified", "TIMESTAMP"),
            ],
        )

        logger.info(
            f"Staging {len(rows)} rows for '{username}' to {staging_table}"
        )
        self.client.load_table_from_json(
            rows, staging_table, job_config=job_config
        ).result()

        merge_sql = f"""
        MERGE `{self.fq_table}` T
        USING (
          SELECT @username AS username, * FROM `{staging_table}`
        ) S
        ON T.username = S.username AND T.game_id = S.game_id

        WHEN MATCHED THEN UPDATE SET
          game_name         = S.game_name,
          subtype           = S.subtype,
          collection_id     = S.collection_id,
          owned             = S.owned,
          previously_owned  = S.previously_owned,
          for_trade         = S.for_trade,
          want              = S.want,
          want_to_play      = S.want_to_play,
          want_to_buy       = S.want_to_buy,
          wishlist          = S.wishlist,
          wishlist_priority = S.wishlist_priority,
          preordered        = S.preordered,
          user_rating       = S.user_rating,
          user_comment      = S.user_comment,
          last_modified     = S.last_modified,
          updated_at        = CURRENT_TIMESTAMP(),
          removed_at        = NULL

        WHEN NOT MATCHED BY TARGET THEN INSERT (
          username, game_id, game_name, subtype, collection_id,
          owned, previously_owned, for_trade, want, want_to_play, want_to_buy,
          wishlist, wishlist_priority, preordered,
          user_rating, user_comment, last_modified,
          first_seen_at, updated_at, removed_at
        ) VALUES (
          S.username, S.game_id, S.game_name, S.subtype, S.collection_id,
          S.owned, S.previously_owned, S.for_trade, S.want, S.want_to_play, S.want_to_buy,
          S.wishlist, S.wishlist_priority, S.preordered,
          S.user_rating, S.user_comment, S.last_modified,
          CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), NULL
        )

        WHEN NOT MATCHED BY SOURCE
          AND T.username = @username
          AND T.removed_at IS NULL
        THEN UPDATE SET
          removed_at = CURRENT_TIMESTAMP(),
          updated_at = CURRENT_TIMESTAMP()
        """

        query_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
            ]
        )
        logger.info(f"Running MERGE for '{username}'")
        self.client.query(merge_sql, job_config=query_config).result()

        self.client.delete_table(staging_table, not_found_ok=True)
        logger.info(f"MERGE complete for '{username}'")

    def get_latest_collection(self, username: str) -> Optional[pl.DataFrame]:
        """Return currently active rows for `username` (removed_at IS NULL)."""
        sql = f"""
        SELECT *
        FROM `{self.fq_table}`
        WHERE username = @username
          AND removed_at IS NULL
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
            ]
        )
        pdf = self.client.query(sql, job_config=cfg).to_dataframe()
        if len(pdf) == 0:
            return None
        return pl.from_pandas(pdf)

    def get_all_rows_including_removed(
        self, username: str
    ) -> Optional[pl.DataFrame]:
        """Return every row for `username`, including soft-deleted ones.

        Used by tests and by callers that need to inspect removal history.
        """
        sql = f"""
        SELECT *
        FROM `{self.fq_table}`
        WHERE username = @username
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
            ]
        )
        pdf = self.client.query(sql, job_config=cfg).to_dataframe()
        if len(pdf) == 0:
            return None
        return pl.from_pandas(pdf)

    def get_owned_game_ids(self, username: str) -> Optional[list[int]]:
        """Return game_ids where owned = TRUE and the row is not soft-deleted."""
        df = self.get_latest_collection(username)
        if df is None:
            return None
        return df.filter(pl.col("owned") == True)["game_id"].to_list()

    def delete_user_rows(self, username: str) -> None:
        """Hard-delete every row for `username`. Used by test teardown."""
        sql = f"DELETE FROM `{self.fq_table}` WHERE username = @username"
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("username", "STRING", username),
            ]
        )
        self.client.query(sql, job_config=cfg).result()
