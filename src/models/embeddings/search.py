"""Nearest neighbor search using BigQuery Vector Search."""

import logging
from typing import List, Optional

import polars as pl
from google.cloud import bigquery

from src.utils.config import Config, load_config

logger = logging.getLogger(__name__)


class NearestNeighborSearch:
    """Query interface for BigQuery VECTOR_SEARCH."""

    def __init__(
        self,
        config: Optional[Config] = None,
        table_id: Optional[str] = None,
    ):
        """Initialize nearest neighbor search.

        Args:
            config: Configuration object.
            table_id: Full BigQuery table ID. If None, uses config.
        """
        self.config = config or load_config()

        if table_id:
            self.table_id = table_id
        elif self.config.embeddings:
            project = self.config.ml_project_id
            dataset = self.config.embeddings.vector_search.dataset
            table = self.config.embeddings.vector_search.table
            self.table_id = f"{project}.{dataset}.{table}"
        else:
            self.table_id = f"{self.config.ml_project_id}.raw.game_embeddings"

        self.client = bigquery.Client(project=self.config.ml_project_id)

    def find_similar_games(
        self,
        game_id: int,
        top_k: int = 10,
        distance_type: str = "COSINE",
        exclude_self: bool = True,
        model_version: Optional[int] = None,
    ) -> pl.DataFrame:
        """Find k nearest neighbors for a game.

        Args:
            game_id: Source game ID.
            top_k: Number of similar games to return.
            distance_type: COSINE, EUCLIDEAN, or DOT_PRODUCT.
            exclude_self: Whether to exclude the source game.
            model_version: Specific version to use. If None, uses latest.

        Returns:
            DataFrame with game_id, name, year_published, distance.
        """
        # Build version filter
        if model_version:
            version_filter = f"embedding_version = {model_version}"
        else:
            version_filter = f"""
            embedding_version = (
                SELECT MAX(embedding_version) FROM `{self.table_id}`
            )
            """

        # Query to find similar games
        query = f"""
        WITH source_game AS (
            SELECT embedding, game_id as source_game_id
            FROM `{self.table_id}`
            WHERE game_id = @game_id AND {version_filter}
            LIMIT 1
        ),
        candidates AS (
            SELECT game_id, name, year_published, embedding
            FROM `{self.table_id}`
            WHERE {version_filter}
        )
        SELECT
            c.game_id,
            c.name,
            c.year_published,
            ML.DISTANCE(c.embedding, s.embedding, '{distance_type}') as distance
        FROM candidates c
        CROSS JOIN source_game s
        WHERE c.game_id != s.source_game_id OR NOT @exclude_self
        ORDER BY distance ASC
        LIMIT @top_k
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("game_id", "INT64", game_id),
                bigquery.ScalarQueryParameter("top_k", "INT64", top_k + (1 if exclude_self else 0)),
                bigquery.ScalarQueryParameter("exclude_self", "BOOL", exclude_self),
            ]
        )

        try:
            result = self.client.query(query, job_config=job_config).to_dataframe()

            # Filter out self if needed
            if exclude_self:
                result = result[result["game_id"] != game_id].head(top_k)

            return pl.from_pandas(result)

        except Exception as e:
            logger.error(f"Error finding similar games: {e}")
            raise

    def find_similar_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 10,
        distance_type: str = "COSINE",
        model_version: Optional[int] = None,
    ) -> pl.DataFrame:
        """Find similar games given an embedding vector.

        Args:
            embedding: Embedding vector to search with.
            top_k: Number of results to return.
            distance_type: COSINE, EUCLIDEAN, or DOT_PRODUCT.
            model_version: Specific version to use.

        Returns:
            DataFrame with game_id, name, year_published, distance.
        """
        # Build version filter
        if model_version:
            version_filter = f"embedding_version = {model_version}"
        else:
            version_filter = f"""
            embedding_version = (
                SELECT MAX(embedding_version) FROM `{self.table_id}`
            )
            """

        # Convert embedding to string for SQL
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        query = f"""
        WITH query_embedding AS (
            SELECT {embedding_str} as embedding
        ),
        candidates AS (
            SELECT game_id, name, year_published, embedding
            FROM `{self.table_id}`
            WHERE {version_filter}
        )
        SELECT
            c.game_id,
            c.name,
            c.year_published,
            ML.DISTANCE(c.embedding, q.embedding, '{distance_type}') as distance
        FROM candidates c
        CROSS JOIN query_embedding q
        ORDER BY distance ASC
        LIMIT @top_k
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
            ]
        )

        try:
            result = self.client.query(query, job_config=job_config).to_dataframe()
            return pl.from_pandas(result)
        except Exception as e:
            logger.error(f"Error searching by embedding: {e}")
            raise

    def find_games_like(
        self,
        game_ids: List[int],
        top_k: int = 10,
        distance_type: str = "COSINE",
        model_version: Optional[int] = None,
    ) -> pl.DataFrame:
        """Find games similar to a set of games (using average embedding).

        Args:
            game_ids: List of game IDs to base search on.
            top_k: Number of results to return.
            distance_type: Distance metric.
            model_version: Specific version to use.

        Returns:
            DataFrame with similar games.
        """
        # Build version filter
        if model_version:
            version_filter = f"embedding_version = {model_version}"
        else:
            version_filter = f"""
            embedding_version = (
                SELECT MAX(embedding_version) FROM `{self.table_id}`
            )
            """

        game_ids_str = ",".join(str(g) for g in game_ids)

        query = f"""
        WITH source_games AS (
            SELECT embedding
            FROM `{self.table_id}`
            WHERE game_id IN ({game_ids_str}) AND {version_filter}
        ),
        avg_embedding AS (
            SELECT ARRAY_AGG(e) as embedding
            FROM source_games,
            UNNEST(embedding) as e WITH OFFSET pos
            GROUP BY pos
            ORDER BY pos
        ),
        query_embedding AS (
            SELECT ARRAY(SELECT AVG(e) FROM UNNEST(embedding) as e) as embedding
            FROM avg_embedding
        ),
        candidates AS (
            SELECT game_id, name, year_published, embedding
            FROM `{self.table_id}`
            WHERE {version_filter}
              AND game_id NOT IN ({game_ids_str})
        )
        SELECT
            c.game_id,
            c.name,
            c.year_published,
            ML.DISTANCE(c.embedding, q.embedding, '{distance_type}') as distance
        FROM candidates c
        CROSS JOIN query_embedding q
        ORDER BY distance ASC
        LIMIT @top_k
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
            ]
        )

        try:
            result = self.client.query(query, job_config=job_config).to_dataframe()
            return pl.from_pandas(result)
        except Exception as e:
            logger.error(f"Error finding games like set: {e}")
            raise

    def get_embedding_stats(self, model_version: Optional[int] = None) -> dict:
        """Get statistics about stored embeddings.

        Args:
            model_version: Specific version to check.

        Returns:
            Dictionary with count, versions, etc.
        """
        query = f"""
        SELECT
            COUNT(*) as total_embeddings,
            COUNT(DISTINCT game_id) as unique_games,
            COUNT(DISTINCT embedding_version) as versions,
            MIN(created_ts) as first_created,
            MAX(created_ts) as last_created,
            MAX(embedding_version) as latest_version
        FROM `{self.table_id}`
        """

        if model_version:
            query += f" WHERE embedding_version = {model_version}"

        result = self.client.query(query).to_dataframe()
        return result.iloc[0].to_dict()
