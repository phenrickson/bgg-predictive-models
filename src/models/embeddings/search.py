"""Nearest neighbor search using BigQuery Vector Search."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import polars as pl
from google.cloud import bigquery

from src.utils.config import Config, load_config

logger = logging.getLogger(__name__)


@dataclass
class SearchFilters:
    """Filters for similarity search results."""

    min_year: Optional[int] = None
    max_year: Optional[int] = None
    min_users_rated: Optional[int] = None
    max_users_rated: Optional[int] = None
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    min_geek_rating: Optional[float] = None
    max_geek_rating: Optional[float] = None
    min_complexity: Optional[float] = None
    max_complexity: Optional[float] = None

    def has_filters(self) -> bool:
        """Check if any filters are set."""
        return any([
            self.min_year, self.max_year,
            self.min_users_rated, self.max_users_rated,
            self.min_rating, self.max_rating,
            self.min_geek_rating, self.max_geek_rating,
            self.min_complexity, self.max_complexity,
        ])


# Valid embedding dimension options
VALID_EMBEDDING_DIMS = [8, 16, 32, 64]


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
            project = self.config.embeddings.vector_search.project or self.config.ml_project_id
            dataset = self.config.embeddings.vector_search.dataset
            table = self.config.embeddings.vector_search.table
            self.table_id = f"{project}.{dataset}.{table}"
        else:
            self.table_id = f"{self.config.ml_project_id}.raw.game_embeddings"

        self.client = bigquery.Client(project=self.config.ml_project_id)

    def _get_embedding_column(self, embedding_dims: Optional[int] = None) -> str:
        """Get the embedding column name for the requested dimensions.

        Args:
            embedding_dims: Number of dimensions (8, 16, 32, or 64/None for full).

        Returns:
            Column name to use for embeddings.
        """
        if embedding_dims is None or embedding_dims == 64:
            return "embedding"
        if embedding_dims not in VALID_EMBEDDING_DIMS:
            raise ValueError(
                f"Invalid embedding_dims: {embedding_dims}. "
                f"Must be one of {VALID_EMBEDDING_DIMS}"
            )
        return f"embedding_{embedding_dims}"

    def _build_filter_clause(self, filters: Optional[SearchFilters]) -> str:
        """Build SQL WHERE clause from filters."""
        if not filters or not filters.has_filters():
            return ""

        conditions = []
        if filters.min_year is not None:
            conditions.append(f"year_published >= {filters.min_year}")
        if filters.max_year is not None:
            conditions.append(f"year_published <= {filters.max_year}")
        if filters.min_users_rated is not None:
            conditions.append(f"users_rated >= {filters.min_users_rated}")
        if filters.max_users_rated is not None:
            conditions.append(f"users_rated <= {filters.max_users_rated}")
        if filters.min_rating is not None:
            conditions.append(f"average_rating >= {filters.min_rating}")
        if filters.max_rating is not None:
            conditions.append(f"average_rating <= {filters.max_rating}")
        if filters.min_geek_rating is not None:
            conditions.append(f"geek_rating >= {filters.min_geek_rating}")
        if filters.max_geek_rating is not None:
            conditions.append(f"geek_rating <= {filters.max_geek_rating}")
        if filters.min_complexity is not None:
            conditions.append(f"complexity >= {filters.min_complexity}")
        if filters.max_complexity is not None:
            conditions.append(f"complexity <= {filters.max_complexity}")

        return " AND " + " AND ".join(conditions) if conditions else ""

    def find_similar_games(
        self,
        game_id: int,
        top_k: int = 10,
        distance_type: str = "COSINE",
        exclude_self: bool = True,
        model_version: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        embedding_dims: Optional[int] = None,
    ) -> pl.DataFrame:
        """Find k nearest neighbors for a game.

        Args:
            game_id: Source game ID.
            top_k: Number of similar games to return.
            distance_type: COSINE, EUCLIDEAN, or DOT_PRODUCT.
            exclude_self: Whether to exclude the source game.
            model_version: Specific version to use. If None, uses latest.
            filters: Optional filters for year, rating, complexity, etc.
            embedding_dims: Number of embedding dimensions to use (8, 16, 32, or 64/None).

        Returns:
            DataFrame with game_id, name, year_published, distance, and filter fields.
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

        # Build filter clause
        filter_clause = self._build_filter_clause(filters)

        # Get embedding column based on requested dimensions
        emb_col = self._get_embedding_column(embedding_dims)

        # Query to find similar games
        query = f"""
        WITH source_game AS (
            SELECT {emb_col} as embedding, game_id as source_game_id
            FROM `{self.table_id}`
            WHERE game_id = @game_id AND {version_filter}
            LIMIT 1
        ),
        candidates AS (
            SELECT game_id, name, year_published, {emb_col} as embedding,
                   users_rated, average_rating, geek_rating, complexity, thumbnail
            FROM `{self.table_id}`
            WHERE {version_filter}{filter_clause}
        )
        SELECT
            c.game_id,
            c.name,
            c.year_published,
            c.users_rated,
            c.average_rating,
            c.geek_rating,
            c.complexity,
            c.thumbnail,
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
        embedding_dims: Optional[int] = None,
    ) -> pl.DataFrame:
        """Find similar games given an embedding vector.

        Args:
            embedding: Embedding vector to search with.
            top_k: Number of results to return.
            distance_type: COSINE, EUCLIDEAN, or DOT_PRODUCT.
            model_version: Specific version to use.
            embedding_dims: Number of embedding dimensions to use. If None, inferred from embedding length.

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

        # Infer embedding_dims from embedding length if not provided
        if embedding_dims is None:
            embedding_dims = len(embedding) if len(embedding) in VALID_EMBEDDING_DIMS else 64

        # Get embedding column based on dimensions
        emb_col = self._get_embedding_column(embedding_dims)

        # Convert embedding to string for SQL
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        query = f"""
        WITH query_embedding AS (
            SELECT {embedding_str} as embedding
        ),
        candidates AS (
            SELECT game_id, name, year_published, {emb_col} as embedding
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
        filters: Optional[SearchFilters] = None,
        embedding_dims: Optional[int] = None,
    ) -> pl.DataFrame:
        """Find games similar to a set of games (using average embedding).

        Args:
            game_ids: List of game IDs to base search on.
            top_k: Number of results to return.
            distance_type: Distance metric.
            model_version: Specific version to use.
            filters: Optional filters for year, rating, complexity, etc.
            embedding_dims: Number of embedding dimensions to use (8, 16, 32, or 64/None).

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

        # Build filter clause
        filter_clause = self._build_filter_clause(filters)

        # Get embedding column based on requested dimensions
        emb_col = self._get_embedding_column(embedding_dims)

        game_ids_str = ",".join(str(g) for g in game_ids)

        query = f"""
        WITH source_games AS (
            SELECT {emb_col} as embedding
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
            SELECT game_id, name, year_published, {emb_col} as embedding,
                   users_rated, average_rating, geek_rating, complexity, thumbnail
            FROM `{self.table_id}`
            WHERE {version_filter}
              AND game_id NOT IN ({game_ids_str}){filter_clause}
        )
        SELECT
            c.game_id,
            c.name,
            c.year_published,
            c.users_rated,
            c.average_rating,
            c.geek_rating,
            c.complexity,
            c.thumbnail,
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
