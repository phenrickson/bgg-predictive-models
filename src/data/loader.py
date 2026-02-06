"""Data loading for BGG predictive models."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from src.utils.config import BigQueryConfig, DataWarehouseConfig

logger = logging.getLogger(__name__)


class BGGDataLoader:
    """Simple loader for BGG data from BigQuery data warehouse."""

    def __init__(self, config: Union[DataWarehouseConfig, BigQueryConfig]):
        """Initialize loader.

        Args:
            config: Data warehouse configuration (or legacy BigQueryConfig for
                backwards compatibility)
        """
        self.config = config
        self.client = config.get_client()

        # Handle both new DataWarehouseConfig and legacy BigQueryConfig
        if isinstance(config, DataWarehouseConfig):
            self.project_id = config.project_id
            self.dataset = config.features_dataset
            self.table = config.features_table
        else:
            # BigQueryConfig
            self.project_id = config.project_id
            self.dataset = config.dataset
            self.table = config.table

    def load_data(
        self,
        where_clause: str = "",
        preprocessor=None,
        timeout: int = 300,
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, Dict[str, pl.Series]]]:
        """Load data from BigQuery warehouse.

        Args:
            where_clause: Optional WHERE clause for the SQL query
            preprocessor: Optional preprocessor to transform data
            timeout: Timeout in seconds for BigQuery query execution (default: 300)

        Returns:
            If preprocessor is None:
                Raw DataFrame from BigQuery
            If preprocessor is provided:
                Tuple of (features_df, target_dict)
        """
        # Build query with optional WHERE clause
        query = f"""
        SELECT * FROM `{self.project_id}.{self.dataset}.{self.table}`
        """

        # Add year_published IS NOT NULL filter
        if not where_clause:
            query += " WHERE year_published IS NOT NULL"
        else:
            query += f" WHERE ({where_clause}) AND year_published IS NOT NULL"

        # Execute query and convert to polars DataFrame
        print(f"Executing query: {query}")
        try:
            query_job = self.client.query(query)
            print("Query job created successfully")

            # Check job status and get results
            query_job.result(timeout=timeout)  # Wait for the query to complete
            print("Query job completed successfully")

            pandas_df = query_job.to_dataframe()
            df = pl.from_pandas(pandas_df)

            print(f"Retrieved {len(df)} rows with {len(df.columns)} columns")
            return df

        except Exception as e:
            print(f"Error executing query: {str(e)}")
            # If possible, print additional job error details
            if hasattr(e, "errors"):
                print("Query Errors:")
                for error in e.errors:
                    print(error)
            raise

        # Apply preprocessor if provided
        if preprocessor is not None:
            return preprocessor.fit_transform(df)

        # Otherwise return raw data
        return df

    def load_training_data(
        self,
        end_train_year: int = 2021,
        min_ratings: int = 25,
        min_weights: Optional[float] = None,
        preprocessor=None,
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, Dict[str, pl.Series]]]:
        """Load training data from warehouse.

        Args:
            end_train_year: Last year to include in training
            min_ratings: Minimum number of ratings threshold
            min_weights: Optional minimum game complexity weight threshold
            preprocessor: Optional preprocessor to transform data

        Returns:
            If preprocessor is None:
                Raw DataFrame from BigQuery
            If preprocessor is provided:
                Tuple of (features_df, target_dict)
        """
        where_clauses = [f"year_published <= {end_train_year}"]

        # Add min_ratings filter
        where_clauses.append(f"users_rated >= {min_ratings}")

        if min_weights is not None:
            where_clauses.append(f"num_weights >= {min_weights}")

        where_clause = " AND ".join(where_clauses)
        return self.load_data(where_clause, preprocessor)

    def load_data_with_embeddings(
        self,
        where_clause: str = "",
        embeddings_table: Optional[str] = None,
        timeout: int = 300,
    ) -> pl.DataFrame:
        """Load data joined with description embeddings from BigQuery.

        Joins the features table with the embeddings table and expands
        the embedding array into individual emb_0..emb_N columns.

        Args:
            where_clause: Optional WHERE clause (use 'f.' prefix for features
                table columns, e.g. "f.year_published >= 2025")
            embeddings_table: Full BigQuery table path for embeddings.
                Defaults to {project_id}.predictions.bgg_description_embeddings
            timeout: Timeout in seconds for BigQuery query execution

        Returns:
            Polars DataFrame with features and expanded embedding columns.
        """
        if embeddings_table is None:
            embeddings_table = (
                f"{self.project_id}.predictions.bgg_description_embeddings"
            )

        features_table = f"{self.project_id}.{self.dataset}.{self.table}"

        query = f"""
        SELECT f.*, e.embedding
        FROM `{features_table}` f
        INNER JOIN `{embeddings_table}` e
            ON f.game_id = e.game_id
        """

        if not where_clause:
            query += " WHERE f.year_published IS NOT NULL"
        else:
            query += f" WHERE ({where_clause}) AND f.year_published IS NOT NULL"

        logger.info(f"Loading features with embeddings: {where_clause or 'no filter'}")
        try:
            query_job = self.client.query(query)
            query_job.result(timeout=timeout)
            pandas_df = query_job.to_dataframe()

            if len(pandas_df) == 0:
                raise ValueError("No data returned from BigQuery query")

            logger.info(f"Retrieved {len(pandas_df)} rows from BigQuery")

            # Expand embedding array into individual columns
            pandas_df = self._expand_embedding_column(pandas_df)

            # Convert through polars (normalizes nullable dtypes)
            return pl.from_pandas(pandas_df)

        except Exception as e:
            logger.error(f"Error loading data with embeddings: {str(e)}")
            raise

    @staticmethod
    def _expand_embedding_column(df: pd.DataFrame) -> pd.DataFrame:
        """Expand embedding array column into individual emb_0..emb_N columns.

        Args:
            df: DataFrame with an 'embedding' column containing arrays.

        Returns:
            DataFrame with embedding expanded to emb_0, emb_1, ..., emb_N columns.
        """
        if "embedding" not in df.columns:
            return df

        sample_embedding = None
        for emb in df["embedding"]:
            if emb is not None and len(emb) > 0:
                sample_embedding = emb
                break

        if sample_embedding is None:
            raise ValueError("No valid embeddings found in data")

        embedding_dim = len(sample_embedding)
        logger.info(f"Embedding dimension: {embedding_dim}")

        emb_columns = [f"emb_{i}" for i in range(embedding_dim)]
        embeddings_matrix = np.vstack(df["embedding"].values)
        embeddings_df = pd.DataFrame(
            embeddings_matrix, columns=emb_columns, index=df.index
        )

        df = pd.concat([df.drop(columns=["embedding"]), embeddings_df], axis=1)
        logger.info(f"Expanded embeddings: {len(df)} rows x {embedding_dim} dims")
        return df

    def load_changed_games_with_embeddings(
        self,
        start_year: int,
        end_year: int,
        ml_project_id: str,
        max_games: int = 50000,
        hurdle_model_version: Optional[int] = None,
        complexity_model_version: Optional[int] = None,
        rating_model_version: Optional[int] = None,
        users_rated_model_version: Optional[int] = None,
        embeddings_table: Optional[str] = None,
        timeout: int = 300,
    ) -> pl.DataFrame:
        """Load games needing re-scoring, joined with embeddings.

        Returns games in the year range that either:
        - Have never been scored
        - Have changed features since last scoring
        - Were scored with a different model version

        Args:
            start_year: Start year for predictions (inclusive)
            end_year: End year for predictions (exclusive)
            ml_project_id: GCP project ID for ML predictions landing table
            max_games: Maximum number of games to load
            hurdle_model_version: Target hurdle model version (rescore if different)
            complexity_model_version: Target complexity model version
            rating_model_version: Target rating model version
            users_rated_model_version: Target users_rated model version
            embeddings_table: Full BigQuery table path for embeddings
            timeout: Timeout in seconds for BigQuery query execution

        Returns:
            Polars DataFrame with features and expanded embedding columns.
        """
        if embeddings_table is None:
            embeddings_table = (
                f"{self.project_id}.predictions.bgg_description_embeddings"
            )

        features_table = f"{self.project_id}.{self.dataset}.{self.table}"

        # Build version mismatch conditions
        version_checks = []
        if hurdle_model_version is not None:
            version_checks.append(f"lp.hurdle_model_version != {hurdle_model_version}")
        if complexity_model_version is not None:
            version_checks.append(f"lp.complexity_model_version != {complexity_model_version}")
        if rating_model_version is not None:
            version_checks.append(f"lp.rating_model_version != {rating_model_version}")
        if users_rated_model_version is not None:
            version_checks.append(f"lp.users_rated_model_version != {users_rated_model_version}")

        version_condition = ""
        if version_checks:
            version_condition = "OR " + "\n          OR ".join(version_checks)

        query = f"""
        SELECT f.*, e.embedding
        FROM `{features_table}` f
        INNER JOIN `{embeddings_table}` e
            ON f.game_id = e.game_id
        WHERE f.game_id IN (
          SELECT gf.game_id
          FROM `{self.project_id}.{self.dataset}.{self.table}` gf
          LEFT JOIN `{self.project_id}.staging.game_features_hash` fh
            ON gf.game_id = fh.game_id
          LEFT JOIN (
            SELECT
              game_id,
              score_ts,
              hurdle_model_version,
              complexity_model_version,
              rating_model_version,
              users_rated_model_version,
              ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY score_ts DESC) as rn
            FROM `{ml_project_id}.raw.ml_predictions_landing`
          ) lp ON gf.game_id = lp.game_id AND lp.rn = 1
          WHERE
            gf.year_published IS NOT NULL
            AND gf.year_published >= {start_year}
            AND gf.year_published < {end_year}
            AND (
              lp.game_id IS NULL
              OR fh.last_updated > lp.score_ts
              {version_condition}
            )
          LIMIT {max_games}
        )
        AND f.year_published IS NOT NULL
        """

        logger.info(
            f"Loading changed games with embeddings "
            f"(years {start_year}-{end_year}, max {max_games})..."
        )
        try:
            query_job = self.client.query(query)
            query_job.result(timeout=timeout)
            pandas_df = query_job.to_dataframe()

            logger.info(f"Found {len(pandas_df)} games needing scoring")

            if len(pandas_df) == 0:
                return pl.DataFrame()

            # Expand embedding array into individual columns
            pandas_df = self._expand_embedding_column(pandas_df)

            # Convert through polars (normalizes nullable dtypes)
            return pl.from_pandas(pandas_df)

        except Exception as e:
            logger.error(f"Error loading changed games: {str(e)}")
            raise

    def load_prediction_data(
        self,
        game_ids: Optional[List[int]] = None,
        preprocessor=None,
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, Dict[str, pl.Series]]]:
        """Load data for making predictions.

        Args:
            game_ids: Optional list of specific game IDs to load
            preprocessor: Optional preprocessor to transform data

        Returns:
            If preprocessor is None:
                Raw DataFrame from BigQuery
            If preprocessor is provided:
                Processed feature matrix
        """
        where_clause = ""
        if game_ids:
            where_clause = f"game_id IN ({','.join(map(str, game_ids))})"

        return self.load_data(where_clause, preprocessor)
