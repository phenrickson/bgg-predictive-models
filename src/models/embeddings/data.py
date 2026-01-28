"""Data loading for embedding models."""

import logging
from typing import Optional

import polars as pl
from google.cloud import bigquery

from src.utils.config import Config, load_config

logger = logging.getLogger(__name__)


class EmbeddingDataLoader:
    """Loads game features with complexity predictions for embedding training.

    Queries BigQuery directly to join games_features with complexity_predictions,
    eliminating the need for local parquet files.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the embedding data loader.

        Args:
            config: Configuration object. If None, loads from config.yaml.
        """
        self.config = config or load_config()

        # Data warehouse config (games_features and complexity_predictions source)
        self.dw_project = self.config.data_warehouse.project_id
        self.dw_dataset = self.config.data_warehouse.features_dataset
        self.dw_table = self.config.data_warehouse.features_table

        # Use data warehouse project for running queries (has job permissions)
        self.client = bigquery.Client(project=self.dw_project)

    def load_embedding_data(
        self,
        end_year: Optional[int] = None,
        min_ratings: int = 0,
    ) -> pl.DataFrame:
        """Load game features joined with complexity predictions.

        Args:
            end_year: Maximum year_published to include. If None, uses config.
            min_ratings: Minimum users_rated threshold.

        Returns:
            DataFrame with game features and predicted_complexity.
        """
        if end_year is None:
            end_year = self.config.years.training.test_through

        # Query that joins games_features with latest complexity predictions
        # Complexity predictions are in bgg-data-warehouse.predictions
        query = f"""
        WITH latest_complexity AS (
            SELECT
                game_id,
                predicted_complexity,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY score_ts DESC
                ) as rn
            FROM `{self.dw_project}.predictions.bgg_complexity_predictions`
        )
        SELECT
            gf.*,
            lc.predicted_complexity
        FROM `{self.dw_project}.{self.dw_dataset}.{self.dw_table}` gf
        INNER JOIN latest_complexity lc
            ON gf.game_id = lc.game_id
            AND lc.rn = 1
        WHERE gf.year_published IS NOT NULL
            AND gf.year_published <= {end_year}
            AND gf.users_rated >= {min_ratings}
        """

        logger.info(f"Loading embedding data for years <= {end_year}")

        try:
            result = self.client.query(query).to_dataframe()
            df = pl.from_pandas(result)

            logger.info(
                f"Loaded {len(df)} games with complexity predictions, "
                f"year range: {df['year_published'].min()}-{df['year_published'].max()}"
            )

            return df

        except Exception as e:
            logger.error(f"Error loading embedding data: {e}")
            raise

    def load_scoring_data(
        self,
        game_ids: Optional[list] = None,
    ) -> pl.DataFrame:
        """Load data for scoring/inference (generating embeddings for games).

        Args:
            game_ids: Optional list of game IDs to load. If None, loads all games
                      with complexity predictions.

        Returns:
            DataFrame with game features and predicted_complexity.
        """
        if game_ids:
            ids_str = ",".join(str(g) for g in game_ids)
            where_clause = f"gf.year_published IS NOT NULL AND gf.game_id IN ({ids_str})"
            log_msg = f"Loading scoring data for {len(game_ids)} specific game IDs"
        else:
            where_clause = "gf.year_published IS NOT NULL"
            log_msg = "Loading scoring data for all games with complexity predictions"

        query = f"""
        WITH latest_complexity AS (
            SELECT
                game_id,
                predicted_complexity,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY score_ts DESC
                ) as rn
            FROM `{self.dw_project}.predictions.bgg_complexity_predictions`
        )
        SELECT
            gf.*,
            lc.predicted_complexity
        FROM `{self.dw_project}.{self.dw_dataset}.{self.dw_table}` gf
        INNER JOIN latest_complexity lc
            ON gf.game_id = lc.game_id
            AND lc.rn = 1
        WHERE {where_clause}
        """

        logger.info(log_msg)

        try:
            result = self.client.query(query).to_dataframe()
            df = pl.from_pandas(result)

            logger.info(f"Loaded {len(df)} games for scoring")
            return df

        except Exception as e:
            logger.error(f"Error loading scoring data: {e}")
            raise
