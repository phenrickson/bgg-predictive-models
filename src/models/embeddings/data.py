"""Data loading for embedding models."""

import logging
from typing import Optional

import numpy as np
import polars as pl
from google.cloud import bigquery

from src.utils.config import Config, load_config

logger = logging.getLogger(__name__)


def _explode_embeddings(df: pl.DataFrame) -> pl.DataFrame:
    """Explode the ``embedding`` list column into ``emb_0..emb_{N-1}`` columns."""
    if "embedding" not in df.columns:
        return df

    sample = None
    for emb in df["embedding"].to_list():
        if emb is not None and len(emb) > 0:
            sample = emb
            break

    if sample is None:
        raise ValueError("No valid embeddings found in data")

    dim = len(sample)
    matrix = np.vstack(df["embedding"].to_list())
    emb_cols = {f"emb_{i}": matrix[:, i] for i in range(dim)}
    return df.drop("embedding").with_columns(
        [pl.Series(name=k, values=v) for k, v in emb_cols.items()]
    )


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

    def _build_query(
        self,
        where_clause: str,
        use_embeddings: bool,
    ) -> str:
        """Build the BigQuery SQL for loading game features.

        Always INNER JOINs latest complexity predictions. When ``use_embeddings``
        is True, also INNER JOINs the latest description-embedding row per game.
        """
        ctes = [
            f"""latest_complexity AS (
                SELECT
                    game_id,
                    predicted_complexity,
                    ROW_NUMBER() OVER (
                        PARTITION BY game_id
                        ORDER BY score_ts DESC
                    ) as rn
                FROM `{self.dw_project}.predictions.bgg_complexity_predictions`
            )"""
        ]
        select_extra = ["lc.predicted_complexity"]
        joins = ["INNER JOIN latest_complexity lc ON gf.game_id = lc.game_id AND lc.rn = 1"]

        if use_embeddings:
            ctes.append(
                f"""latest_description_emb AS (
                    SELECT
                        game_id,
                        embedding,
                        ROW_NUMBER() OVER (
                            PARTITION BY game_id
                            ORDER BY created_ts DESC
                        ) as rn
                    FROM `{self.dw_project}.predictions.bgg_description_embeddings`
                )"""
            )
            select_extra.append("lde.embedding")
            joins.append(
                "INNER JOIN latest_description_emb lde "
                "ON gf.game_id = lde.game_id AND lde.rn = 1"
            )

        return (
            "WITH " + ",\n".join(ctes) + "\n"
            "SELECT gf.*, " + ", ".join(select_extra) + "\n"
            f"FROM `{self.dw_project}.{self.dw_dataset}.{self.dw_table}` gf\n"
            + "\n".join(joins) + "\n"
            f"WHERE {where_clause}"
        )

    def load_embedding_data(
        self,
        end_year: Optional[int] = None,
        min_ratings: int = 0,
        use_embeddings: bool = False,
    ) -> pl.DataFrame:
        """Load game features joined with complexity predictions.

        Args:
            end_year: Maximum year_published to include. If None, uses config.
            min_ratings: Minimum users_rated threshold.
            use_embeddings: If True, also INNER JOIN latest description embeddings
                and explode the embedding array into ``emb_0..emb_{N-1}`` columns.

        Returns:
            DataFrame with game features and predicted_complexity (and optionally
            ``emb_*`` columns).
        """
        if end_year is None:
            end_year = self.config.years.training.test_through

        where_clause = (
            f"gf.year_published IS NOT NULL "
            f"AND gf.year_published <= {end_year} "
            f"AND gf.users_rated >= {min_ratings}"
        )
        query = self._build_query(where_clause, use_embeddings=use_embeddings)

        logger.info(
            f"Loading embedding data for years <= {end_year} "
            f"(use_embeddings={use_embeddings})"
        )

        try:
            result = self.client.query(query).to_dataframe()
            df = pl.from_pandas(result)

            if use_embeddings:
                df = _explode_embeddings(df)

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
        use_embeddings: bool = False,
    ) -> pl.DataFrame:
        """Load data for scoring/inference (generating embeddings for games).

        Args:
            game_ids: Optional list of game IDs to load. If None, loads all games
                      with complexity predictions.
            use_embeddings: If True, also INNER JOIN latest description embeddings
                and explode the embedding array into ``emb_0..emb_{N-1}`` columns.

        Returns:
            DataFrame with game features and predicted_complexity (and optionally
            ``emb_*`` columns).
        """
        if game_ids:
            ids_str = ",".join(str(g) for g in game_ids)
            where_clause = f"gf.year_published IS NOT NULL AND gf.game_id IN ({ids_str})"
            log_msg = f"Loading scoring data for {len(game_ids)} specific game IDs"
        else:
            where_clause = "gf.year_published IS NOT NULL"
            log_msg = "Loading scoring data for all games with complexity predictions"

        query = self._build_query(where_clause, use_embeddings=use_embeddings)

        logger.info(f"{log_msg} (use_embeddings={use_embeddings})")

        try:
            result = self.client.query(query).to_dataframe()
            df = pl.from_pandas(result)

            if use_embeddings:
                df = _explode_embeddings(df)

            logger.info(f"Loaded {len(df)} games for scoring")
            return df

        except Exception as e:
            logger.error(f"Error loading scoring data: {e}")
            raise
