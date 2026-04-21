"""Process raw user collection into a joined, game-universe-aware dataframe.

Outcome-agnostic: produces one unlabeled dataframe per user. Labeling is
applied downstream via src.collection.outcomes.apply_outcome.

Raw collection columns come from the BGG XML API (see CollectionLoader).
The processor normalizes those names into a canonical schema used by the
Outcomes config and the rest of the modeling pipeline.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

from src.collection.collection_storage import CollectionStorage
from src.utils.config import BigQueryConfig, DataWarehouseConfig

logger = logging.getLogger(__name__)


BGG_TO_CANONICAL = {
    "previously_owned": "prev_owned",
}


@dataclass
class ProcessorConfig:
    """Configuration for collection processing."""

    games_only: bool = True
    """If True, filter out non-boardgame subtypes."""

    use_predicted_complexity: bool = False
    """If True, join latest predicted_complexity from
    bgg-data-warehouse.predictions.bgg_complexity_predictions."""

    use_embeddings: bool = False
    """If True, join latest description embeddings from
    bgg-data-warehouse.predictions.bgg_description_embeddings and
    explode into emb_0..emb_N columns."""


class CollectionProcessor:
    """Join raw user collection with game universe features.

    Outcome-agnostic: returns a single unlabeled dataframe containing all
    games the user has any relationship to (owned, prev_owned, rated, etc.),
    joined with the game universe feature set. Labeling is applied by the
    pipeline downstream via apply_outcome().
    """

    def __init__(
        self,
        config: BigQueryConfig,
        environment: str = "dev",
        processor_config: Optional[ProcessorConfig] = None,
    ):
        self.bq_config = config
        self.environment = environment
        self.processor_config = processor_config or ProcessorConfig()
        self.storage = CollectionStorage(environment=environment)

    def process(self, username: str) -> Optional[pl.DataFrame]:
        """Produce the unlabeled, joined dataframe for one user.

        Returns None if the user has no stored collection.
        """
        logger.info(f"Processing collection for user '{username}'")

        collection_df = self.storage.get_latest_collection(username)
        if collection_df is None:
            logger.error(f"No collection found for user '{username}'")
            return None

        collection_df = self._to_canonical(collection_df)

        if self.processor_config.games_only:
            if "subtype" not in collection_df.columns:
                logger.warning(
                    "games_only=True but 'subtype' column missing from collection; "
                    "returning unfiltered collection"
                )
            else:
                before = len(collection_df)
                collection_df = collection_df.filter(pl.col("subtype") == "boardgame")
                logger.info(f"Filtered to boardgames: {before} -> {len(collection_df)}")

        logger.info("Loading game universe features from warehouse")
        features_df = self._load_features()

        logger.info("Joining collection with game features")
        joined = collection_df.join(features_df, on="game_id", how="left", suffix="_features")

        logger.info(
            f"Processed {len(joined)} rows × {len(joined.columns)} columns for '{username}'"
        )
        return joined

    def _load_features(self) -> pl.DataFrame:
        """Load the game universe feature set, optionally enriched with
        predicted_complexity and/or description embeddings.

        Reads from the data warehouse project/dataset configured on
        ``self.bq_config``. When enrichment flags on ``self.processor_config``
        are set, LEFT JOINs the latest-per-game row from the predictions
        datasets and (for embeddings) explodes the ``embedding`` list column
        into ``emb_0..emb_{N-1}`` columns.
        """
        project_id, dataset = self._dw_project_and_dataset()
        features_table = self.bq_config.features_table if isinstance(
            self.bq_config, DataWarehouseConfig
        ) else self.bq_config.table

        cfg = self.processor_config

        # Build CTEs dynamically based on enrichment flags.
        ctes = [
            f"""base AS (
  SELECT *
  FROM `{project_id}.{dataset}.{features_table}`
  WHERE year_published IS NOT NULL
)"""
        ]
        select_extra_cols = []
        joins = []

        if cfg.use_predicted_complexity:
            ctes.append(
                f"""complexity AS (
  SELECT game_id, predicted_complexity
  FROM (
    SELECT
      game_id,
      predicted_complexity,
      ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY score_ts DESC) AS rn
    FROM `{project_id}.predictions.bgg_complexity_predictions`
  )
  WHERE rn = 1
)"""
            )
            select_extra_cols.append("complexity.predicted_complexity")
            joins.append("LEFT JOIN complexity USING (game_id)")

        if cfg.use_embeddings:
            ctes.append(
                f"""embeddings AS (
  SELECT game_id, embedding
  FROM (
    SELECT
      game_id,
      embedding,
      ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY created_ts DESC) AS rn
    FROM `{project_id}.predictions.bgg_description_embeddings`
  )
  WHERE rn = 1
)"""
            )
            select_extra_cols.append("embeddings.embedding")
            joins.append("LEFT JOIN embeddings USING (game_id)")

        select_cols = "base.*"
        if select_extra_cols:
            select_cols = select_cols + ", " + ", ".join(select_extra_cols)

        sql_parts = ["WITH " + ",\n".join(ctes)]
        sql_parts.append(f"SELECT {select_cols}")
        sql_parts.append("FROM base")
        sql_parts.extend(joins)
        sql = "\n".join(sql_parts)

        logger.info("Executing feature universe query")
        client = self.bq_config.get_client()
        pandas_df = client.query(sql).to_dataframe()

        df = pl.from_pandas(pandas_df)

        if cfg.use_embeddings:
            df = self._explode_embeddings(df)

        logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")
        return df

    def _dw_project_and_dataset(self) -> tuple[str, str]:
        """Return ``(project_id, dataset)`` for the game universe features
        table, handling both DataWarehouseConfig and legacy BigQueryConfig.
        """
        if isinstance(self.bq_config, DataWarehouseConfig):
            return self.bq_config.project_id, self.bq_config.features_dataset
        return self.bq_config.project_id, self.bq_config.dataset

    @staticmethod
    def _explode_embeddings(df: pl.DataFrame) -> pl.DataFrame:
        """Explode a list-typed ``embedding`` column into ``emb_0..emb_{N-1}``.

        If ``embedding`` is absent or entirely null, drop it (if present) and
        return the frame unchanged otherwise.
        """
        if "embedding" not in df.columns:
            return df

        # Find dimension from the first non-null embedding.
        dim: Optional[int] = None
        for value in df["embedding"].to_list():
            if value is not None and len(value) > 0:
                dim = len(value)
                break

        if dim is None:
            # Nothing to explode; drop the empty list column.
            return df.drop("embedding")

        # Build emb_0..emb_{dim-1} by materializing via numpy (tolerates nulls
        # by filling with NaN) and concatenating.
        embeddings = df["embedding"].to_list()
        matrix = np.full((len(embeddings), dim), np.nan, dtype=float)
        for i, vec in enumerate(embeddings):
            if vec is not None and len(vec) == dim:
                matrix[i, :] = vec

        emb_cols = {f"emb_{i}": matrix[:, i] for i in range(dim)}
        emb_df = pl.DataFrame(emb_cols)

        return df.drop("embedding").hstack(emb_df)

    @staticmethod
    def _to_canonical(df: pl.DataFrame) -> pl.DataFrame:
        """Rename raw BGG column names to the canonical names used downstream."""
        rename_map = {src: dst for src, dst in BGG_TO_CANONICAL.items() if src in df.columns}
        return df.rename(rename_map) if rename_map else df
