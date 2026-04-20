"""Process raw user collection into a joined, game-universe-aware dataframe.

Outcome-agnostic: produces one unlabeled dataframe per user. Labeling is
applied downstream via src.collection.outcomes.apply_outcome.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import polars as pl

from src.collection.collection_storage import CollectionStorage
from src.data.loader import BGGDataLoader
from src.utils.config import BigQueryConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for collection processing."""
    games_only: bool = True
    """If True, filter out non-boardgame subtypes."""


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
        self.data_loader = BGGDataLoader(config)
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

        if self.processor_config.games_only and "subtype" in collection_df.columns:
            before = len(collection_df)
            collection_df = collection_df.filter(pl.col("subtype") == "boardgame")
            logger.info(f"Filtered to boardgames: {before} -> {len(collection_df)}")

        logger.info("Loading game universe features from warehouse")
        features_df = self.data_loader.load_data()

        logger.info("Joining collection with game features")
        joined = collection_df.join(features_df, on="game_id", how="left", suffix="_features")

        logger.info(
            f"Processed {len(joined)} rows × {len(joined.columns)} columns for '{username}'"
        )
        return joined
