"""Process raw user collection into a canonical-schema dataframe.

Outcome-agnostic: produces one unlabeled, user-only dataframe per user
(no game features). Game features are loaded separately via
:meth:`src.data.loader.BGGDataLoader.load_features` and joined later in
the pipeline; this processor only normalizes the user's collection
columns.

Raw collection columns come from the BGG XML API (see CollectionLoader).
The processor normalizes those names into a canonical schema used by the
Outcomes config and the rest of the modeling pipeline.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import polars as pl

from src.collection.collection_storage import CollectionStorage
from src.utils.config import BigQueryConfig

logger = logging.getLogger(__name__)


BGG_TO_CANONICAL = {
    "previously_owned": "prev_owned",
}


@dataclass
class ProcessorConfig:
    """Configuration for collection processing."""

    games_only: bool = True
    """If True, filter out non-boardgame subtypes."""


class CollectionProcessor:
    """Normalize a user's stored collection into canonical schema.

    Outcome-agnostic: returns the user's collection rows with canonical
    column names and (optionally) filtered to boardgame subtypes. Game
    features are not joined here — the caller joins with the universe
    (loaded via :class:`BGGDataLoader`) before labeling and splitting.
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

        logger.info(
            f"Processed {len(collection_df)} rows × {len(collection_df.columns)} columns for '{username}'"
        )
        return collection_df

    @staticmethod
    def _to_canonical(df: pl.DataFrame) -> pl.DataFrame:
        """Rename raw BGG column names to the canonical names used downstream."""
        rename_map = {src: dst for src, dst in BGG_TO_CANONICAL.items() if src in df.columns}
        return df.rename(rename_map) if rename_map else df
