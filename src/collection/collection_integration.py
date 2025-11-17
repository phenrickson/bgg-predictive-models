"""Integration layer for combining collection data with game features."""

import logging
from typing import Optional

import polars as pl

from src.collection.collection_storage import CollectionStorage
from src.data.loader import BGGDataLoader
from src.utils.config import BigQueryConfig

logger = logging.getLogger(__name__)


class CollectionIntegration:
    """Integrates user collections with game features for modeling."""

    def __init__(self, config: BigQueryConfig, environment: str = "dev"):
        """Initialize integration layer.

        Args:
            config: BigQuery configuration
            environment: Environment name (dev/prod)
        """
        self.config = config
        self.environment = environment
        self.data_loader = BGGDataLoader(config)
        self.storage = CollectionStorage(environment=environment)

    def get_collection_with_features(
        self,
        username: str,
        owned_only: bool = True,
        games_only: bool = True,
    ) -> Optional[pl.DataFrame]:
        """Get user's collection joined with game features.

        Args:
            username: BGG username
            owned_only: Only include owned games (default: True)
            games_only: Only include boardgames, not expansions (default: True)

        Returns:
            DataFrame with collection data joined to game features
        """
        try:
            # Get latest collection from storage
            logger.info(f"Fetching collection for user '{username}'")
            collection_df = self.storage.get_latest_collection(username)

            if collection_df is None:
                logger.error(f"No collection found for user '{username}'")
                return None

            # Filter collection if needed
            if owned_only:
                collection_df = collection_df.filter(pl.col("owned") == True)
                logger.info(f"Filtered to {len(collection_df)} owned items")

            if games_only:
                collection_df = collection_df.filter(pl.col("subtype") == "boardgame")
                logger.info(f"Filtered to {len(collection_df)} boardgames")

            # Load game features from warehouse
            logger.info("Loading game features from warehouse")
            features_df = self.data_loader.load_data()

            # Join collection with features
            logger.info("Joining collection with game features")
            joined_df = collection_df.join(
                features_df,
                on="game_id",
                how="left",
                suffix="_features",
            )

            logger.info(
                f"Successfully joined collection with features: {len(joined_df)} rows, {len(joined_df.columns)} columns"
            )

            return joined_df

        except Exception as e:
            logger.error(f"Error joining collection with features: {e}")
            import traceback

            traceback.print_exc()
            return None

    def get_collection_training_data(
        self,
        username: str,
        end_train_year: Optional[int] = None,
        min_ratings: int = 25,
    ) -> Optional[pl.DataFrame]:
        """Get collection data suitable for training collection-specific models.

        Args:
            username: BGG username
            end_train_year: Optional cutoff year for training data
            min_ratings: Minimum number of ratings threshold

        Returns:
            DataFrame with collection games that meet training criteria
        """
        try:
            # Get collection with features
            df = self.get_collection_with_features(
                username, owned_only=True, games_only=True
            )

            if df is None:
                return None

            # Apply training filters
            logger.info("Applying training data filters")

            # Filter by year if specified
            if end_train_year is not None:
                df = df.filter(pl.col("year_published") <= end_train_year)
                logger.info(f"Filtered to games published <= {end_train_year}")

            # Filter by minimum ratings
            df = df.filter(pl.col("users_rated") >= min_ratings)
            logger.info(f"Filtered to games with >= {min_ratings} ratings")

            logger.info(f"Training data prepared: {len(df)} games")

            return df

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            import traceback

            traceback.print_exc()
            return None

    def get_missing_games_for_prediction(
        self, username: str, min_year: Optional[int] = None
    ) -> Optional[pl.DataFrame]:
        """Get games NOT in user's collection that could be recommended.

        Args:
            username: BGG username
            min_year: Optional minimum year for games to recommend

        Returns:
            DataFrame with games not in collection
        """
        try:
            # Get user's owned game IDs
            owned_game_ids = self.storage.get_owned_game_ids(username)

            if owned_game_ids is None:
                logger.error(f"Could not get owned games for user '{username}'")
                return None

            logger.info(f"User owns {len(owned_game_ids)} games")

            # Load all game features
            logger.info("Loading all game features")
            features_df = self.data_loader.load_data()

            # Filter to games NOT owned
            missing_df = features_df.filter(~pl.col("game_id").is_in(owned_game_ids))
            logger.info(f"Found {len(missing_df)} games not in collection")

            # Filter by year if specified
            if min_year is not None:
                missing_df = missing_df.filter(pl.col("year_published") >= min_year)
                logger.info(f"Filtered to games published >= {min_year}")

            return missing_df

        except Exception as e:
            logger.error(f"Error getting missing games: {e}")
            import traceback

            traceback.print_exc()
            return None

    def get_collection_summary_stats(self, username: str) -> Optional[dict]:
        """Get summary statistics about user's collection relative to BGG dataset.

        Args:
            username: BGG username

        Returns:
            Dictionary with summary statistics
        """
        try:
            # Get collection with features
            df = self.get_collection_with_features(
                username, owned_only=True, games_only=True
            )

            if df is None:
                return None

            # Calculate statistics
            summary = {
                "username": username,
                "total_owned_games": len(df),
                "avg_complexity": (
                    df.select(pl.col("complexity").mean()).item()
                    if "complexity" in df.columns
                    else None
                ),
                "avg_bgg_rating": (
                    df.select(pl.col("geek_rating").mean()).item()
                    if "geek_rating" in df.columns
                    else None
                ),
                "avg_year_published": (
                    df.select(pl.col("year_published").mean()).item()
                    if "year_published" in df.columns
                    else None
                ),
                "avg_playing_time": (
                    df.select(pl.col("playing_time").mean()).item()
                    if "playing_time" in df.columns
                    else None
                ),
                "avg_min_players": (
                    df.select(pl.col("min_players").mean()).item()
                    if "min_players" in df.columns
                    else None
                ),
                "avg_max_players": (
                    df.select(pl.col("max_players").mean()).item()
                    if "max_players" in df.columns
                    else None
                ),
            }

            # Get most common categories/mechanics if available
            if "categories" in df.columns:
                # This would require unnesting - simplified for now
                logger.info("Category/mechanic analysis not yet implemented")

            logger.info(f"Generated summary statistics for '{username}'")
            return summary

        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")
            import traceback

            traceback.print_exc()
            return None
