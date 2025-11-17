"""Process and analyze BGG collection data."""

import polars as pl
from typing import Dict, List, Optional


class CollectionProcessor:
    """Process and analyze BGG collection data."""

    def __init__(self, collection_df: pl.DataFrame):
        """Initialize collection processor.

        Args:
            collection_df: DataFrame containing BGG collection data
        """
        self.df = collection_df

    def get_games(self, owned_only: bool = False) -> pl.DataFrame:
        """Get boardgames from collection.

        Args:
            owned_only: Only include owned games (default: False)

        Returns:
            DataFrame containing only boardgames
        """
        query = pl.col("subtype") == "boardgame"
        if owned_only:
            query = query & pl.col("owned")
        return self.df.filter(query)

    def get_expansions(self, owned_only: bool = False) -> pl.DataFrame:
        """Get expansions from collection.

        Args:
            owned_only: Only include owned expansions (default: False)

        Returns:
            DataFrame containing only expansions
        """
        query = pl.col("subtype") == "boardgameexpansion"
        if owned_only:
            query = query & pl.col("owned")
        return self.df.filter(query)

    def get_summary(self) -> Dict:
        """Get summary statistics about the collection.

        Returns:
            Dictionary with collection summary
        """
        summary = {
            "total_items": len(self.df),
            "owned_games": (
                len(self.df.filter(pl.col("owned") == True))
                if "owned" in self.df.columns
                else 0
            ),
            "unique_games": (
                len(self.df.filter(pl.col("subtype") == "boardgame"))
                if "subtype" in self.df.columns
                else 0
            ),
            "expansions": (
                len(self.df.filter(pl.col("subtype") == "boardgameexpansion"))
                if "subtype" in self.df.columns
                else 0
            ),
            "avg_user_rating": (
                self.df.select(pl.col("user_rating").mean()).item()
                if "user_rating" in self.df.columns
                else None
            ),
            "avg_bgg_rating": (
                self.df.select(pl.col("average_rating").mean()).item()
                if "average_rating" in self.df.columns
                else None
            ),
            "top_rated_game": None,
        }

        # Get top rated game by user
        if "user_rating" in self.df.columns and "game_name" in self.df.columns:
            top_rated = (
                self.df.filter(pl.col("user_rating").is_not_null())
                .sort("user_rating", descending=True)
                .head(1)
            )
            if len(top_rated) > 0:
                summary["top_rated_game"] = {
                    "name": top_rated.select("game_name").item(),
                    "rating": top_rated.select("user_rating").item(),
                }

        return summary

    def get_ratings_distribution(self) -> pl.DataFrame:
        """Get distribution of user ratings.

        Returns:
            DataFrame with rating counts
        """
        return (
            self.df.filter(pl.col("user_rating").is_not_null())
            .groupby("user_rating")
            .agg(pl.count().alias("count"))
            .sort("user_rating")
        )

    def get_top_rated(self, n: int = 10, min_rating: float = None) -> pl.DataFrame:
        """Get top rated games.

        Args:
            n: Number of games to return (default: 10)
            min_rating: Minimum rating to include (default: None)

        Returns:
            DataFrame with top rated games
        """
        query = pl.col("user_rating").is_not_null()
        if min_rating is not None:
            query = query & (pl.col("user_rating") >= min_rating)

        return (
            self.df.filter(query)
            .sort("user_rating", descending=True)
            .head(n)
            .select(["game_name", "user_rating", "average_rating", "bayes_average"])
        )
