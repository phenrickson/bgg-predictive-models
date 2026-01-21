"""Generate analysis artifacts for dash viewer consumption."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import polars as pl
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """Configuration for collection analysis."""

    top_n_recommendations: int = 100
    """Number of top recommendations to generate."""

    top_n_categories: int = 10
    """Number of top categories to include in affinity analysis."""

    top_n_mechanics: int = 10
    """Number of top mechanics to include in affinity analysis."""

    min_games_for_affinity: int = 3
    """Minimum games in a category/mechanic to calculate affinity."""


class CollectionAnalyzer:
    """Generate analysis artifacts for dash viewer consumption.

    Creates JSON and parquet artifacts that can be loaded into a Dash
    application for visualizing user collection analysis and recommendations.

    Example usage:
        >>> analyzer = CollectionAnalyzer("phenrickson", collection_df, predictions_df)
        >>> summary = analyzer.generate_summary_stats()
        >>> recommendations = analyzer.generate_top_recommendations()
    """

    def __init__(
        self,
        username: str,
        collection_df: pl.DataFrame,
        predictions_df: Optional[pl.DataFrame] = None,
        game_universe_df: Optional[pl.DataFrame] = None,
        config: Optional[AnalyzerConfig] = None,
    ):
        """Initialize analyzer with collection and predictions.

        Args:
            username: BGG username
            collection_df: User's collection with features
            predictions_df: Optional DataFrame with ownership predictions
            game_universe_df: Optional full game universe for enriching recommendations
            config: Analyzer configuration
        """
        self.username = username
        self.collection_df = collection_df
        self.predictions_df = predictions_df
        self.game_universe_df = game_universe_df
        self.config = config or AnalyzerConfig()

        # Filter to owned games
        self.owned_df = collection_df.filter(pl.col("owned") == True)

        logger.info(f"Initialized analyzer for user '{username}'")
        logger.info(f"Collection: {len(collection_df)} items, {len(self.owned_df)} owned")

    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate collection summary statistics.

        Returns:
            Dictionary with collection stats, preference profile, etc.
        """
        owned = self.owned_df

        # Basic collection stats
        collection_stats = {
            "total_owned": len(owned),
            "total_in_collection": len(self.collection_df),
        }

        # Count by status
        for col in ["wishlist", "previously_owned", "want_to_play"]:
            if col in self.collection_df.columns:
                count = self.collection_df.filter(pl.col(col) == True).height
                collection_stats[col] = count

        # Rating stats
        if "user_rating" in owned.columns:
            rated = owned.filter(pl.col("user_rating").is_not_null())
            collection_stats["total_rated"] = len(rated)
            if len(rated) > 0:
                collection_stats["avg_user_rating"] = round(
                    rated["user_rating"].mean(), 2
                )
                collection_stats["min_user_rating"] = rated["user_rating"].min()
                collection_stats["max_user_rating"] = rated["user_rating"].max()

        # Preference profile from owned games
        preference_profile = {}

        numeric_cols = [
            ("complexity", "avg_complexity"),
            ("geek_rating", "avg_geek_rating"),
            ("year_published", "avg_year_published"),
            ("min_playtime", "avg_min_playtime"),
            ("max_playtime", "avg_max_playtime"),
            ("min_players", "avg_min_players"),
            ("max_players", "avg_max_players"),
        ]

        for col, key in numeric_cols:
            if col in owned.columns:
                val = owned[col].mean()
                if val is not None:
                    preference_profile[key] = round(val, 2)

        # Year range
        if "year_published" in owned.columns:
            preference_profile["year_range"] = {
                "min": owned["year_published"].min(),
                "max": owned["year_published"].max(),
            }

        # Model performance (if predictions available)
        model_performance = None
        if self.predictions_df is not None and "metrics" in dir(self):
            model_performance = getattr(self, "metrics", None)

        summary = {
            "username": self.username,
            "generated_at": datetime.now().isoformat(),
            "collection_stats": collection_stats,
            "preference_profile": preference_profile,
            "model_performance": model_performance,
        }

        logger.info(f"Generated summary stats for '{self.username}'")
        return summary

    def generate_category_affinity(self) -> Dict[str, Any]:
        """Analyze user's affinity for categories and mechanics.

        Returns:
            Dictionary with category/mechanic preferences and ownership rates.
        """
        owned = self.owned_df

        affinity = {
            "username": self.username,
            "generated_at": datetime.now().isoformat(),
            "categories": [],
            "mechanics": [],
        }

        # Analyze categories
        if "categories" in owned.columns:
            affinity["categories"] = self._analyze_list_column(
                owned, "categories", self.config.top_n_categories
            )

        # Analyze mechanics
        if "mechanics" in owned.columns:
            affinity["mechanics"] = self._analyze_list_column(
                owned, "mechanics", self.config.top_n_mechanics
            )

        logger.info(
            f"Generated affinity analysis: "
            f"{len(affinity['categories'])} categories, "
            f"{len(affinity['mechanics'])} mechanics"
        )

        return affinity

    def _analyze_list_column(
        self, df: pl.DataFrame, column: str, top_n: int
    ) -> List[Dict[str, Any]]:
        """Analyze a list column (categories, mechanics) for affinity.

        Args:
            df: DataFrame with owned games
            column: Column name containing lists
            top_n: Number of top items to return

        Returns:
            List of dictionaries with item name and counts
        """
        # Explode the list column and count
        try:
            exploded = df.select(["game_id", column]).explode(column)
            counts = (
                exploded.group_by(column)
                .agg(pl.count().alias("count"))
                .sort("count", descending=True)
            )

            # Filter by minimum games
            counts = counts.filter(pl.col("count") >= self.config.min_games_for_affinity)

            # Take top N
            top_items = counts.head(top_n)

            return [
                {"name": row[column], "count": row["count"]}
                for row in top_items.iter_rows(named=True)
            ]
        except Exception as e:
            logger.warning(f"Error analyzing {column}: {e}")
            return []

    def generate_top_recommendations(
        self,
        n: Optional[int] = None,
        exclude_owned: bool = True,
        min_year: Optional[int] = None,
        min_geek_rating: Optional[float] = None,
        max_complexity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Generate top N game recommendations.

        Args:
            n: Number of recommendations (default from config)
            exclude_owned: Exclude games already owned
            min_year: Minimum year published
            min_geek_rating: Minimum geek rating
            max_complexity: Maximum complexity

        Returns:
            List of recommended games with metadata
        """
        if self.predictions_df is None:
            logger.warning("No predictions available for recommendations")
            return []

        n = n or self.config.top_n_recommendations

        # Start with predictions
        recs = self.predictions_df.clone()

        # Exclude owned games
        if exclude_owned:
            owned_ids = set(self.owned_df["game_id"].to_list())
            recs = recs.filter(~pl.col("game_id").is_in(list(owned_ids)))

        # Join with game universe for full metadata
        if self.game_universe_df is not None:
            recs = recs.join(
                self.game_universe_df,
                on="game_id",
                how="left",
            )

        # Apply filters
        if min_year is not None and "year_published" in recs.columns:
            recs = recs.filter(pl.col("year_published") >= min_year)

        if min_geek_rating is not None and "geek_rating" in recs.columns:
            recs = recs.filter(pl.col("geek_rating") >= min_geek_rating)

        if max_complexity is not None and "complexity" in recs.columns:
            recs = recs.filter(pl.col("complexity") <= max_complexity)

        # Sort by ownership probability and take top N
        recs = recs.sort("ownership_probability", descending=True).head(n)

        # Build recommendation list
        recommendations = []
        for i, row in enumerate(recs.iter_rows(named=True)):
            rec = {
                "rank": i + 1,
                "game_id": row["game_id"],
                "ownership_probability": round(row["ownership_probability"], 4),
            }

            # Add available metadata
            for col in ["name", "game_name", "year_published", "geek_rating",
                       "complexity", "categories", "mechanics", "thumbnail"]:
                if col in row and row[col] is not None:
                    rec[col] = row[col]

            recommendations.append(rec)

        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations

    def generate_feature_importance(
        self, pipeline: Pipeline
    ) -> pl.DataFrame:
        """Extract feature importance from trained model.

        Args:
            pipeline: Trained sklearn pipeline

        Returns:
            DataFrame with feature names and importance values
        """
        try:
            model = pipeline.named_steps["model"]
            preprocessor = pipeline.named_steps["preprocessor"]

            # Get feature names
            feature_names = preprocessor.named_steps[
                "bgg_preprocessor"
            ].get_feature_names_out()

            # Get importance
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                import numpy as np
                importance = np.abs(model.coef_[0])
            else:
                logger.warning("Model does not support feature importance")
                return pl.DataFrame({"feature": [], "importance": []})

            # Build DataFrame
            fi_df = pl.DataFrame({
                "feature": feature_names,
                "importance": importance,
            }).sort("importance", descending=True)

            # Normalize importance
            total = fi_df["importance"].sum()
            if total > 0:
                fi_df = fi_df.with_columns(
                    (pl.col("importance") / total).alias("importance_normalized")
                )

            logger.info(f"Generated feature importance for {len(fi_df)} features")
            return fi_df

        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
            return pl.DataFrame({"feature": [], "importance": []})

    def generate_rating_analysis(self) -> Dict[str, Any]:
        """Analyze user rating patterns.

        Returns:
            Dictionary with rating distribution and patterns
        """
        owned = self.owned_df

        if "user_rating" not in owned.columns:
            return {"error": "No user ratings available"}

        rated = owned.filter(pl.col("user_rating").is_not_null())

        if len(rated) == 0:
            return {"error": "No rated games"}

        # Rating distribution
        distribution = (
            rated.group_by("user_rating")
            .agg(pl.count().alias("count"))
            .sort("user_rating")
        )

        # Correlation with BGG ratings
        correlation = None
        if "geek_rating" in rated.columns:
            user_ratings = rated["user_rating"].to_numpy()
            geek_ratings = rated["geek_rating"].to_numpy()
            import numpy as np
            correlation = round(np.corrcoef(user_ratings, geek_ratings)[0, 1], 3)

        # Top rated games
        top_rated = (
            rated.sort("user_rating", descending=True)
            .head(10)
            .select(["game_id", "game_name", "user_rating", "geek_rating"])
        )

        return {
            "total_rated": len(rated),
            "distribution": distribution.to_dicts(),
            "correlation_with_bgg": correlation,
            "top_rated": top_rated.to_dicts(),
        }

    def set_metrics(self, metrics: Dict[str, float]) -> None:
        """Store model metrics for inclusion in summary.

        Args:
            metrics: Dictionary of model evaluation metrics
        """
        self.metrics = metrics
