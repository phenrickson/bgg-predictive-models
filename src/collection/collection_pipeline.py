"""End-to-end pipeline for user collection modeling."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import polars as pl

from src.collection.collection_loader import BGGCollectionLoader
from src.collection.collection_integration import CollectionIntegration
from src.collection.collection_artifact_storage import (
    CollectionArtifactStorage,
    ArtifactStorageConfig,
)
from src.collection.collection_split import CollectionSplit, SplitConfig
from src.collection.collection_model import CollectionModel, ModelConfig
from src.collection.collection_analyzer import CollectionAnalyzer, AnalyzerConfig
from src.data.loader import BGGDataLoader
from src.utils.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full collection pipeline."""

    storage_config: ArtifactStorageConfig = field(default_factory=ArtifactStorageConfig)
    split_config: SplitConfig = field(default_factory=SplitConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    analyzer_config: AnalyzerConfig = field(default_factory=AnalyzerConfig)

    train_end_year: Optional[int] = None
    """Year to end training data for time-based splits."""

    min_ratings_for_universe: int = 25
    """Minimum ratings for games in the universe."""


class CollectionPipeline:
    """End-to-end pipeline for user collection modeling.

    Orchestrates the full workflow:
    1. Fetch/load user collection
    2. Join with game features
    3. Create train/val/test splits
    4. Train ownership model
    5. Generate predictions for all games
    6. Create analysis artifacts
    7. Save all artifacts to GCS

    Example usage:
        >>> pipeline = CollectionPipeline("phenrickson")
        >>> results = pipeline.run_full_pipeline()
    """

    def __init__(
        self,
        username: str,
        config: Optional[PipelineConfig] = None,
    ):
        """Initialize pipeline for a specific user.

        Args:
            username: BGG username
            config: Optional pipeline configuration
        """
        self.username = username
        self.config = config or PipelineConfig()

        # Initialize components
        self.storage = CollectionArtifactStorage(username, self.config.storage_config)

        # Load project config
        project_config = load_config()
        self.bq_config = project_config.get_bigquery_config()

        logger.info(f"Initialized pipeline for user '{username}'")

    def run_full_pipeline(
        self,
        refresh_collection: bool = True,
    ) -> Dict[str, Any]:
        """Run the complete pipeline.

        Args:
            refresh_collection: Whether to fetch fresh collection from BGG API

        Returns:
            Dictionary with pipeline results and artifact locations
        """
        logger.info(f"Starting full pipeline for user '{self.username}'")
        start_time = datetime.now()

        results = {
            "username": self.username,
            "started_at": start_time.isoformat(),
            "steps": {},
            "artifacts": {},
        }

        try:
            # Step 1: Get collection with features
            logger.info("Step 1: Loading collection with features")
            collection_df = self._get_collection_with_features(refresh_collection)
            results["steps"]["collection"] = {
                "status": "success",
                "total_items": len(collection_df),
                "owned_games": collection_df.filter(pl.col("owned") == True).height,
            }

            # Save collection
            collection_path = self.storage.save_collection(collection_df)
            results["artifacts"]["collection"] = collection_path

            # Step 2: Load game universe
            logger.info("Step 2: Loading game universe")
            game_universe_df = self._load_game_universe()
            results["steps"]["universe"] = {
                "status": "success",
                "total_games": len(game_universe_df),
            }

            # Step 3: Create splits
            logger.info("Step 3: Creating train/val/test splits")
            splitter = CollectionSplit(
                collection_df=collection_df,
                game_universe_df=game_universe_df,
                config=self.config.split_config,
            )

            train_df, val_df, test_df = splitter.create_ownership_splits(
                train_end_year=self.config.train_end_year
            )

            results["steps"]["splits"] = {
                "status": "success",
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "train_positive_rate": train_df["target"].mean(),
            }

            # Save splits
            split_paths = self.storage.save_splits(train_df, val_df, test_df)
            results["artifacts"]["splits"] = split_paths

            # Step 4: Train model
            logger.info("Step 4: Training ownership model")
            model = CollectionModel(self.username, self.config.model_config)
            pipeline, best_params = model.train(train_df, val_df)

            # Find optimal threshold
            threshold = model.find_optimal_threshold(pipeline, val_df)

            # Evaluate on test set
            metrics = model.evaluate(pipeline, test_df, threshold)

            results["steps"]["training"] = {
                "status": "success",
                "model_type": self.config.model_config.model_type,
                "best_params": best_params,
                "threshold": threshold,
                "test_metrics": metrics,
            }

            # Save model
            model_metadata = model.get_model_metadata(
                pipeline, best_params, metrics, threshold
            )
            model_path = self.storage.save_model(
                pipeline, model_metadata, threshold
            )
            results["artifacts"]["model"] = model_path

            # Step 5: Generate predictions
            logger.info("Step 5: Generating predictions for all games")
            predictions_df = model.predict(pipeline, game_universe_df, threshold)

            # Enrich predictions with game info
            predictions_df = predictions_df.join(
                game_universe_df.select([
                    "game_id", "name", "year_published", "geek_rating",
                    "complexity", "categories", "mechanics"
                ]),
                on="game_id",
                how="left",
            )

            # Mark games in collection
            owned_ids = set(collection_df.filter(pl.col("owned") == True)["game_id"].to_list())
            wishlist_ids = set(
                collection_df.filter(
                    (pl.col("wishlist") == True) if "wishlist" in collection_df.columns
                    else pl.lit(False)
                )["game_id"].to_list()
            )

            predictions_df = predictions_df.with_columns([
                pl.col("game_id").is_in(list(owned_ids)).alias("in_collection"),
                pl.when(pl.col("game_id").is_in(list(owned_ids)))
                .then(pl.lit("owned"))
                .when(pl.col("game_id").is_in(list(wishlist_ids)))
                .then(pl.lit("wishlist"))
                .otherwise(pl.lit("none"))
                .alias("collection_status"),
            ])

            results["steps"]["predictions"] = {
                "status": "success",
                "total_games_scored": len(predictions_df),
            }

            # Step 6: Generate analysis artifacts
            logger.info("Step 6: Generating analysis artifacts")
            analyzer = CollectionAnalyzer(
                username=self.username,
                collection_df=collection_df,
                predictions_df=predictions_df,
                game_universe_df=game_universe_df,
                config=self.config.analyzer_config,
            )
            analyzer.set_metrics(metrics)

            summary_stats = analyzer.generate_summary_stats()
            category_affinity = analyzer.generate_category_affinity()
            top_recommendations = analyzer.generate_top_recommendations()
            feature_importance = model.get_feature_importance(pipeline)

            results["steps"]["analysis"] = {
                "status": "success",
                "top_recommendations_count": len(top_recommendations),
            }

            # Save predictions
            pred_paths = self.storage.save_predictions(predictions_df, top_recommendations)
            results["artifacts"]["predictions"] = pred_paths

            # Save analysis artifacts
            if feature_importance is not None:
                analysis_paths = self.storage.save_analysis_artifacts(
                    summary_stats, feature_importance, category_affinity
                )
                results["artifacts"]["analysis"] = analysis_paths

            # Save user metadata
            metadata = {
                "last_pipeline_run": datetime.now().isoformat(),
                "collection_size": len(collection_df),
                "owned_games": collection_df.filter(pl.col("owned") == True).height,
                "model_version": results["artifacts"].get("model"),
            }
            self.storage.save_user_metadata(metadata)

            # Success
            end_time = datetime.now()
            results["completed_at"] = end_time.isoformat()
            results["duration_seconds"] = (end_time - start_time).total_seconds()
            results["status"] = "success"

            logger.info(
                f"Pipeline completed successfully in {results['duration_seconds']:.1f}s"
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            raise

        return results

    def _get_collection_with_features(
        self, refresh: bool = True
    ) -> pl.DataFrame:
        """Get user's collection joined with game features.

        Args:
            refresh: Whether to fetch fresh from BGG API

        Returns:
            Collection DataFrame with features
        """
        integration = CollectionIntegration(
            config=self.bq_config,
            environment=self.config.storage_config.environment or "dev",
        )

        if refresh:
            # Fetch from BGG API and save to BigQuery
            loader = BGGCollectionLoader(self.username)
            collection_df = loader.get_collection()

            if collection_df is None:
                raise ValueError(f"Could not fetch collection for user '{self.username}'")

            # Save to BigQuery storage
            from src.collection.collection_storage import CollectionStorage
            bq_storage = CollectionStorage(
                environment=self.config.storage_config.environment or "dev"
            )
            bq_storage.save_collection(self.username, collection_df)

        # Get collection with features
        return integration.get_collection_with_features(
            self.username, owned_only=False, games_only=True
        )

    def _load_game_universe(self) -> pl.DataFrame:
        """Load the full game universe for predictions.

        Returns:
            DataFrame with all games and features
        """
        loader = BGGDataLoader(self.bq_config)
        return loader.load_training_data(
            min_ratings=self.config.min_ratings_for_universe
        )

    def refresh_predictions_only(
        self,
        model_version: Optional[int] = None,
    ) -> Dict[str, str]:
        """Re-run predictions with existing model.

        Useful when game universe is updated but model hasn't changed.

        Args:
            model_version: Model version to use (latest if not specified)

        Returns:
            Dictionary mapping artifact names to GCS paths
        """
        logger.info(f"Refreshing predictions for user '{self.username}'")

        # Load model
        pipeline, metadata, threshold = self.storage.load_model(model_version)
        logger.info(f"Loaded model v{metadata.get('version', '?')}")

        # Load game universe
        game_universe_df = self._load_game_universe()

        # Load collection
        collection_df = self.storage.load_collection()
        if collection_df is None:
            raise ValueError("No collection found in storage. Run full pipeline first.")

        # Generate predictions
        model = CollectionModel(self.username, self.config.model_config)
        predictions_df = model.predict(pipeline, game_universe_df, threshold)

        # Enrich and save
        predictions_df = predictions_df.join(
            game_universe_df.select([
                "game_id", "name", "year_published", "geek_rating",
                "complexity", "categories", "mechanics"
            ]),
            on="game_id",
            how="left",
        )

        # Mark collection status
        owned_ids = set(collection_df.filter(pl.col("owned") == True)["game_id"].to_list())
        predictions_df = predictions_df.with_columns([
            pl.col("game_id").is_in(list(owned_ids)).alias("in_collection"),
        ])

        # Generate recommendations
        analyzer = CollectionAnalyzer(
            username=self.username,
            collection_df=collection_df,
            predictions_df=predictions_df,
            game_universe_df=game_universe_df,
            config=self.config.analyzer_config,
        )
        top_recommendations = analyzer.generate_top_recommendations()

        # Save
        paths = self.storage.save_predictions(predictions_df, top_recommendations)

        logger.info("Predictions refreshed successfully")
        return paths

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of pipeline artifacts.

        Returns:
            Dictionary with artifact existence and metadata
        """
        return self.storage.get_artifact_status()
