"""End-to-end pipeline for user collection modeling.

Loops over outcomes declared in config.yaml (`collections.outcomes`) and trains
one model per (user, outcome). Per-outcome artifacts (model, splits, predictions,
analysis) are versioned independently under GCS.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import polars as pl

from src.collection.collection_loader import BGGCollectionLoader
from src.collection.collection_processor import CollectionProcessor
from src.collection.collection_artifact_storage import (
    CollectionArtifactStorage,
    ArtifactStorageConfig,
)
from src.collection.collection_split import (
    CollectionSplitter,
    ClassificationSplitConfig,
    RegressionSplitConfig,
)
from src.collection.collection_model import (
    CollectionModel,
    ClassificationModelConfig,
    RegressionModelConfig,
)
from src.collection.collection_analyzer import CollectionAnalyzer, AnalyzerConfig
from src.collection.collection_storage import CollectionStorage
from src.collection.outcomes import (
    OutcomeDefinition,
    apply_outcome,
    load_outcomes,
)
from src.data.loader import BGGDataLoader
from src.utils.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the collection pipeline.

    The same split/model configs are applied to every outcome. Per-outcome
    overrides are not supported yet; add them here when a real need appears.
    """

    storage_config: ArtifactStorageConfig = field(default_factory=ArtifactStorageConfig)
    classification_split_config: ClassificationSplitConfig = field(
        default_factory=ClassificationSplitConfig
    )
    regression_split_config: RegressionSplitConfig = field(
        default_factory=RegressionSplitConfig
    )
    classification_model_config: ClassificationModelConfig = field(
        default_factory=ClassificationModelConfig
    )
    regression_model_config: RegressionModelConfig = field(
        default_factory=RegressionModelConfig
    )
    analyzer_config: AnalyzerConfig = field(default_factory=AnalyzerConfig)

    min_ratings_for_universe: int = 25
    """Minimum ratings for games in the universe."""


class CollectionPipeline:
    """End-to-end pipeline for user collection modeling.

    Workflow:
    1. Fetch raw collection from BGG (optional) and persist to BQ
    2. Process once: canonicalize schema + join with game universe
    3. Load outcomes from config
    4. For each outcome: apply_outcome -> split -> train -> evaluate -> save
    """

    def __init__(
        self,
        username: str,
        config: Optional[PipelineConfig] = None,
    ):
        self.username = username
        self.config = config or PipelineConfig()

        self.storage = CollectionArtifactStorage(username, self.config.storage_config)

        self._project_config = load_config()
        self.bq_config = self._project_config.get_bigquery_config()
        self._environment = self.config.storage_config.environment or "dev"

        logger.info(f"Initialized pipeline for user '{username}'")

    def run_full_pipeline(
        self,
        refresh_collection: bool = True,
        outcome_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train models for all outcomes (or a filtered subset) for this user.

        Args:
            refresh_collection: Fetch fresh collection from BGG API before training.
            outcome_filter: Restrict to these outcome names. If None, trains all.

        Returns:
            {
                "username": str,
                "started_at": iso,
                "finished_at": iso,
                "collection_rows": int,
                "outcomes": {
                    "own": {"version": int, "metrics": {...}, "best_params": {...}},
                    ...
                },
            }
        """
        logger.info(f"Starting full pipeline for user '{self.username}'")
        start = datetime.now()

        # Step 1: fetch + persist raw collection (optional) + process
        if refresh_collection:
            self._fetch_and_persist_collection()

        processed = self._process_collection()
        logger.info(f"Processed collection: {processed.height} rows")

        # Step 2: save the processed snapshot (outcome-agnostic)
        self.storage.save_collection(processed)

        # Step 3: load outcomes from config
        outcomes = load_outcomes(self._project_config.raw_config)
        if outcome_filter:
            missing = set(outcome_filter) - set(outcomes)
            if missing:
                raise ValueError(f"Unknown outcomes requested: {sorted(missing)}")
            outcomes = {k: v for k, v in outcomes.items() if k in outcome_filter}
        if not outcomes:
            raise ValueError("No outcomes to train")

        # Step 4: shared splitter (universe = processed dataframe)
        splitter = CollectionSplitter(
            universe_df=processed,
            classification_config=self.config.classification_split_config,
            regression_config=self.config.regression_split_config,
        )

        results: Dict[str, Any] = {
            "username": self.username,
            "started_at": start.isoformat(),
            "collection_rows": processed.height,
            "outcomes": {},
        }

        for name, outcome in outcomes.items():
            logger.info(f"--- outcome: {name} ({outcome.task}) ---")
            try:
                results["outcomes"][name] = self._train_one_outcome(
                    processed, outcome, splitter
                )
            except Exception as exc:
                logger.exception(f"Outcome {name!r} failed: {exc}")
                results["outcomes"][name] = {"error": str(exc)}

        finished = datetime.now()
        results["finished_at"] = finished.isoformat()
        results["duration_seconds"] = (finished - start).total_seconds()
        logger.info(
            f"Pipeline finished for user '{self.username}' "
            f"in {results['duration_seconds']:.1f}s"
        )
        return results

    def refresh_predictions_only(
        self,
        outcome_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Regenerate predictions using the latest registered model per outcome.

        Does not retrain. Uses the processed collection snapshot from storage
        as the prediction universe.
        """
        logger.info(f"Refreshing predictions for user '{self.username}'")

        processed = self._process_collection()
        if processed is None:
            raise ValueError(
                f"No stored collection for user '{self.username}'. "
                "Run the full pipeline first."
            )

        outcomes = load_outcomes(self._project_config.raw_config)
        if outcome_filter:
            outcomes = {k: v for k, v in outcomes.items() if k in outcome_filter}

        results: Dict[str, Any] = {}
        for name, outcome in outcomes.items():
            version = self.storage.latest_version(name)
            if version is None:
                logger.warning(
                    f"No trained model for outcome {name!r}; skipping"
                )
                continue
            results[name] = self._predict_one_outcome(processed, outcome, version)

        return results

    def get_pipeline_status(self) -> Dict[str, Any]:
        return self.storage.get_artifact_status()

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _fetch_and_persist_collection(self) -> None:
        """Fetch raw collection from BGG and persist to BQ."""
        loader = BGGCollectionLoader(self.username)
        raw = loader.get_collection()
        if raw is None:
            raise ValueError(
                f"Could not fetch collection for user '{self.username}'"
            )
        bq_storage = CollectionStorage(environment=self._environment)
        bq_storage.save_collection(self.username, raw)
        logger.info(f"Persisted {raw.height} raw collection rows for '{self.username}'")

    def _process_collection(self) -> pl.DataFrame:
        """Load + canonicalize + join collection with game universe."""
        processor = CollectionProcessor(
            config=self.bq_config, environment=self._environment
        )
        result = processor.process(self.username)
        if result is None:
            raise ValueError(
                f"No stored collection for user '{self.username}'. "
                "Run with refresh_collection=True."
            )
        return result

    def _train_one_outcome(
        self,
        processed: pl.DataFrame,
        outcome: OutcomeDefinition,
        splitter: CollectionSplitter,
    ) -> Dict[str, Any]:
        labeled = apply_outcome(processed, outcome)
        train_df, val_df, test_df = splitter.split(labeled, outcome)

        if train_df.height == 0:
            raise ValueError(f"Train split is empty for outcome {outcome.name!r}")

        model = CollectionModel(
            username=self.username,
            outcome=outcome,
            classification_config=self.config.classification_model_config,
            regression_config=self.config.regression_model_config,
        )
        pipeline_obj, best_params = model.train(train_df, val_df)
        metrics = model.evaluate(pipeline_obj, test_df)

        threshold: Optional[float] = None
        if outcome.task == "classification":
            threshold = model.find_threshold(pipeline_obj, val_df)

        version = self.storage._next_version(outcome.name)

        self.storage.save_splits(outcome.name, train_df, val_df, test_df, version=version)
        metadata = {
            "task": outcome.task,
            "best_params": best_params,
            "metrics": metrics,
        }
        self.storage.save_model(
            outcome.name, pipeline_obj, metadata, threshold=threshold, version=version
        )

        logger.info(
            f"Trained {outcome.name} v{version}: "
            f"train={train_df.height} val={val_df.height} test={test_df.height}; "
            f"metrics={metrics}"
        )

        return {
            "version": version,
            "task": outcome.task,
            "threshold": threshold,
            "best_params": best_params,
            "metrics": metrics,
            "split_sizes": {
                "train": train_df.height,
                "val": val_df.height,
                "test": test_df.height,
            },
        }

    def _predict_one_outcome(
        self,
        processed: pl.DataFrame,
        outcome: OutcomeDefinition,
        version: int,
    ) -> Dict[str, Any]:
        pipeline_obj, metadata, threshold = self.storage.load_model(
            outcome.name, version=version
        )
        X = processed.to_pandas()
        if outcome.task == "classification":
            proba = pipeline_obj.predict_proba(X)[:, 1]
            predictions_df = pl.DataFrame(
                {
                    "game_id": processed["game_id"].to_list(),
                    "score": proba,
                }
            )
        else:
            preds = pipeline_obj.predict(X)
            predictions_df = pl.DataFrame(
                {
                    "game_id": processed["game_id"].to_list(),
                    "score": preds,
                }
            )

        self.storage.save_predictions(
            outcome.name, version, predictions_df, top_recommendations=[]
        )
        logger.info(
            f"Refreshed predictions for {outcome.name} v{version}: "
            f"{predictions_df.height} rows"
        )
        return {"version": version, "rows": predictions_df.height}
