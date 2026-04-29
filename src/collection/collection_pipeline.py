"""End-to-end pipeline for user collection modeling.

Loops over outcomes declared in config.yaml (`collections.outcomes`) and trains
one model per (user, outcome). Per-outcome artifacts (model, splits, predictions,
analysis) are versioned independently on the local filesystem. GCS round-trips
are handled separately by ``sync_collections``.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

from src.collection.collection_loader import BGGCollectionLoader
from src.collection.collection_processor import CollectionProcessor, ProcessorConfig
from src.data.loader import BGGDataLoader
from src.collection.collection_artifact_storage import CollectionArtifactStorage
from src.collection.collection_split import (
    ClassificationSplitConfig,
    CollectionSplitter,
    RegressionSplitConfig,
    downsample_negatives,
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
from src.utils.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the collection pipeline.

    The same split/model configs are applied to every outcome. Per-outcome
    overrides are not supported yet; add them here when a real need appears.
    """

    local_root: Union[str, Path] = "models/collections"
    """Root directory for local artifact storage."""

    environment: Optional[str] = None
    """Environment name (e.g. ``"dev"``, ``"prod"``). Defaults to the
    environment reported by :func:`src.utils.config.load_config`."""

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

    processor_config: Optional[ProcessorConfig] = None
    """Optional ProcessorConfig passed to :class:`CollectionProcessor`."""

    use_predicted_complexity: bool = False
    """If True, the loaded universe carries ``predicted_complexity``."""

    use_embeddings: bool = False
    """If True, the loaded universe carries description-embedding columns
    (``emb_0..emb_{N-1}``)."""

    downsample_negatives_ratio: Optional[float] = None
    """If set, apply :func:`downsample_negatives` to the training split of
    classification outcomes using this negatives-per-positive ratio. Val and
    test splits are never downsampled. ``None`` disables downsampling."""

    downsample_protect_min_ratings: int = 25
    """Floor for protected (always-kept) negatives during downsampling.
    Negatives with ``users_rated >= downsample_protect_min_ratings`` are
    never sampled out — only the low-rating tail gets thinned. Set to 0
    for uniform sampling over all negatives."""


def fetch_and_persist(username: str, environment: str) -> int:
    """Fetch a user's collection from the BGG API and upsert into BigQuery.

    Returns the number of rows persisted.
    """
    loader = BGGCollectionLoader(username)
    raw = loader.get_collection()
    if raw is None:
        raise ValueError(f"Could not fetch collection for user '{username}'")
    bq_storage = CollectionStorage(environment=environment)
    bq_storage.save_collection(username, raw)
    logger.info(f"Persisted {raw.height} raw collection rows for '{username}'")
    return raw.height


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

        self.storage = CollectionArtifactStorage(
            username,
            local_root=self.config.local_root,
            environment=self.config.environment,
        )

        self._project_config = load_config()
        self.bq_config = self._project_config.get_bigquery_config()
        self._environment = (
            self.config.environment or self._project_config.get_environment_prefix()
        )

        # Lazily constructed so collection processing and universe loading
        # share one processor (and therefore one ProcessorConfig).
        self._processor: Optional[CollectionProcessor] = None

        logger.info(f"Initialized pipeline for user '{username}'")

    def _get_processor(self) -> CollectionProcessor:
        """Return (constructing once) the shared CollectionProcessor.

        Using a single instance guarantees that the collection processing
        path and the universe-feature-loading path see the same
        ``ProcessorConfig`` — so features are identical on both sides.
        """
        if self._processor is None:
            self._processor = CollectionProcessor(
                config=self.bq_config,
                environment=self._environment,
                processor_config=self.config.processor_config,
            )
        return self._processor

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

        # Step 3: load the game universe (features for every game).
        universe_df = self._load_game_universe()
        logger.info(f"Loaded game universe: {universe_df.height} games")

        # Step 4: join universe (features) with the user's collection
        # (status). Every universe row gets the user's status if any.
        # This single frame is what the splitter operates on.
        joined = universe_df.join(processed, on="game_id", how="left")
        logger.info(f"Joined frame: {joined.height} rows × {len(joined.columns)} columns")

        # Step 5: load outcomes from config
        outcomes = load_outcomes(self._project_config.raw_config)
        if outcome_filter:
            missing = set(outcome_filter) - set(outcomes)
            if missing:
                raise ValueError(f"Unknown outcomes requested: {sorted(missing)}")
            outcomes = {k: v for k, v in outcomes.items() if k in outcome_filter}
        if not outcomes:
            raise ValueError("No outcomes to train")

        # Step 6: shared splitter (no universe arg — just splits the
        # joined frame).
        splitter = CollectionSplitter(
            classification_config=self.config.classification_split_config,
            regression_config=self.config.regression_split_config,
        )

        results: Dict[str, Any] = {
            "username": self.username,
            "started_at": start.isoformat(),
            "collection_rows": processed.height,
            "universe_rows": universe_df.height,
            "outcomes": {},
        }

        for name, outcome in outcomes.items():
            logger.info(f"--- outcome: {name} ({outcome.task}) ---")
            try:
                results["outcomes"][name] = self._train_one_outcome(
                    joined, outcome, splitter
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

        Does not retrain. Scores the full BGG game universe so predictions
        cover games the user does not currently own.
        """
        logger.info(f"Refreshing predictions for user '{self.username}'")

        universe_df = self._load_game_universe()

        outcomes = load_outcomes(self._project_config.raw_config)
        if outcome_filter:
            missing = set(outcome_filter) - set(outcomes)
            if missing:
                raise ValueError(f"Unknown outcomes requested: {sorted(missing)}")
            outcomes = {k: v for k, v in outcomes.items() if k in outcome_filter}

        results: Dict[str, Any] = {}
        for name, outcome in outcomes.items():
            version = self.storage.latest_version(name)
            if version is None:
                logger.warning(
                    f"No trained model for outcome {name!r}; skipping"
                )
                continue
            results[name] = self._predict_one_outcome(universe_df, outcome, version)

        return results

    def get_pipeline_status(self) -> Dict[str, Any]:
        return self.storage.get_artifact_status()

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _fetch_and_persist_collection(self) -> None:
        """Fetch raw collection from BGG and persist to BQ."""
        fetch_and_persist(self.username, self._environment)

    def _process_collection(self) -> pl.DataFrame:
        """Load + canonicalize + join collection with game universe."""
        processor = self._get_processor()
        result = processor.process(self.username)
        if result is None:
            raise ValueError(
                f"No stored collection for user '{self.username}'. "
                "Run with refresh_collection=True."
            )
        return result

    def _load_game_universe(self) -> pl.DataFrame:
        """Full BGG game universe (features for every game)."""
        return BGGDataLoader(self.bq_config).load_features(
            use_predicted_complexity=self.config.use_predicted_complexity,
            use_embeddings=self.config.use_embeddings,
        )

    def _train_one_outcome(
        self,
        joined: pl.DataFrame,
        outcome: OutcomeDefinition,
        splitter: CollectionSplitter,
    ) -> Dict[str, Any]:
        labeled = apply_outcome(joined, outcome)
        train_df, val_df, test_df = splitter.split(labeled, outcome)

        if (
            outcome.task == "classification"
            and self.config.downsample_negatives_ratio is not None
        ):
            before = train_df.height
            train_df = downsample_negatives(
                train_df,
                ratio=self.config.downsample_negatives_ratio,
                protect_min_ratings=self.config.downsample_protect_min_ratings,
                random_seed=self.config.classification_split_config.random_seed,
            )
            logger.info(
                f"Downsampled train negatives: {before} -> {train_df.height} rows"
            )

        if train_df.height == 0:
            raise ValueError(f"Train split is empty for outcome {outcome.name!r}")

        model = CollectionModel(
            username=self.username,
            outcome=outcome,
            classification_config=self.config.classification_model_config,
            regression_config=self.config.regression_model_config,
        )
        best_params, _ = model.tune(train_df, val_df)

        if outcome.task == "classification":
            model.find_threshold(val_df)  # stashes onto model.threshold

        val_metrics = model.evaluate(val_df)
        test_metrics = model.evaluate(test_df)

        version = self.storage.next_version(outcome.name)

        self.storage.save_splits(outcome.name, train_df, val_df, test_df, version=version)
        metadata = {
            "task": outcome.task,
            "best_params": best_params,
            # Backward compat: "metrics" continues to mean test metrics.
            "metrics": test_metrics,
            "val_metrics": val_metrics,
        }
        self.storage.save_model(
            outcome.name,
            model.fitted_pipeline,
            metadata,
            threshold=model.threshold,
            version=version,
        )

        logger.info(
            f"Trained {outcome.name} v{version}: "
            f"train={train_df.height} val={val_df.height} test={test_df.height}; "
            f"val_metrics={val_metrics}; test_metrics={test_metrics}"
        )

        return {
            "version": version,
            "task": outcome.task,
            "threshold": model.threshold,
            "best_params": best_params,
            # Backward compat: "metrics" still points at test metrics.
            "metrics": test_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "split_sizes": {
                "train": train_df.height,
                "val": val_df.height,
                "test": test_df.height,
            },
        }

    def _predict_one_outcome(
        self,
        universe_df: pl.DataFrame,
        outcome: OutcomeDefinition,
        version: int,
    ) -> Dict[str, Any]:
        pipeline_obj, metadata, threshold = self.storage.load_model(
            outcome.name, version=version
        )
        X = universe_df.to_pandas()
        if outcome.task == "classification":
            proba = pipeline_obj.predict_proba(X)[:, 1]
            predictions_df = pl.DataFrame(
                {
                    "game_id": universe_df["game_id"].to_list(),
                    "score": proba,
                }
            )
        else:
            preds = pipeline_obj.predict(X)
            predictions_df = pl.DataFrame(
                {
                    "game_id": universe_df["game_id"].to_list(),
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
