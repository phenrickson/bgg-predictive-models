"""Local-filesystem storage layer for user collection artifacts.

Writes everything under ``{local_root}/{environment}/{username}/``. This mirrors
the pattern used by :class:`~src.models.experiments.ExperimentTracker` for
universe-level models. GCS round-trips are handled separately by a
``sync_collections`` utility — not this module.

Path layout per user::

    {local_root}/{env}/{username}/
        metadata.json                           # global user metadata, outcome-agnostic
        collection/latest.parquet               # raw snapshot, outcome-agnostic
        {outcome}/v{N}/                         # production-winner path (single best model)
            model.pkl
            threshold.json                      # classification only
            registration.json
            splits/{train,validation,test}.parquet
            predictions/...
            analysis/...
        {outcome}/_splits/v{N}/                 # canonical splits shared by candidate runs
            {train,validation,test}.parquet
        {outcome}/{candidate}/v{N}/             # per-candidate experiment runs
            model.pkl
            threshold.json                      # classification only
            registration.json                   # candidate spec + metrics + splits_version
            tuning_results.parquet              # full hyperparameter search trace
            train_used.parquet                  # actual training frame after downsampling/slicing
            predictions/...
            analysis/...
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class CollectionArtifactStorage:
    """Handles local-filesystem storage for user collection artifacts.

    Stores model pipelines, predictions, and analysis artifacts on the local
    filesystem, organized by environment, username, and outcome. Each
    ``(user, outcome)`` pair gets its own versioned subdirectory::

        {local_root}/{environment}/{username}/
            ├── metadata.json
            ├── collection/latest.parquet
            ├── {outcome}/v{N}/
            │   ├── model.pkl
            │   ├── threshold.json          (classification only)
            │   ├── registration.json
            │   ├── splits/{train,validation,test}.parquet
            │   ├── predictions/{predictions.parquet,top_recommendations.json}
            │   └── analysis/{summary_stats.json,feature_importance.parquet,category_affinity.json}
            └── ...
    """

    def __init__(
        self,
        username: str,
        local_root: Union[str, Path] = "models/collections",
        environment: Optional[str] = None,
    ):
        """Initialize artifact storage for a specific user.

        Args:
            username: BGG username
            local_root: Root directory for all collection artifacts. Defaults
                to ``models/collections`` (relative to cwd).
            environment: Environment name (e.g. ``"dev"``, ``"prod"``). If
                ``None``, read from :func:`src.utils.config.load_config`
                via ``Config.get_environment_prefix()``.
        """
        self.username = username

        if environment is None:
            project_config = load_config()
            environment = project_config.get_environment_prefix()
        self.environment = environment

        self.local_root = Path(local_root)
        self.base_dir: Path = self.local_root / self.environment / username
        self.base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized artifact storage for user '{username}'")
        logger.info(f"Local path: {self.base_dir}")

    # --- Internal path / upload / download helpers ---

    def _path(self, *parts: str) -> Path:
        """Build an absolute local path under :attr:`base_dir`."""
        return self.base_dir.joinpath(*parts)

    def _exists(self, *parts: str) -> bool:
        return self._path(*parts).exists()

    def _upload_bytes(self, rel_path: Union[str, Path], data: bytes) -> str:
        """Write raw bytes to a file under :attr:`base_dir`."""
        target = self.base_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    def _upload_json(self, rel_path: Union[str, Path], data: Dict[str, Any]) -> str:
        """Write a JSON document to a file under :attr:`base_dir`."""
        target = self.base_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, default=str))
        return str(target)

    def _download_json(self, rel_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Read a JSON document, returning ``None`` if missing."""
        target = self.base_dir / rel_path
        if not target.exists():
            return None
        return json.loads(target.read_text())

    def _upload_parquet(self, rel_path: Union[str, Path], df: pl.DataFrame) -> str:
        """Write a polars DataFrame as parquet."""
        target = self.base_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(target)
        return str(target)

    def _download_parquet(self, rel_path: Union[str, Path]) -> Optional[pl.DataFrame]:
        """Read a parquet file into a polars DataFrame, or ``None`` if missing."""
        target = self.base_dir / rel_path
        if not target.exists():
            return None
        return pl.read_parquet(target)

    def _upload_pickle(self, rel_path: Union[str, Path], obj: Any) -> str:
        """Write a pickled object."""
        target = self.base_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pickle.dumps(obj))
        return str(target)

    def _download_pickle(self, rel_path: Union[str, Path]) -> Optional[Any]:
        """Load a pickled object, or ``None`` if missing."""
        target = self.base_dir / rel_path
        if not target.exists():
            return None
        return pickle.loads(target.read_bytes())

    # --- Version helpers ---

    def _list_versions(self, outcome: str) -> List[int]:
        """Return all versions that exist under ``{outcome}/v*``, sorted asc."""
        outcome_dir = self.base_dir / outcome
        if not outcome_dir.exists():
            return []
        versions: List[int] = []
        for child in outcome_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("v"):
                continue
            try:
                versions.append(int(name[1:]))
            except ValueError:
                continue
        versions.sort()
        return versions

    def latest_version(self, outcome: str) -> Optional[int]:
        """Return the highest version number under ``{outcome}/``, or ``None``.

        Args:
            outcome: Outcome name (e.g. ``"own"``, ``"love"``, ``"rating"``)

        Returns:
            Highest version number found, or ``None``.
        """
        versions = self._list_versions(outcome)
        return versions[-1] if versions else None

    def next_version(self, outcome: str) -> int:
        """Return the next version number for a given outcome.

        Args:
            outcome: Outcome name

        Returns:
            ``latest_version(outcome) + 1``, or ``1`` if no versions exist.
        """
        return (self.latest_version(outcome) or 0) + 1

    # --- Collection Operations (outcome-agnostic) ---

    def save_collection(self, collection_df: pl.DataFrame) -> str:
        """Save collection snapshot to disk.

        Args:
            collection_df: Collection DataFrame with features

        Returns:
            Local path where collection was saved.
        """
        path = self._upload_parquet(Path("collection") / "latest.parquet", collection_df)
        logger.info(f"Saved collection ({len(collection_df)} games) to {path}")
        return path

    def load_collection(self) -> Optional[pl.DataFrame]:
        """Load the latest collection snapshot.

        Returns:
            Collection DataFrame or ``None`` if not found.
        """
        df = self._download_parquet(Path("collection") / "latest.parquet")
        if df is not None:
            logger.info(f"Loaded collection ({len(df)} games) from disk")
        return df

    # --- Split Operations ---

    def save_splits(
        self,
        outcome: str,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Save train/val/test splits under ``{outcome}/v{version}/splits/``.

        Args:
            outcome: Outcome name (e.g. ``"own"``, ``"love"``, ``"rating"``)
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            version: Version number; auto-incremented via ``next_version`` if ``None``.

        Returns:
            Dictionary with ``"version"`` and per-split local paths.
        """
        if version is None:
            version = self.next_version(outcome)

        paths: Dict[str, Any] = {"version": version}
        for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            rel = Path(outcome) / f"v{version}" / "splits" / f"{name}.parquet"
            paths[name] = self._upload_parquet(rel, df)
            logger.info(
                f"Saved {outcome}/v{version} {name} split ({len(df)} rows) to {paths[name]}"
            )
        return paths

    def load_splits(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load splits from disk.

        Args:
            outcome: Outcome name
            version: Version number; defaults to latest for the outcome.

        Returns:
            Dictionary with ``"version"`` and ``train``/``validation``/``test``
            DataFrames, or ``None`` if not found.
        """
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                logger.warning(f"No splits found for outcome '{outcome}'")
                return None

        splits: Dict[str, Any] = {"version": version}
        for name in ["train", "validation", "test"]:
            rel = Path(outcome) / f"v{version}" / "splits" / f"{name}.parquet"
            df = self._download_parquet(rel)
            if df is None:
                logger.warning(f"Split '{name}' not found for {outcome}/v{version}")
                return None
            splits[name] = df
            logger.info(f"Loaded {outcome}/v{version} {name} split ({len(df)} rows)")
        return splits

    # --- Model Operations ---

    def save_model(
        self,
        outcome: str,
        pipeline: Any,
        metadata: Dict[str, Any],
        threshold: Optional[float] = None,
        version: Optional[int] = None,
    ) -> str:
        """Save model pipeline and metadata to disk.

        Args:
            outcome: Outcome name (e.g. ``"own"``, ``"love"``, ``"rating"``).
            pipeline: Trained sklearn pipeline.
            metadata: Model metadata (metrics, params, etc.).
            threshold: Optimal classification threshold; ``None`` for regression.
            version: Model version (auto-incremented if not provided).

        Returns:
            Local path (as string) to the version directory where artifacts were saved.
        """
        if version is None:
            version = self.next_version(outcome)

        version_rel = Path(outcome) / f"v{version}"

        # Save pipeline
        self._upload_pickle(version_rel / "model.pkl", pipeline)

        # Save threshold (classification only)
        if threshold is not None:
            self._upload_json(version_rel / "threshold.json", {"threshold": threshold})

        # Build and save registration metadata
        registration = {
            "username": self.username,
            "outcome": outcome,
            "version": version,
            "created_at": datetime.now().isoformat(),
            **metadata,
        }
        if threshold is not None:
            registration["threshold"] = threshold

        self._upload_json(version_rel / "registration.json", registration)

        version_dir = self.base_dir / version_rel
        logger.info(f"Saved {outcome} model v{version} to {version_dir}/")
        return f"{version_dir}/"

    def load_model(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any], Optional[float]]:
        """Load a model from disk.

        Args:
            outcome: Outcome name
            version: Model version (latest if not specified)

        Returns:
            Tuple of ``(pipeline, metadata, threshold)``; ``threshold`` is
            ``None`` for regression outcomes.
        """
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                raise ValueError(
                    f"No models found for user '{self.username}', outcome '{outcome}'"
                )

        version_rel = Path(outcome) / f"v{version}"

        pipeline = self._download_pickle(version_rel / "model.pkl")
        if pipeline is None:
            raise ValueError(
                f"Model not found for user '{self.username}', "
                f"outcome '{outcome}', version {version}"
            )

        metadata = self._download_json(version_rel / "registration.json") or {}

        threshold_data = self._download_json(version_rel / "threshold.json")
        threshold = threshold_data.get("threshold") if threshold_data else None

        logger.info(f"Loaded {outcome} model v{version} for user '{self.username}'")
        return pipeline, metadata, threshold

    def list_model_versions(self, outcome: str) -> List[Dict[str, Any]]:
        """List all model versions (registration metadata) for a given outcome.

        Args:
            outcome: Outcome name

        Returns:
            List of registration dicts, sorted by version ascending.
        """
        versions: List[Dict[str, Any]] = []
        for v in self._list_versions(outcome):
            registration = self._download_json(Path(outcome) / f"v{v}" / "registration.json")
            if registration is not None:
                versions.append(registration)
        return sorted(versions, key=lambda x: x.get("version", 0))

    # --- Predictions Operations ---

    def save_predictions(
        self,
        outcome: str,
        version: int,
        predictions_df: pl.DataFrame,
        top_recommendations: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Save predictions and recommendations to disk.

        Args:
            outcome: Outcome name.
            version: Explicit version (predictions belong to a specific model version).
            predictions_df: DataFrame with all game predictions.
            top_recommendations: List of top recommended games.

        Returns:
            Mapping of artifact name to local path.
        """
        paths: Dict[str, str] = {}

        pred_rel = Path(outcome) / f"v{version}" / "predictions" / "predictions.parquet"
        paths["predictions"] = self._upload_parquet(pred_rel, predictions_df)
        logger.info(f"Saved {outcome}/v{version} predictions ({len(predictions_df)} games)")

        rec_rel = Path(outcome) / f"v{version}" / "predictions" / "top_recommendations.json"
        rec_data = {
            "username": self.username,
            "outcome": outcome,
            "version": version,
            "generated_at": datetime.now().isoformat(),
            "total_games_scored": len(predictions_df),
            "recommendations": top_recommendations,
        }
        paths["recommendations"] = self._upload_json(rec_rel, rec_data)
        logger.info(f"Saved {outcome}/v{version} top {len(top_recommendations)} recommendations")

        return paths

    def load_predictions(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[pl.DataFrame]:
        """Load predictions from disk.

        Args:
            outcome: Outcome name.
            version: Version number; defaults to latest for the outcome.

        Returns:
            Predictions DataFrame or ``None`` if not found.
        """
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                return None
        return self._download_parquet(
            Path(outcome) / f"v{version}" / "predictions" / "predictions.parquet"
        )

    def load_recommendations(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load top recommendations from disk.

        Args:
            outcome: Outcome name.
            version: Version number; defaults to latest for the outcome.

        Returns:
            Recommendations dictionary or ``None`` if not found.
        """
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                return None
        return self._download_json(
            Path(outcome) / f"v{version}" / "predictions" / "top_recommendations.json"
        )

    # --- Analysis Operations ---

    def save_analysis_artifacts(
        self,
        outcome: str,
        version: int,
        summary_stats: Dict[str, Any],
        feature_importance: pl.DataFrame,
        category_affinity: Dict[str, Any],
    ) -> Dict[str, str]:
        """Save analysis artifacts to disk.

        Args:
            outcome: Outcome name.
            version: Explicit version (analysis belongs to a specific model version).
            summary_stats: Collection summary statistics.
            feature_importance: Feature importance DataFrame.
            category_affinity: Category/mechanic preferences.

        Returns:
            Mapping of artifact name to local path.
        """
        paths: Dict[str, str] = {}

        paths["summary_stats"] = self._upload_json(
            Path(outcome) / f"v{version}" / "analysis" / "summary_stats.json",
            summary_stats,
        )
        paths["feature_importance"] = self._upload_parquet(
            Path(outcome) / f"v{version}" / "analysis" / "feature_importance.parquet",
            feature_importance,
        )
        paths["category_affinity"] = self._upload_json(
            Path(outcome) / f"v{version}" / "analysis" / "category_affinity.json",
            category_affinity,
        )

        logger.info(f"Saved {outcome}/v{version} analysis artifacts for user '{self.username}'")
        return paths

    def load_summary_stats(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load summary statistics from disk."""
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                return None
        return self._download_json(
            Path(outcome) / f"v{version}" / "analysis" / "summary_stats.json"
        )

    def load_feature_importance(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[pl.DataFrame]:
        """Load feature importance from disk."""
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                return None
        return self._download_parquet(
            Path(outcome) / f"v{version}" / "analysis" / "feature_importance.parquet"
        )

    def load_category_affinity(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load category affinity from disk."""
        if version is None:
            version = self.latest_version(outcome)
            if version is None:
                return None
        return self._download_json(
            Path(outcome) / f"v{version}" / "analysis" / "category_affinity.json"
        )

    # --- Metadata Operations (outcome-agnostic) ---

    def save_user_metadata(self, metadata: Dict[str, Any]) -> str:
        """Save user metadata to disk.

        Args:
            metadata: User metadata dictionary.

        Returns:
            Local path where metadata was saved.
        """
        metadata = dict(metadata)
        metadata["username"] = self.username
        metadata["updated_at"] = datetime.now().isoformat()
        return self._upload_json("metadata.json", metadata)

    def load_user_metadata(self) -> Optional[Dict[str, Any]]:
        """Load user metadata from disk.

        Returns:
            User metadata dictionary or ``None`` if not found.
        """
        return self._download_json("metadata.json")

    # --- Status ---

    def list_outcomes(self) -> List[str]:
        """Return outcome names (top-level directories under :attr:`base_dir`)
        that contain at least one production ``v{N}`` directory, a candidate
        run, or canonical splits.
        """
        reserved = {"collection"}
        outcomes: List[str] = []
        if not self.base_dir.exists():
            return outcomes
        for child in self.base_dir.iterdir():
            if not child.is_dir() or child.name in reserved:
                continue
            outcome = child.name
            has_production_versions = bool(self._list_versions(outcome))
            has_candidates = bool(self.list_candidates(outcome))
            has_splits = bool(self._list_split_versions(outcome))
            if has_production_versions or has_candidates or has_splits:
                outcomes.append(outcome)
        return sorted(outcomes)

    # --- Candidate experiment runs ---
    #
    # Candidate runs live at ``{outcome}/{candidate}/v{N}/`` and are versioned
    # independently of the production-winner path at ``{outcome}/v{N}/``.
    # Splits used by candidate runs live at ``{outcome}/_splits/v{N}/`` so
    # multiple candidates can share the exact same val/test for honest
    # comparison.

    _CANDIDATE_RESERVED_NAMES = {"_splits"}

    def _validate_candidate_name(self, candidate: str) -> None:
        if not candidate or "/" in candidate or candidate.startswith("_") or candidate.startswith("v"):
            raise ValueError(
                f"Invalid candidate name {candidate!r}: must be non-empty, "
                f"cannot start with '_' or 'v', cannot contain '/'."
            )
        if candidate in self._CANDIDATE_RESERVED_NAMES:
            raise ValueError(f"Candidate name {candidate!r} is reserved.")

    def _list_candidate_versions(self, outcome: str, candidate: str) -> List[int]:
        candidate_dir = self.base_dir / outcome / candidate
        if not candidate_dir.exists():
            return []
        versions: List[int] = []
        for child in candidate_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("v"):
                continue
            try:
                versions.append(int(child.name[1:]))
            except ValueError:
                continue
        versions.sort()
        return versions

    def latest_candidate_version(self, outcome: str, candidate: str) -> Optional[int]:
        """Return the highest version number under ``{outcome}/{candidate}/``,
        or ``None`` if the candidate has no runs.
        """
        self._validate_candidate_name(candidate)
        versions = self._list_candidate_versions(outcome, candidate)
        return versions[-1] if versions else None

    def _next_candidate_version(self, outcome: str, candidate: str) -> int:
        return (self.latest_candidate_version(outcome, candidate) or 0) + 1

    def list_candidates(self, outcome: str) -> List[str]:
        """Return all candidate names that have at least one versioned run for
        ``outcome``. The reserved ``_splits`` directory is excluded.
        """
        outcome_dir = self.base_dir / outcome
        if not outcome_dir.exists():
            return []
        candidates: List[str] = []
        for child in outcome_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name in self._CANDIDATE_RESERVED_NAMES or name.startswith("v"):
                continue
            if self._list_candidate_versions(outcome, name):
                candidates.append(name)
        return sorted(candidates)

    # --- Canonical splits (shared across candidates) ---

    def _list_split_versions(self, outcome: str) -> List[int]:
        splits_dir = self.base_dir / outcome / "_splits"
        if not splits_dir.exists():
            return []
        versions: List[int] = []
        for child in splits_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("v"):
                continue
            try:
                versions.append(int(child.name[1:]))
            except ValueError:
                continue
        versions.sort()
        return versions

    def latest_canonical_splits_version(self, outcome: str) -> Optional[int]:
        versions = self._list_split_versions(outcome)
        return versions[-1] if versions else None

    def save_canonical_splits(
        self,
        outcome: str,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Save splits under ``{outcome}/_splits/v{version}/``. These are the
        canonical splits that candidate runs reference by version. Auto-
        increments if ``version`` is ``None``.
        """
        if version is None:
            version = (self.latest_canonical_splits_version(outcome) or 0) + 1

        paths: Dict[str, Any] = {"version": version}
        for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            rel = Path(outcome) / "_splits" / f"v{version}" / f"{name}.parquet"
            paths[name] = self._upload_parquet(rel, df)
            logger.info(
                f"Saved {outcome}/_splits/v{version} {name} ({len(df)} rows)"
            )
        return paths

    def load_canonical_splits(
        self,
        outcome: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load canonical splits. Defaults to the latest version. Returns
        ``None`` if no splits exist."""
        if version is None:
            version = self.latest_canonical_splits_version(outcome)
            if version is None:
                return None
        result: Dict[str, Any] = {"version": version}
        for name in ["train", "validation", "test"]:
            rel = Path(outcome) / "_splits" / f"v{version}" / f"{name}.parquet"
            df = self._download_parquet(rel)
            if df is None:
                logger.warning(
                    f"Canonical split {name!r} missing for {outcome}/_splits/v{version}"
                )
                return None
            result[name] = df
        return result

    # --- Candidate model save/load ---

    def save_candidate_run(
        self,
        outcome: str,
        candidate: str,
        pipeline: Any,
        registration: Dict[str, Any],
        tuning_results: Optional[pl.DataFrame] = None,
        train_used: Optional[pl.DataFrame] = None,
        threshold: Optional[float] = None,
        version: Optional[int] = None,
    ) -> str:
        """Persist all artifacts for one candidate run.

        ``registration`` should already contain the candidate spec, metrics,
        params, and ``splits_version``. ``tuning_results`` is the per-config
        hyperparameter search trace returned by
        :meth:`~CollectionModel.tune` / :meth:`~CollectionModel.tune_cv`.
        ``train_used`` is the actual training frame after downsampling or
        feature-slicing (different from the canonical training split).

        Finalization (refit on train+val+test through ``finalize_through``)
        is a separate step — see :meth:`save_finalized_pipeline`.

        Returns the absolute path to the version directory.
        """
        self._validate_candidate_name(candidate)

        if version is None:
            version = self._next_candidate_version(outcome, candidate)

        version_rel = Path(outcome) / candidate / f"v{version}"

        self._upload_pickle(version_rel / "model.pkl", pipeline)

        if threshold is not None:
            self._upload_json(
                version_rel / "threshold.json", {"threshold": threshold}
            )

        full_registration = {
            "username": self.username,
            "outcome": outcome,
            "candidate": candidate,
            "version": version,
            "created_at": datetime.now().isoformat(),
            **registration,
        }
        if threshold is not None:
            full_registration["threshold"] = threshold
        self._upload_json(version_rel / "registration.json", full_registration)

        if tuning_results is not None:
            self._upload_parquet(
                version_rel / "tuning_results.parquet", tuning_results
            )

        if train_used is not None:
            self._upload_parquet(version_rel / "train_used.parquet", train_used)

        version_dir = self.base_dir / version_rel
        logger.info(
            f"Saved candidate run {outcome}/{candidate} v{version} to {version_dir}/"
        )
        return f"{version_dir}/"

    def save_finalized_pipeline(
        self,
        outcome: str,
        candidate: str,
        version: int,
        pipeline: Any,
        finalize_through: int,
    ) -> str:
        """Add a finalized pipeline to an existing candidate run directory.

        Writes ``finalized.pkl`` next to the existing ``model.pkl`` and
        updates ``registration.json`` with ``finalize_through`` and the
        timestamp the finalize was applied. Returns the path to the
        ``finalized.pkl`` file.
        """
        self._validate_candidate_name(candidate)
        version_rel = Path(outcome) / candidate / f"v{version}"
        if not (self.base_dir / version_rel).exists():
            raise ValueError(
                f"No run at {version_rel}; can't add finalized pipeline"
            )

        finalized_path = self._upload_pickle(
            version_rel / "finalized.pkl", pipeline
        )

        reg_path = version_rel / "registration.json"
        registration = self._download_json(reg_path) or {}
        registration["finalize_through"] = int(finalize_through)
        registration["finalized_at"] = datetime.now().isoformat()
        self._upload_json(reg_path, registration)

        logger.info(
            f"Saved finalized pipeline to {self.base_dir / finalized_path}"
        )
        return f"{self.base_dir / finalized_path}"

    def load_finalized_pipeline(
        self,
        outcome: str,
        candidate: str,
        version: Optional[int] = None,
    ) -> Optional[Any]:
        """Load the finalized pipeline for an existing run, or ``None`` if
        the candidate has not been finalized yet."""
        self._validate_candidate_name(candidate)
        if version is None:
            version = self.latest_candidate_version(outcome, candidate)
            if version is None:
                return None
        return self._download_pickle(
            Path(outcome) / candidate / f"v{version}" / "finalized.pkl"
        )

    def load_candidate_run(
        self,
        outcome: str,
        candidate: str,
        version: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any], Optional[float]]:
        """Load a candidate run's pipeline, registration, and threshold.

        Defaults to the latest version. Raises ``ValueError`` if no run exists.
        """
        self._validate_candidate_name(candidate)
        if version is None:
            version = self.latest_candidate_version(outcome, candidate)
            if version is None:
                raise ValueError(
                    f"No runs for candidate {candidate!r} on outcome {outcome!r}"
                )
        version_rel = Path(outcome) / candidate / f"v{version}"

        pipeline = self._download_pickle(version_rel / "model.pkl")
        if pipeline is None:
            raise ValueError(
                f"Pipeline missing at {version_rel} for user {self.username!r}"
            )
        registration = self._download_json(version_rel / "registration.json") or {}
        threshold_data = self._download_json(version_rel / "threshold.json")
        threshold = threshold_data.get("threshold") if threshold_data else None
        return pipeline, registration, threshold

    def load_candidate_registration(
        self,
        outcome: str,
        candidate: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load just the registration (no pipeline). Cheap; useful for
        comparison loops that only need metrics + spec."""
        self._validate_candidate_name(candidate)
        if version is None:
            version = self.latest_candidate_version(outcome, candidate)
            if version is None:
                return None
        return self._download_json(
            Path(outcome) / candidate / f"v{version}" / "registration.json"
        )

    def load_candidate_tuning_results(
        self,
        outcome: str,
        candidate: str,
        version: Optional[int] = None,
    ) -> Optional[pl.DataFrame]:
        """Load the per-config tuning trace, or ``None`` if missing."""
        self._validate_candidate_name(candidate)
        if version is None:
            version = self.latest_candidate_version(outcome, candidate)
            if version is None:
                return None
        return self._download_parquet(
            Path(outcome) / candidate / f"v{version}" / "tuning_results.parquet"
        )

    def list_candidate_runs(
        self, outcome: str, candidate: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return registration dicts for one candidate (all versions) or every
        candidate under an outcome (all versions). Sorted by candidate, version.
        """
        candidates = (
            [candidate] if candidate is not None else self.list_candidates(outcome)
        )
        runs: List[Dict[str, Any]] = []
        for cand in candidates:
            for v in self._list_candidate_versions(outcome, cand):
                reg = self.load_candidate_registration(outcome, cand, version=v)
                if reg is not None:
                    runs.append(reg)
        return sorted(
            runs, key=lambda r: (r.get("candidate", ""), r.get("version", 0))
        )

    def get_artifact_status(self) -> Dict[str, Any]:
        """Get status of all artifacts for this user.

        Enumerates top-level subdirectories under :attr:`base_dir` (excluding
        ``collection`` and ``metadata.json``) and treats each as an outcome.
        For each outcome, lists its versioned subdirectories.

        Returns:
            Dictionary with per-outcome version info::

                {
                    "username": str,
                    "base_path": str,
                    "collection_exists": bool,
                    "outcomes": {
                        "own": {"latest_version": int | None, "versions": [1, 2, ...]},
                        ...
                    },
                }
        """
        base_path = str(self.base_dir) + "/"
        collection_exists = (self.base_dir / "collection" / "latest.parquet").exists()

        outcomes: Dict[str, Dict[str, Any]] = {}
        for outcome in self.list_outcomes():
            versions = self._list_versions(outcome)
            outcomes[outcome] = {
                "latest_version": max(versions) if versions else None,
                "versions": versions,
            }

        return {
            "username": self.username,
            "base_path": base_path,
            "collection_exists": collection_exists,
            "outcomes": outcomes,
        }
