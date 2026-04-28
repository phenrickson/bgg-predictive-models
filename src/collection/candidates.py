"""Candidate model specs and training for collection-model experiments.

A :class:`CollectionCandidate` describes one combination of (model type,
hyperparameter tuning strategy, optional feature subset, optional fixed
params, training-set downsampling) that we want to train and evaluate as a
distinct experiment. Candidates are persisted alongside their fitted
pipeline + metrics so multiple candidates can be compared after the fact
without retraining.

:func:`train_candidate` runs one candidate end-to-end and persists artifacts
under ``{outcome}/{candidate.name}/v{N}/`` via
:class:`CollectionArtifactStorage`. :func:`train_candidates` runs many.

Comparison across runs lives in :mod:`src.collection.candidate_comparison`.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

import polars as pl

from src.collection.collection_artifact_storage import CollectionArtifactStorage
from src.collection.collection_model import (
    ClassificationModelConfig,
    CollectionModel,
    RegressionModelConfig,
)
from src.collection.collection_split import downsample_negatives
from src.collection.outcomes import OutcomeDefinition

logger = logging.getLogger(__name__)


# Columns the model needs even if a candidate restricts ``feature_columns``.
# Includes ``users_rated`` and ``year_published`` so stratified eval and
# downsampling work, plus ``game_id``/``label`` which are required.
PROTECTED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "label",
    "year_published",
    "users_rated",
)


_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass
class CollectionCandidate:
    """One named experiment configuration.

    A candidate carries everything needed to reproduce a run *except* the
    splits — those live separately in storage so multiple candidates can
    share the same val/test for honest comparison.

    Args:
        name: Filesystem-safe candidate identifier. Used in storage paths
            (e.g. ``{outcome}/{name}/v{N}/``). Letters, digits, ``_``, ``-``;
            must start with a letter. Cannot start with ``v`` (reserved for
            version directories) or ``_`` (reserved for ``_splits``).
        classification_config: Required when the outcome is classification.
        regression_config: Required when the outcome is regression.
        feature_columns: Optional explicit feature subset. ``None`` means
            "use whatever columns are in the training frame". When set, the
            runner slices the train/val/test frames to ``feature_columns``
            plus any protected columns (``game_id``, ``label``, etc.).
        tuning: ``"holdout"`` (val-set tuning), ``"cv"`` (k-fold on train),
            or ``"none"`` (skip tuning, fit ``fixed_params``).
        cv_folds: Folds for ``tuning="cv"``. Ignored otherwise.
        fixed_params: Required when ``tuning="none"``; otherwise unused.
            Keys use the same ``model__<param>`` shape that the tuners
            return as ``best_params``.
        downsample_negatives_ratio: If set, applied to the *training frame
            only* before tuning. ``None`` disables. The protected-min-ratings
            floor for downsampling is ``downsample_protect_min_ratings``.
        downsample_protect_min_ratings: Negatives with
            ``users_rated >= this`` are always kept; only the low-rating
            tail gets sampled out.
        notes: Free-form description (recorded in the registration metadata).
    """

    name: str
    classification_config: Optional[ClassificationModelConfig] = None
    regression_config: Optional[RegressionModelConfig] = None
    feature_columns: Optional[List[str]] = None
    tuning: Literal["holdout", "cv", "none"] = "cv"
    cv_folds: int = 5
    fixed_params: Optional[Dict[str, Any]] = None
    downsample_negatives_ratio: Optional[float] = None
    downsample_protect_min_ratings: int = 25
    notes: str = ""

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"Invalid candidate name {self.name!r}: must match {_NAME_RE.pattern}"
            )
        if self.name.startswith("v") or self.name.startswith("_"):
            raise ValueError(
                f"Candidate name {self.name!r} cannot start with 'v' or '_' "
                "(reserved for version and internal directories)."
            )
        if self.tuning not in ("holdout", "cv", "none"):
            raise ValueError(f"Invalid tuning strategy: {self.tuning!r}")
        if self.tuning == "cv" and self.cv_folds < 2:
            raise ValueError(
                f"cv_folds must be >= 2 when tuning='cv', got {self.cv_folds}"
            )
        if self.tuning == "none" and self.fixed_params is None:
            raise ValueError(
                "fixed_params is required when tuning='none' "
                "(set fixed_params={} to fit with model defaults)"
            )
        if self.downsample_negatives_ratio is not None and self.downsample_negatives_ratio <= 0:
            raise ValueError(
                f"downsample_negatives_ratio must be > 0 (got {self.downsample_negatives_ratio})"
            )
        if self.downsample_protect_min_ratings < 0:
            raise ValueError(
                f"downsample_protect_min_ratings must be >= 0 "
                f"(got {self.downsample_protect_min_ratings})"
            )

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation suitable for ``registration.json``."""
        d = asdict(self)
        # asdict already turns the inner dataclass configs into dicts.
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollectionCandidate":
        """Inverse of :meth:`to_dict`. Reconstructs nested config dataclasses
        from their dict form. Unknown fields are ignored to allow the spec
        to evolve without breaking older saved runs.
        """
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {k: v for k, v in data.items() if k in known}

        if isinstance(kwargs.get("classification_config"), dict):
            kwargs["classification_config"] = ClassificationModelConfig(
                **kwargs["classification_config"]
            )
        if isinstance(kwargs.get("regression_config"), dict):
            kwargs["regression_config"] = RegressionModelConfig(
                **kwargs["regression_config"]
            )
        return cls(**kwargs)


def load_candidates(config: Dict[str, Any]) -> Dict[str, CollectionCandidate]:
    """Parse ``collections.candidates`` from config into a name-keyed registry.

    Mirrors :func:`src.collection.outcomes.load_outcomes`. Each list entry is
    re-hydrated via :meth:`CollectionCandidate.from_dict`. Duplicate names raise.

    Returns:
        ``{candidate.name: CollectionCandidate}``. Empty dict if the config
        section is absent — candidates are optional.

    Raises:
        ValueError: if ``collections.candidates`` is not a list, or if any
            entry produces a duplicate name.
    """
    section = config.get("collections", {}).get("candidates", [])
    if not section:
        return {}
    if not isinstance(section, list):
        raise ValueError(
            f"config.collections.candidates must be a list, got {type(section).__name__}"
        )
    out: Dict[str, CollectionCandidate] = {}
    for entry in section:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Each candidate entry must be a mapping, got {type(entry).__name__}"
            )
        candidate = CollectionCandidate.from_dict(entry)
        if candidate.name in out:
            raise ValueError(f"Duplicate candidate name in config: {candidate.name!r}")
        out[candidate.name] = candidate
    return out


# --- Training ---


@dataclass
class CandidateRunResult:
    """In-memory artifacts from training one candidate. Pure result —
    nothing has been persisted yet. Pass to :func:`save_candidate_run` to
    write to disk.

    ``model`` is the trained :class:`CollectionModel`. It carries the
    fitted pipeline and the optimized threshold (for classification) on
    its own state, so call ``result.model.feature_importance()``,
    ``result.model.top_games(df)``, ``result.model.evaluate(df)`` etc.
    Run-specific metadata (val/test metrics, best_params, splits_version,
    etc.) lives on the result, since it isn't a property of the model
    itself.
    """

    candidate: CollectionCandidate
    outcome: OutcomeDefinition
    model: CollectionModel
    best_params: Dict[str, Any]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_used: pl.DataFrame  # the (possibly downsampled / feature-sliced) train frame
    train_n: int
    val_n: int
    test_n: int
    splits_version: Optional[int] = None
    tuning_results: Optional[pl.DataFrame] = None


def train_candidate(
    candidate: CollectionCandidate,
    outcome: OutcomeDefinition,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    splits_version: Optional[int] = None,
    username: str = "unknown",
    protected_columns: Sequence[str] = PROTECTED_COLUMNS,
) -> CandidateRunResult:
    """Train one candidate. Pure compute — does not persist anything.

    Applies the candidate's feature slicing and training-set downsampling
    to the provided splits, runs the configured tuning strategy, picks a
    classification threshold on val, evaluates on val and test, and
    returns a :class:`CandidateRunResult` carrying the fitted pipeline
    and metrics in memory. Pass the result to :func:`save_candidate_run`
    to write artifacts.

    Raises:
        ValueError: If the candidate's config does not match the outcome's
            task, or requested feature columns are missing from the splits.
    """
    _validate_candidate_for_outcome(candidate, outcome)

    if candidate.feature_columns is not None:
        keep = _resolve_feature_columns(
            candidate.feature_columns, protected_columns, train_df.columns
        )
        train_df = train_df.select(keep)
        val_df = val_df.select(keep)
        test_df = test_df.select(keep)

    train_used = train_df
    if candidate.downsample_negatives_ratio is not None and outcome.task == "classification":
        before = train_used.height
        train_used = downsample_negatives(
            train_used,
            ratio=candidate.downsample_negatives_ratio,
            protect_min_ratings=candidate.downsample_protect_min_ratings,
        )
        logger.info(
            f"[{candidate.name}] downsampled train negatives: "
            f"{before} -> {train_used.height} rows"
        )

    model = CollectionModel(
        username=username,
        outcome=outcome,
        classification_config=candidate.classification_config,
        regression_config=candidate.regression_config,
    )

    best_params, tuning_results = _run_tuning(
        candidate, model, train_used, val_df
    )

    if outcome.task == "classification":
        model.find_threshold(val_df)  # stashes onto model.threshold

    val_metrics = model.evaluate(val_df)
    test_metrics = model.evaluate(test_df)

    tuning_results_pl: Optional[pl.DataFrame] = None
    if tuning_results is not None and len(tuning_results) > 0:
        tuning_results_pl = _coerce_tuning_results(tuning_results)

    return CandidateRunResult(
        candidate=candidate,
        outcome=outcome,
        model=model,
        best_params=best_params,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_used=train_used,
        train_n=train_used.height,
        val_n=val_df.height,
        test_n=test_df.height,
        splits_version=splits_version,
        tuning_results=tuning_results_pl,
    )


def train_candidates(
    candidates: Sequence[CollectionCandidate],
    outcome: OutcomeDefinition,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    splits_version: Optional[int] = None,
    username: str = "unknown",
) -> List[CandidateRunResult]:
    """Train each candidate sequentially. No persistence; no error handling.

    If one candidate raises, the rest don't run. Use a CLI-level loop if
    you need continue-on-error semantics.
    """
    return [
        train_candidate(
            c, outcome, train_df, val_df, test_df,
            splits_version=splits_version, username=username,
        )
        for c in candidates
    ]


def save_candidate_run(
    result: CandidateRunResult,
    storage: CollectionArtifactStorage,
) -> str:
    """Persist a :class:`CandidateRunResult` under
    ``{outcome}/{candidate.name}/v{N}/`` via ``storage``. Returns the
    artifact directory path.

    Stamps the registration with the storage user, current git SHA, and
    timestamp at save time.
    """
    registration: Dict[str, Any] = {
        "task": result.outcome.task,
        "outcome_name": result.outcome.name,
        "candidate_spec": result.candidate.to_dict(),
        "splits_version": result.splits_version,
        "tuning_strategy": result.candidate.tuning,
        "best_params": result.best_params,
        "metrics": result.test_metrics,
        "val_metrics": result.val_metrics,
        "n_train_used": int(result.train_n),
        "n_val": int(result.val_n),
        "n_test": int(result.test_n),
        "git_sha": _git_sha(),
        "trained_at": datetime.now().isoformat(),
    }
    return storage.save_candidate_run(
        outcome=result.outcome.name,
        candidate=result.candidate.name,
        pipeline=result.model.fitted_pipeline,
        registration=registration,
        tuning_results=result.tuning_results,
        train_used=result.train_used,
        threshold=result.model.threshold,
    )


def _validate_candidate_for_outcome(
    candidate: CollectionCandidate, outcome: OutcomeDefinition
) -> None:
    if outcome.task == "classification" and candidate.classification_config is None:
        raise ValueError(
            f"Candidate {candidate.name!r} has no classification_config but "
            f"outcome {outcome.name!r} is classification."
        )
    if outcome.task == "regression" and candidate.regression_config is None:
        raise ValueError(
            f"Candidate {candidate.name!r} has no regression_config but "
            f"outcome {outcome.name!r} is regression."
        )


def _resolve_feature_columns(
    requested: Iterable[str],
    protected: Iterable[str],
    available: Iterable[str],
) -> List[str]:
    available_set = set(available)
    requested_list = list(requested)
    missing = [c for c in requested_list if c not in available_set]
    if missing:
        raise ValueError(
            f"Requested feature_columns missing from splits: {missing}"
        )
    keep_order: List[str] = []
    seen: set[str] = set()
    for col in list(protected) + requested_list:
        if col in available_set and col not in seen:
            keep_order.append(col)
            seen.add(col)
    return keep_order


def _run_tuning(
    candidate: CollectionCandidate,
    model: CollectionModel,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
):
    """Dispatch on candidate.tuning. Stashes the fitted pipeline on ``model``;
    returns ``(best_params, tuning_results)``.
    """
    if candidate.tuning == "cv":
        return model.tune_cv(train_df, cv_folds=candidate.cv_folds)
    if candidate.tuning == "holdout":
        return model.tune(train_df, val_df)
    if candidate.tuning == "none":
        params = candidate.fixed_params or {}
        model.train(train_df, params=params)
        return dict(params), None
    raise ValueError(f"Unknown tuning strategy: {candidate.tuning!r}")


def _coerce_tuning_results(results) -> pl.DataFrame:
    """Tuners return pandas; storage writes parquet via polars. Convert,
    stringifying the ``params`` column since polars cannot infer a uniform
    schema for dict-typed columns.
    """
    if isinstance(results, pl.DataFrame):
        return results
    pdf = results.copy()
    if "params" in pdf.columns:
        pdf["params"] = pdf["params"].astype(str)
    return pl.from_pandas(pdf)


def _git_sha() -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
