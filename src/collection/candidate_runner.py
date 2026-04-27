"""Run a single :class:`CollectionCandidate` end-to-end and persist artifacts.

A "run" trains, tunes (or fits with fixed params), evaluates, and saves
everything under ``{outcome}/{candidate.name}/v{N}/`` via
:class:`CollectionArtifactStorage`. Splits are loaded from a canonical
``{outcome}/_splits/v{splits_version}/`` directory; this module never
creates splits — they must already exist in storage so multiple candidates
can share the exact same val/test for honest comparison.

Comparison across runs lives in :mod:`src.collection.candidate_comparison`.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import polars as pl

from src.collection.candidates import CollectionCandidate
from src.collection.collection_artifact_storage import CollectionArtifactStorage
from src.collection.collection_model import CollectionModel
from src.collection.collection_split import downsample_negatives
from src.collection.outcomes import OutcomeDefinition

logger = logging.getLogger(__name__)


# Columns the model needs even if the candidate restricts ``feature_columns``.
# Includes ``users_rated`` and ``year_published`` so the runner's stratified
# eval and downsampling work, plus ``game_id``/``label`` which are required.
PROTECTED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "label",
    "year_published",
    "users_rated",
)


@dataclass
class CandidateRunResult:
    """Pointer to a persisted candidate run plus the in-memory artifacts."""

    candidate: CollectionCandidate
    outcome: OutcomeDefinition
    version: int
    splits_version: int
    threshold: Optional[float]
    best_params: Dict[str, Any]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_n: int
    val_n: int
    test_n: int
    artifact_dir: str


def train_candidate(
    candidate: CollectionCandidate,
    outcome: OutcomeDefinition,
    storage: CollectionArtifactStorage,
    splits_version: Optional[int] = None,
    protected_columns: Sequence[str] = PROTECTED_COLUMNS,
) -> CandidateRunResult:
    """Train one candidate and persist its artifacts.

    Loads the canonical splits at ``{outcome}/_splits/v{splits_version}/``
    (latest if ``splits_version=None``), applies the candidate's feature
    slicing and training-set downsampling, runs the configured tuning
    strategy, picks a classification threshold on val, evaluates on val
    and test, and writes all artifacts under
    ``{outcome}/{candidate.name}/v{N}/``.

    Args:
        candidate: The experiment spec to run.
        outcome: Outcome definition (carries ``task`` and ``name``).
        storage: Configured artifact storage for the user.
        splits_version: Canonical splits version to consume. ``None`` =
            use the latest; raises if no splits have been persisted.
        protected_columns: Columns kept even when ``candidate.feature_columns``
            is set. Defaults to :data:`PROTECTED_COLUMNS`.

    Returns:
        :class:`CandidateRunResult` describing what was saved.

    Raises:
        ValueError: If splits cannot be loaded, the candidate's config does
            not match the outcome's task, or requested feature columns are
            missing from the splits.
    """
    _validate_candidate_for_outcome(candidate, outcome)

    splits = storage.load_canonical_splits(outcome.name, version=splits_version)
    if splits is None:
        raise ValueError(
            f"No canonical splits available for outcome {outcome.name!r}. "
            f"Persist them with storage.save_canonical_splits(...) first."
        )
    splits_version_used: int = splits["version"]
    train_df: pl.DataFrame = splits["train"]
    val_df: pl.DataFrame = splits["validation"]
    test_df: pl.DataFrame = splits["test"]

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
        username=storage.username,
        outcome=outcome,
        classification_config=candidate.classification_config,
        regression_config=candidate.regression_config,
    )

    pipeline_obj, best_params, tuning_results = _run_tuning(
        candidate, model, train_used, val_df
    )

    threshold: Optional[float] = None
    if outcome.task == "classification":
        threshold = model.find_threshold(pipeline_obj, val_df)

    val_metrics = model.evaluate(pipeline_obj, val_df, threshold=threshold)
    test_metrics = model.evaluate(pipeline_obj, test_df, threshold=threshold)

    registration: Dict[str, Any] = {
        "task": outcome.task,
        "outcome_name": outcome.name,
        "candidate_spec": candidate.to_dict(),
        "splits_version": splits_version_used,
        "tuning_strategy": candidate.tuning,
        "best_params": best_params,
        "metrics": test_metrics,
        "val_metrics": val_metrics,
        "n_train_used": int(train_used.height),
        "n_train_canonical": int(train_df.height),
        "n_val": int(val_df.height),
        "n_test": int(test_df.height),
        "git_sha": _git_sha(),
        "trained_at": datetime.now().isoformat(),
    }

    tuning_results_pl: Optional[pl.DataFrame] = None
    if tuning_results is not None and len(tuning_results) > 0:
        tuning_results_pl = _coerce_tuning_results(tuning_results)

    artifact_dir = storage.save_candidate_run(
        outcome=outcome.name,
        candidate=candidate.name,
        pipeline=pipeline_obj,
        registration=registration,
        tuning_results=tuning_results_pl,
        train_used=train_used,
        threshold=threshold,
    )

    version = storage.latest_candidate_version(outcome.name, candidate.name)
    assert version is not None  # we just wrote it

    return CandidateRunResult(
        candidate=candidate,
        outcome=outcome,
        version=version,
        splits_version=splits_version_used,
        threshold=threshold,
        best_params=best_params,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_n=train_used.height,
        val_n=val_df.height,
        test_n=test_df.height,
        artifact_dir=artifact_dir,
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
    """Dispatch on candidate.tuning. Returns (pipeline, best_params, tuning_results)."""
    if candidate.tuning == "cv":
        return model.tune_cv(train_df, cv_folds=candidate.cv_folds)
    if candidate.tuning == "holdout":
        return model.tune(train_df, val_df)
    if candidate.tuning == "none":
        params = candidate.fixed_params or {}
        pipeline = model.train(train_df, params=params)
        return pipeline, dict(params), None
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
