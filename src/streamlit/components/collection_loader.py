"""Discovery + read helpers for the Collections Streamlit page.

Reads only what's already on disk under ``models/collections/{user}/``;
never trains or fits. All filesystem access goes through
:class:`CollectionArtifactStorage` so the Streamlit page mirrors the
shape used by the CLI workflow (train_candidate / compare / finalize).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from src.collection.candidate_comparison import (
    compare_runs,
    load_candidate_runs,
    summarize_runs,
)
from src.collection.collection_artifact_storage import CollectionArtifactStorage

logger = logging.getLogger(__name__)


COLLECTIONS_ROOT = Path("models/collections")


def list_users(root: Path = COLLECTIONS_ROOT) -> List[str]:
    """Return usernames that have at least one outcome directory on disk."""
    if not root.exists():
        return []
    users: List[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        # A valid user dir has at least one non-reserved subdirectory.
        for sub in child.iterdir():
            if sub.is_dir() and sub.name != "collection":
                users.append(child.name)
                break
    return users


def get_storage(username: str, root: Path = COLLECTIONS_ROOT) -> CollectionArtifactStorage:
    return CollectionArtifactStorage(
        username=username,
        local_root=root,
        environment="local",
    )


def list_outcomes(storage: CollectionArtifactStorage) -> List[str]:
    return storage.list_outcomes()


def list_candidates(storage: CollectionArtifactStorage, outcome: str) -> List[str]:
    return storage.list_candidates(outcome)


def load_runs(
    storage: CollectionArtifactStorage, outcome: str
) -> List[Dict[str, Any]]:
    """Latest registration per candidate for an outcome."""
    return load_candidate_runs(storage, outcome=outcome, versions="latest")


def comparison_frame(runs: List[Dict[str, Any]]) -> pl.DataFrame:
    """Wide one-row-per-(candidate, split) table — the Overview view."""
    return summarize_runs(runs)


def long_metrics_frame(runs: List[Dict[str, Any]]) -> pl.DataFrame:
    """Tall one-row-per-(candidate, split, metric) frame for charting."""
    return compare_runs(runs)


def load_predictions(
    storage: CollectionArtifactStorage,
    outcome: str,
    candidate: str,
    split: str,
    version: Optional[int] = None,
) -> Optional[pl.DataFrame]:
    """Load predictions parquet for a candidate run.

    ``split`` is ``"test"`` or ``"val"``.
    """
    if split not in ("test", "val"):
        raise ValueError(f"split must be 'test' or 'val', got {split!r}")
    if version is None:
        version = storage.latest_candidate_version(outcome, candidate)
        if version is None:
            return None
    rel = Path(outcome) / candidate / f"v{version}" / "predictions" / f"{split}.parquet"
    full = storage.base_dir / rel
    if not full.exists():
        return None
    return pl.read_parquet(full)


def load_feature_importance(
    storage: CollectionArtifactStorage,
    outcome: str,
    candidate: str,
    version: Optional[int] = None,
) -> Optional[pl.DataFrame]:
    if version is None:
        version = storage.latest_candidate_version(outcome, candidate)
        if version is None:
            return None
    rel = Path(outcome) / candidate / f"v{version}" / "feature_importance.parquet"
    full = storage.base_dir / rel
    if not full.exists():
        return None
    return pl.read_parquet(full)


def load_tuning_results(
    storage: CollectionArtifactStorage,
    outcome: str,
    candidate: str,
    version: Optional[int] = None,
) -> Optional[pl.DataFrame]:
    return storage.load_candidate_tuning_results(outcome, candidate, version=version)


def list_finalized_candidates(
    storage: CollectionArtifactStorage, outcome: str
) -> List[Tuple[str, int]]:
    """Return ``(candidate, version)`` pairs that have a ``finalized.pkl``."""
    out: List[Tuple[str, int]] = []
    for candidate in storage.list_candidates(outcome):
        version = storage.latest_candidate_version(outcome, candidate)
        if version is None:
            continue
        finalized_path = (
            storage.base_dir
            / outcome
            / candidate
            / f"v{version}"
            / "finalized.pkl"
        )
        if finalized_path.exists():
            out.append((candidate, version))
    return out


def load_finalized_run(
    storage: CollectionArtifactStorage,
    outcome: str,
    candidate: str,
    version: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the finalized pipeline plus its run registration."""
    pipeline = storage.load_finalized_pipeline(outcome, candidate, version=version)
    if pipeline is None:
        raise ValueError(
            f"No finalized.pkl for {candidate} v{version} on {outcome}"
        )
    registration = (
        storage.load_candidate_registration(outcome, candidate, version=version) or {}
    )
    return pipeline, registration
