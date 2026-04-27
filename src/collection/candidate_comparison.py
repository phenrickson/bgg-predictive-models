"""Pure comparison utilities over persisted candidate runs.

Comparison reads only what's already on disk — it never trains. Use this
after :func:`src.collection.candidate_runner.train_candidate` has been
called for two or more candidates.

The headline function :func:`compare_runs` returns a tall metrics frame
(one row per candidate × split × metric) with the candidate spec,
splits_version, and version attached so you can filter/group however you
like.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import polars as pl

from src.collection.collection_artifact_storage import CollectionArtifactStorage

logger = logging.getLogger(__name__)


def load_candidate_runs(
    storage: CollectionArtifactStorage,
    outcome: str,
    candidate_names: Optional[List[str]] = None,
    versions: str = "latest",
) -> List[Dict[str, Any]]:
    """Load registration dicts (no pipelines) for runs under ``outcome``.

    Args:
        storage: Configured artifact storage.
        outcome: Outcome name, e.g. ``"own"``.
        candidate_names: Restrict to these candidates. ``None`` = every
            candidate that has at least one run.
        versions: ``"latest"`` returns one registration per candidate
            (the highest version); ``"all"`` returns every version of
            every selected candidate.

    Returns:
        List of registration dicts, sorted by ``(candidate, version)``.
    """
    if versions not in ("latest", "all"):
        raise ValueError(f"versions must be 'latest' or 'all', got {versions!r}")

    selected = (
        candidate_names
        if candidate_names is not None
        else storage.list_candidates(outcome)
    )

    runs: List[Dict[str, Any]] = []
    for candidate in selected:
        if versions == "latest":
            v = storage.latest_candidate_version(outcome, candidate)
            if v is None:
                logger.warning(
                    f"No runs found for candidate {candidate!r} on outcome {outcome!r}"
                )
                continue
            reg = storage.load_candidate_registration(outcome, candidate, version=v)
            if reg is not None:
                runs.append(reg)
        else:
            runs.extend(storage.list_candidate_runs(outcome, candidate=candidate))

    return sorted(
        runs, key=lambda r: (r.get("candidate", ""), r.get("version", 0))
    )


def compare_runs(runs: List[Dict[str, Any]]) -> pl.DataFrame:
    """Build a tall metrics frame from a list of run registrations.

    Args:
        runs: Output of :func:`load_candidate_runs`.

    Returns:
        Polars DataFrame with one row per ``(candidate, version, split,
        metric)``. Columns: ``candidate``, ``version``, ``splits_version``,
        ``task``, ``split``, ``metric``, ``value``, ``threshold``,
        ``best_params`` (str), ``n_train_used``, ``n_val``, ``n_test``,
        ``git_sha``, ``trained_at``.

        Empty frame (correct schema) if ``runs`` is empty.

    Raises:
        ValueError: If runs reference different ``splits_version`` values
            for the same outcome (silent comparison across different
            val/test would be misleading).
    """
    rows: List[Dict[str, Any]] = []
    splits_versions_seen: set = set()

    for r in runs:
        splits_versions_seen.add(r.get("splits_version"))
        common = {
            "candidate": r.get("candidate"),
            "version": r.get("version"),
            "splits_version": r.get("splits_version"),
            "task": r.get("task"),
            "threshold": r.get("threshold"),
            "best_params": str(r.get("best_params")),
            "n_train_used": r.get("n_train_used"),
            "n_val": r.get("n_val"),
            "n_test": r.get("n_test"),
            "git_sha": r.get("git_sha"),
            "trained_at": r.get("trained_at"),
        }
        for split_label, metric_dict_key in (("val", "val_metrics"), ("test", "metrics")):
            metrics = r.get(metric_dict_key) or {}
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                rows.append(
                    {
                        **common,
                        "split": split_label,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )

    if len(splits_versions_seen) > 1:
        raise ValueError(
            f"Cannot compare runs across different splits_versions: "
            f"{sorted(v for v in splits_versions_seen if v is not None)}. "
            f"Re-run candidates against the same canonical splits."
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "candidate": pl.Utf8,
                "version": pl.Int64,
                "splits_version": pl.Int64,
                "task": pl.Utf8,
                "threshold": pl.Float64,
                "best_params": pl.Utf8,
                "n_train_used": pl.Int64,
                "n_val": pl.Int64,
                "n_test": pl.Int64,
                "git_sha": pl.Utf8,
                "trained_at": pl.Utf8,
                "split": pl.Utf8,
                "metric": pl.Utf8,
                "value": pl.Float64,
            }
        )
    return pl.DataFrame(rows)
