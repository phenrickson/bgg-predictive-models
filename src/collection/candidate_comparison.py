"""Pure comparison utilities over persisted candidate runs.

Comparison reads only what's already on disk — it never trains. Use this
after :func:`src.collection.candidates.train_candidate` has been called
for two or more candidates.

The headline function :func:`compare_runs` returns a tall metrics frame
(one row per candidate × split × metric) with the candidate spec,
splits_version, and version attached so you can filter/group however you
like.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

import polars as pl

from src.collection.collection_artifact_storage import CollectionArtifactStorage

if TYPE_CHECKING:
    from src.collection.candidates import CandidateRunResult

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


def summarize_runs(runs: List[Dict[str, Any]]) -> pl.DataFrame:
    """Wide, metric-focused view of the selected run per candidate.

    Two rows per candidate (val + test), one column per metric, no metadata
    clutter. Use this for at-a-glance candidate comparison; use
    :func:`compare_runs` when you need the long-form view (filtering by
    metric, joining metadata, etc.).

    Args:
        runs: Output of :func:`load_candidate_runs`.

    Returns:
        Polars DataFrame with one row per ``(candidate, split)``, sorted by
        candidate then split (val before test). Empty frame with just
        ``candidate`` and ``split`` columns if ``runs`` is empty.
    """
    tall = compare_runs(runs)
    if tall.height == 0:
        return pl.DataFrame(schema={"candidate": pl.Utf8, "split": pl.Utf8})

    wide = tall.pivot(values="value", index=["candidate", "split"], on="metric")
    return wide.sort(
        ["candidate", "split"],
        descending=[False, True],  # val before test alphabetically
    )


def compare_top_games(
    results: Sequence["CandidateRunResult"],
    df: pl.DataFrame,
    n: int = 25,
    exclude_game_ids: Optional[Iterable[int]] = None,
    include_columns: Sequence[str] = ("game_id", "name", "year_published"),
) -> pl.DataFrame:
    """Per-model top-``n`` games, unioned across candidates.

    Each candidate scores ``df`` and contributes its own top-``n`` games to
    the output set. Games appear in the result if they made *any* model's
    top-``n``. Each row carries every model's score for that game (so
    disagreements are visible), plus a ``picked_by`` column listing which
    models had it in their top.

    Args:
        results: In-memory ``CandidateRunResult``s (from ``train_candidates``).
        df: Rows to score. Must have the columns each model was trained on.
        n: Top-N per candidate. Output row count is the union — between
            ``n`` (perfect agreement) and ``n * len(results)`` (no overlap).
        exclude_game_ids: ``game_id``s to drop before ranking.
        include_columns: Columns from ``df`` to surface alongside the
            scores. Missing columns are silently skipped.

    Returns:
        Polars DataFrame with the kept ``include_columns``, one
        ``score_<candidate>`` column per result, and a ``picked_by``
        column. Sorted by mean score descending so games every model
        liked float to the top.
    """
    if not results:
        raise ValueError("compare_top_games requires at least one result")
    if "game_id" not in df.columns:
        raise ValueError("compare_top_games requires a 'game_id' column on df")

    names = [r.candidate.name for r in results]
    score_cols = [f"score_{nm}" for nm in names]

    if df.height == 0:
        empty_cols = [c for c in include_columns if c in df.columns] + score_cols
        empty = df.head(0).select([
            pl.lit(None).cast(pl.Float64).alias(c) for c in empty_cols
        ])
        return empty.with_columns(pl.lit("").alias("picked_by"))

    # Score every row with every model. (Cheaper than re-scoring per model
    # and aligning afterwards; the predict cost is the same either way and
    # we end up with a single wide frame.)
    scored = df
    for r, col in zip(results, score_cols):
        pipeline = r.model._require_fitted()
        X = df.drop("label") if "label" in df.columns else df
        X = X.to_pandas()
        if r.outcome.task == "classification":
            score = pipeline.predict_proba(X)[:, 1]
        else:
            score = pipeline.predict(X)
        scored = scored.with_columns(pl.Series(col, score))

    if exclude_game_ids is not None:
        excluded = list(set(exclude_game_ids))
        if excluded:
            scored = scored.filter(~pl.col("game_id").is_in(excluded))

    # Each candidate's top-N by its own score. Track which models picked
    # each game so we can build picked_by after unioning.
    picks: Dict[int, List[str]] = {}
    for r, col in zip(results, score_cols):
        top = scored.sort(col, descending=True).head(n)
        for gid in top["game_id"].to_list():
            picks.setdefault(gid, []).append(r.candidate.name)

    if not picks:
        kept = [c for c in include_columns if c in scored.columns]
        return scored.head(0).select(kept + score_cols).with_columns(
            pl.lit("").alias("picked_by")
        )

    union = scored.filter(pl.col("game_id").is_in(list(picks.keys())))

    # Attach picked_by ("logistic, lgbm") in the candidate order callers
    # passed in.
    picked_by_map = {gid: ", ".join(ms) for gid, ms in picks.items()}
    union = union.with_columns(
        pl.col("game_id").map_elements(
            picked_by_map.get, return_dtype=pl.Utf8
        ).alias("picked_by")
    )

    kept = [c for c in include_columns if c in union.columns]
    out = union.select(kept + score_cols + ["picked_by"])
    return out.with_columns(
        pl.mean_horizontal(*score_cols).alias("_mean_score")
    ).sort("_mean_score", descending=True).drop("_mean_score")
