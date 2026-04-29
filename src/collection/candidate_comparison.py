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


EXCLUDED_METRICS: frozenset[str] = frozenset({"accuracy"})


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
                if metric_name in EXCLUDED_METRICS:
                    continue
                rows.append(
                    {
                        **common,
                        "split": split_label,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )

        oof_overall = (r.get("oof_metrics") or {}).get("overall") or {}
        for metric_name, value in oof_overall.items():
            if not isinstance(value, (int, float)):
                continue
            if metric_name in EXCLUDED_METRICS:
                continue
            rows.append(
                {
                    **common,
                    "split": "oof",
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
    split_order = pl.DataFrame(
        {"split": ["val", "oof", "test"], "__split_order": [0, 1, 2]}
    )
    wide = wide.join(split_order, on="split", how="left").with_columns(
        pl.col("__split_order").fill_null(99)
    )
    return wide.sort(["candidate", "__split_order"]).drop("__split_order")


def compare_top_games(
    results: Sequence["CandidateRunResult"],
    df: pl.DataFrame,
    n: int = 25,
    rank_by: Optional[str] = None,
    exclude_game_ids: Optional[Iterable[int]] = None,
    include_columns: Sequence[str] = ("game_id", "name", "year_published"),
) -> pl.DataFrame:
    """Score ``df`` with each candidate's model and return a wide top-N view.

    One row per game, one ``score_<candidate>`` column per result, plus the
    columns in ``include_columns`` that exist on ``df``. Useful for
    eyeballing where models agree and disagree.

    Args:
        results: In-memory ``CandidateRunResult``s (from ``train_candidates``).
        df: Rows to score. Must have the columns each model was trained on.
        n: Top-N rows to return after sorting.
        rank_by: Candidate name to sort by. ``None`` (default) sorts by the
            mean score across all candidates, so games every model likes
            float to the top.
        exclude_game_ids: ``game_id``s to drop before ranking (e.g. games
            the user already owns).
        include_columns: Columns from ``df`` to surface alongside the
            scores. Missing columns are silently skipped.

    Returns:
        Polars DataFrame with the kept ``include_columns`` plus one
        ``score_<candidate>`` column per result, sorted descending by
        ``rank_by`` (or mean score), top-``n`` rows.
    """
    if not results:
        raise ValueError("compare_top_games requires at least one result")

    names = [r.candidate.name for r in results]
    if rank_by is not None and rank_by not in names:
        raise ValueError(
            f"rank_by={rank_by!r} not in candidate names: {names}"
        )

    if df.height == 0:
        empty_cols = [c for c in include_columns if c in df.columns] + [
            f"score_{n_}" for n_ in names
        ]
        return df.head(0).select([
            pl.lit(None).cast(pl.Float64).alias(c) for c in empty_cols
        ])

    # Score with each model. CollectionModel.top_games uses self.fitted_pipeline
    # internally; we replicate that scoring logic without the per-model sort,
    # because we want all rows aligned for the wide view.
    scored = df
    score_cols: List[str] = []
    for r in results:
        col = f"score_{r.candidate.name}"
        score_cols.append(col)
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
        if excluded and "game_id" in scored.columns:
            scored = scored.filter(~pl.col("game_id").is_in(excluded))

    if rank_by is not None:
        sort_col = f"score_{rank_by}"
    else:
        sort_col = "_mean_score"
        scored = scored.with_columns(
            pl.mean_horizontal(*score_cols).alias(sort_col)
        )

    kept = [c for c in include_columns if c in scored.columns]
    out = scored.select(kept + score_cols + ([sort_col] if sort_col == "_mean_score" else []))
    out = out.sort(sort_col, descending=True).head(n)
    if sort_col == "_mean_score":
        out = out.drop(sort_col)
    return out


def top_games_by_rank(
    results: Sequence["CandidateRunResult"],
    df: pl.DataFrame,
    n: int = 10,
    exclude_game_ids: Optional[Iterable[int]] = None,
    label_column: str = "name",
) -> pl.DataFrame:
    """Side-by-side top-``n`` picks per candidate, aligned by rank.

    One row per rank position (1..n). For each candidate, two columns:
    ``<label_column>_<candidate>`` (the game's display name) and
    ``score_<candidate>`` (the model's score). Reads as "model A's
    top 1 pick was X with score 0.94; model B's was Y with score 0.92."

    Args:
        results: In-memory ``CandidateRunResult``s (from ``train_candidates``).
        df: Rows to score. Must have ``label_column`` plus the columns
            each model was trained on.
        n: Top-N per candidate.
        exclude_game_ids: ``game_id``s to drop before ranking.
        label_column: Column to surface as each game's identifier
            (default ``"name"``).
    """
    if not results:
        raise ValueError("top_games_by_rank requires at least one result")
    if label_column not in df.columns:
        raise ValueError(
            f"label_column {label_column!r} missing from df; "
            f"columns: {df.columns[:10]}"
        )

    excluded = set(exclude_game_ids) if exclude_game_ids is not None else set()

    columns: Dict[str, List[Any]] = {"rank": list(range(1, n + 1))}
    for r in results:
        pipeline = r.model._require_fitted()
        X = df.drop("label") if "label" in df.columns else df
        X = X.to_pandas()
        if r.outcome.task == "classification":
            score = pipeline.predict_proba(X)[:, 1]
        else:
            score = pipeline.predict(X)
        scored = df.with_columns(pl.Series("_score", score))
        if excluded and "game_id" in scored.columns:
            scored = scored.filter(~pl.col("game_id").is_in(list(excluded)))
        top = scored.sort("_score", descending=True).head(n)

        labels = top[label_column].to_list()
        scores = top["_score"].to_list()
        # Pad to ``n`` rows if df was smaller than n after exclusions.
        labels += [None] * (n - len(labels))
        scores += [None] * (n - len(scores))

        columns[f"{label_column}_{r.candidate.name}"] = labels
        columns[f"score_{r.candidate.name}"] = scores

    return pl.DataFrame(columns)
