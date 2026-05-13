"""Per-split finalize: refit a candidate on train+tune+test of one split.

For a given split, refits the candidate pipeline on the universe filtered to
``year_published <= split.test_through`` (i.e. every row the split treated as
in-distribution training-window data). Writes ``finalized.pkl`` alongside the
per-split ``pipeline.pkl`` under ``results/{split_name}/``.

Upstream cascade is resolved from the candidate's ``registration.json``. For
each upstream, this loads the upstream's *per-split finalized* pipeline (same
split as we're finalizing on) and joins its predictions onto the universe
before refitting. Finalize is run in cascade order (complexity → rating &
users_rated → geek_rating) so upstream finalized pipelines exist.

CLI::

    uv run python -m src.pipeline.finalize \\
        --model complexity --candidate ard-complexity \\
        --snapshot-version 1 --split yoy_2021
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import polars as pl
from sklearn.base import clone

from src.models.outcomes.data import select_X_y
from src.models.outcomes.train import get_model_class
from src.models.snapshot_storage import DEFAULT_BASE_DIR, SnapshotStorage
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _join_upstream_predictions(
    df: pl.DataFrame,
    upstream: Dict[str, str],
    split_name: str,
    storage: SnapshotStorage,
) -> pl.DataFrame:
    """Use each upstream's per-split finalized pipeline to predict on df.

    Column name convention mirrors score.py:
      complexity   → predicted_complexity
      rating       → predicted_rating
      users_rated  → predicted_users_rated  (log-scale raw output)
    """
    # Process upstreams in cascade order so an upstream whose own predict()
    # depends on another upstream's column sees it on the frame first.
    # Concretely: rating's finalized pipeline reads predicted_users_rated, so
    # users_rated must be joined before rating.
    _CASCADE_ORDER = ["complexity", "users_rated", "rating"]
    ordered = sorted(
        upstream.items(),
        key=lambda kv: (
            _CASCADE_ORDER.index(kv[0]) if kv[0] in _CASCADE_ORDER else len(_CASCADE_ORDER),
            kv[0],
        ),
    )
    for upstream_type, upstream_candidate in ordered:
        versions = storage.list_candidate_versions(upstream_type, upstream_candidate)
        if not versions:
            raise FileNotFoundError(
                f"Upstream {upstream_type}/{upstream_candidate} has no versions"
            )
        v = versions[-1]
        upstream_pipeline = storage.load_finalized_pipeline(
            upstream_type, upstream_candidate, v, split_name
        )
        if upstream_pipeline is None:
            raise FileNotFoundError(
                f"No finalized.pkl for upstream {upstream_type}/{upstream_candidate}/v{v} "
                f"on split {split_name!r} — finalize upstream first"
            )

        col_map = {
            "complexity": "predicted_complexity",
            "rating": "predicted_rating",
            "users_rated": "predicted_users_rated",
        }
        pred_col = col_map.get(upstream_type, f"predicted_{upstream_type}")

        df_pd = df.to_pandas()
        preds = upstream_pipeline.predict(df_pd)
        df = df.with_columns(pl.Series(pred_col, preds))
        logger.info(
            f"Joined {pred_col} from finalized {upstream_type}/{upstream_candidate}/v{v} "
            f"({split_name})"
        )

    return df


def finalize(
    snapshot_version: int,
    model_type: str,
    candidate: str,
    split_name: str,
    candidate_version: Optional[int] = None,
    upstream: Optional[Dict[str, str]] = None,
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
) -> Path:
    """Refit ``candidate`` on universe ≤ ``split.test_through`` and save finalized.pkl.

    The refit cutoff comes from the split's own metadata — every YoY split
    finalizes through its own ``test_through``.
    """
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)
    universe = storage.load_universe()
    if universe is None:
        raise FileNotFoundError(f"No snapshot v{snapshot_version}")

    split = storage.load_split(split_name)
    if split is None:
        raise FileNotFoundError(
            f"No split {split_name!r} under snapshot v{snapshot_version}"
        )
    test_through = int(split["metadata"]["test_through"])

    if candidate_version is None:
        versions = storage.list_candidate_versions(model_type, candidate)
        if not versions:
            raise FileNotFoundError(
                f"No versions for {model_type}/{candidate}"
            )
        candidate_version = versions[-1]

    # Upstream cascade defaults to whatever was recorded at train time.
    if upstream is None:
        reg = storage.load_candidate_registration(model_type, candidate, candidate_version) or {}
        recorded = reg.get("upstream_experiments") or {}
        if recorded:
            upstream = dict(recorded)
            logger.info(f"Using upstream from registration.json: {upstream}")

    # Clone the per-split pipeline as the refit template.
    base_result = storage.load_result(model_type, candidate, candidate_version, split_name)
    if base_result is None:
        raise FileNotFoundError(
            f"No result for {model_type}/{candidate}/v{candidate_version} on "
            f"split {split_name!r} — run pipeline.train against this split first"
        )
    template_pipeline = base_result["pipeline"]

    df = universe.filter(pl.col("year_published") <= test_through)
    logger.info(
        f"Finalizing {model_type}/{candidate}/v{candidate_version} on {split_name}: "
        f"refit through {test_through} ({df.height} rows)"
    )

    if upstream:
        df = _join_upstream_predictions(df, upstream, split_name, storage)

    # Instantiate the model the same way train.py does so prepare_features
    # applies the candidate's filters (min_ratings/min_weights) and any
    # outcome-specific transforms (geek_rating's 0→prior substitution, etc).
    candidate_config = (
        storage.load_candidate_config(model_type, candidate, candidate_version) or {}
    )
    model_class = get_model_class(model_type)
    model_kwargs: Dict[str, Any] = {}
    for k in ("min_ratings", "min_weights", "mode", "include_predictions"):
        if k in candidate_config:
            model_kwargs[k] = candidate_config[k]
    model = model_class(**model_kwargs)

    X, y = select_X_y(df, model.target_column)
    prep_args = SimpleNamespace(
        use_embeddings=bool(candidate_config.get("use_embeddings", False)),
        sub_model_experiments=candidate_config.get("sub_model_experiments", {}),
        mode=candidate_config.get("mode"),
        include_predictions=candidate_config.get("include_predictions", True),
    )
    X, y = model.prepare_features(X, y, "train", prep_args)
    logger.info(
        f"  after prepare_features: {len(X)} rows "
        f"(dropped {df.height - len(X)} via filters)"
    )

    finalized_pipeline = clone(template_pipeline)
    finalized_pipeline.fit(X, y)

    path = storage.save_finalized_pipeline(
        model_type, candidate, candidate_version, split_name, finalized_pipeline
    )

    reg = storage.load_candidate_registration(model_type, candidate, candidate_version) or {}
    finalize_log: Dict[str, Any] = reg.get("finalize", {}) or {}
    finalize_log[split_name] = {
        "finalize_through": test_through,
        "finalized_at": datetime.now().isoformat(),
        "upstream": upstream or {},
    }
    reg["finalize"] = finalize_log
    storage.save_candidate_registration(model_type, candidate, candidate_version, reg)

    logger.info(f"Finalized {model_type}/{candidate}/v{candidate_version}/{split_name} → {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--snapshot-version", type=int, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--candidate-version", type=int, default=None)
    parser.add_argument("--upstream", type=str, default=None,
                        help="Comma-separated upstream like 'complexity=ard-complexity'")
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    setup_logging()

    upstream: Dict[str, str] = {}
    if args.upstream:
        for pair in args.upstream.split(","):
            k, v = pair.split("=", 1)
            upstream[k.strip()] = v.strip()

    finalize(
        snapshot_version=args.snapshot_version,
        model_type=args.model,
        candidate=args.candidate,
        split_name=args.split,
        candidate_version=args.candidate_version,
        upstream=upstream or None,
        base_dir=args.base_dir,
    )
    print(f"finalized: {args.model}/{args.candidate}/{args.split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
