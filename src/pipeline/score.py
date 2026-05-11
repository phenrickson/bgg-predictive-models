"""Snapshot-aware scoring orchestrator.

Scores the snapshot universe with a candidate's per-split pipeline and
writes ``score.parquet`` under each (candidate, split). Downstream
candidates read these files to construct their training features.

CLI::

    uv run python -m src.pipeline.score \\
        --model complexity --candidate ard-complexity \\
        --snapshot-version 1 --splits standard,yoy_2018 \\
        [--candidate-version N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

from src.models.snapshot_storage import DEFAULT_BASE_DIR, SnapshotStorage
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


_PRED_COL = {
    "complexity": "predicted_complexity",
    "rating": "predicted_rating",
    "users_rated": "predicted_users_rated",
    "geek_rating": "predicted_geek_rating",
    "hurdle": "predicted_hurdle",
}


def score(
    snapshot_version: int,
    model_type: str,
    candidate: str,
    candidate_version: Optional[int] = None,
    splits: Optional[List[str]] = None,
    upstream: Optional[Dict[str, str]] = None,
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
) -> int:
    """Score the snapshot universe for one candidate, on each split's
    trained pipeline. Writes ``score.parquet`` per result dir.

    Returns the candidate version actually scored.
    """
    upstream = upstream or {}
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)
    universe = storage.load_universe()
    if universe is None:
        raise FileNotFoundError(f"No snapshot v{snapshot_version}")

    if candidate_version is None:
        versions = storage.list_candidate_versions(model_type, candidate)
        if not versions:
            raise FileNotFoundError(
                f"No versions for {model_type}/{candidate} in v{snapshot_version}"
            )
        candidate_version = versions[-1]

    if splits is None:
        # Score every split that has a result for this candidate version
        splits = []
        cand_dir = storage.experiment_dir(model_type, candidate, candidate_version) / "results"
        if cand_dir.exists():
            splits = sorted(p.name for p in cand_dir.iterdir() if p.is_dir())

    pred_col = _PRED_COL.get(model_type, "prediction")

    for split_name in splits:
        logger.info(f"Scoring {model_type}/{candidate}/v{candidate_version} on {split_name}")
        result = storage.load_result(model_type, candidate, candidate_version, split_name)
        if result is None:
            raise FileNotFoundError(
                f"No result for {model_type}/{candidate}/v{candidate_version}/{split_name}"
            )
        pipeline = result["pipeline"]

        # Join upstream score columns onto the universe (so the pipeline
        # can compute features that depend on, e.g. predicted_complexity)
        scoring_universe = universe
        for upstream_type, upstream_candidate in upstream.items():
            versions = storage.list_candidate_versions(upstream_type, upstream_candidate)
            if not versions:
                raise FileNotFoundError(
                    f"Upstream {upstream_type}/{upstream_candidate} not found"
                )
            uv = versions[-1]
            us = storage.load_score_predictions(upstream_type, upstream_candidate, uv, split_name)
            if us is None:
                raise FileNotFoundError(
                    f"Upstream {upstream_type}/{upstream_candidate}/v{uv} has no "
                    f"score.parquet for split {split_name!r}"
                )
            join_cols = [c for c in us.columns if c == "game_id" or c not in scoring_universe.columns]
            scoring_universe = scoring_universe.join(us.select(join_cols), on="game_id", how="left")

        X = scoring_universe.to_pandas()
        preds = pipeline.predict(X)
        score_df = scoring_universe.select(["game_id"]).clone().with_columns(
            pl.Series(pred_col, preds)
        )

        # For upstream models (complexity/rating/users_rated), replace the
        # in-sample predictions on train rows with K-fold OOF predictions
        # so downstream training reads honest features.
        if model_type in {"complexity", "rating", "users_rated"}:
            split = storage.load_split(split_name)
            if split is None:
                raise FileNotFoundError(f"Split {split_name!r} no longer present")
            score_df = _replace_train_rows_with_oof(
                score_df=score_df,
                pred_col=pred_col,
                split_name=split_name,
                pipeline=pipeline,
                train_df=split["train"],
                model_type=model_type,
                candidate=candidate,
                candidate_version=candidate_version,
                storage=storage,
                upstream=upstream,
            )

        # Save by re-saving the full result with the new score predictions
        # (preserving existing tune/test predictions and metadata).
        storage.save_result(
            model_type=model_type,
            candidate=candidate,
            version=candidate_version,
            split_name=split_name,
            pipeline=result["pipeline"],
            metrics=result["metrics"],
            parameters=result["parameters"],
            tune_predictions=result.get("tune_predictions"),
            test_predictions=result.get("test_predictions"),
            score_predictions=score_df,
        )
        logger.info(f"Wrote score.parquet for {model_type}/{candidate}/v{candidate_version}/{split_name}")

    return candidate_version


def _replace_train_rows_with_oof(
    *,
    score_df: pl.DataFrame,
    pred_col: str,
    split_name: str,
    pipeline,
    train_df: pl.DataFrame,
    model_type: str,
    candidate: str,
    candidate_version: int,
    storage: SnapshotStorage,
    upstream: Dict[str, str],
) -> pl.DataFrame:
    """Run K-fold OOF on the train fold and substitute into score_df."""
    from sklearn.base import clone
    from src.models import oof
    from src.models.outcomes.train import get_model_class
    from src.models.outcomes.data import select_X_y

    cfg = storage.load_candidate_config(model_type, candidate, candidate_version) or {}
    oof_folds = int(cfg.get("oof_folds", 5))

    # Clone the trained pipeline → an unfitted pipeline with the same structure
    fresh_pipeline = clone(pipeline)

    # Join upstream onto the train frame so X has the same columns as it
    # did at train time
    enriched_train = train_df
    for upstream_type, upstream_candidate in upstream.items():
        versions = storage.list_candidate_versions(upstream_type, upstream_candidate)
        if not versions:
            continue
        uv = versions[-1]
        us = storage.load_score_predictions(upstream_type, upstream_candidate, uv, split_name)
        if us is None:
            continue
        join_cols = [c for c in us.columns if c == "game_id" or c not in enriched_train.columns]
        enriched_train = enriched_train.join(us.select(join_cols), on="game_id", how="left")

    target_column = get_model_class(model_type)().target_column
    X, y = select_X_y(enriched_train, target_column)

    oof_preds = oof.kfold_oof_predict(
        pipeline=fresh_pipeline,
        X=X,
        y=y,
        k=oof_folds,
        seed=42,
    )

    train_id_to_oof = dict(zip(enriched_train["game_id"].to_list(), oof_preds))
    out_pd = score_df.to_pandas()
    train_ids = set(enriched_train["game_id"].to_list())
    mask = out_pd["game_id"].isin(train_ids)
    out_pd.loc[mask, pred_col] = out_pd.loc[mask, "game_id"].map(train_id_to_oof)
    return pl.from_pandas(out_pd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--snapshot-version", type=int, required=True)
    parser.add_argument("--candidate-version", type=int, default=None)
    parser.add_argument("--splits", type=str, default=None,
                        help="Comma-separated split names (default: every split with a result)")
    parser.add_argument("--upstream", type=str, default=None)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    setup_logging()

    # Default upstream from the candidate's config; --upstream overrides.
    from src.models.candidate_config import find_candidate
    candidate_config = find_candidate(model_type=args.model, candidate=args.candidate)
    upstream: Dict[str, str] = dict(candidate_config.get("upstream") or {})
    if args.upstream:
        for pair in args.upstream.split(","):
            k, v = pair.split("=", 1)
            upstream[k.strip()] = v.strip()

    splits = (
        [s.strip() for s in args.splits.split(",") if s.strip()] if args.splits else None
    )

    version = score(
        snapshot_version=args.snapshot_version,
        model_type=args.model,
        candidate=args.candidate,
        candidate_version=args.candidate_version,
        splits=splits,
        upstream=upstream,
        base_dir=args.base_dir,
    )
    print(f"scored: {args.model}/{args.candidate}/v{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
