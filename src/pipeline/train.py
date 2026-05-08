"""Snapshot-aware orchestration for outcome-model training.

Loads frames from ``SnapshotStorage``, joins upstream score columns,
calls ``train_one`` per split, writes results back. The Makefile's
``make hurdle``/``make complexity``/etc still invoke
``uv run -m src.pipeline.train`` — only the CLI args change.

CLI::

    uv run python -m src.pipeline.train \\
        --model rating --candidate ard-ridge-rating \\
        --snapshot-version 1 --splits standard,yoy_2018 \\
        [--upstream complexity=ard-complexity]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

from src.models.candidate_config import find_candidate
from src.models.outcomes.train import train_one
from src.models.snapshot_storage import DEFAULT_BASE_DIR, SnapshotStorage
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def train(
    snapshot_version: int,
    model_type: str,
    candidate: str,
    candidate_config: Dict[str, Any],
    splits: List[str],
    upstream: Optional[Dict[str, str]] = None,
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
) -> int:
    """Run training for one candidate over one or more splits.

    Returns the candidate version number assigned to this run.
    """
    upstream = upstream or {}
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)
    if storage.load_universe() is None:
        raise FileNotFoundError(f"No snapshot v{snapshot_version}")

    candidate_version = storage.next_candidate_version(model_type, candidate)

    storage.save_candidate_config(model_type, candidate, candidate_version, candidate_config)
    storage.save_candidate_registration(
        model_type, candidate, candidate_version,
        {
            "snapshot_version": snapshot_version,
            "model_type": model_type,
            "candidate": candidate,
            "version": candidate_version,
            "created_at": datetime.now().isoformat(),
            "upstream_experiments": upstream,
            "splits": splits,
        },
    )

    for split_name in splits:
        logger.info(f"Training {model_type}/{candidate}/v{candidate_version} on {split_name}")
        split = storage.load_split(split_name)
        if split is None:
            raise FileNotFoundError(f"Split {split_name!r} not found in v{snapshot_version}")

        train_df, tune_df, test_df = split["train"], split["tune"], split["test"]
        train_df, tune_df, test_df = _join_upstream(
            storage, upstream, split_name, train_df, tune_df, test_df,
        )

        artifacts = train_one(
            model_type=model_type,
            candidate_config=candidate_config,
            train_df=train_df,
            tune_df=tune_df,
            test_df=test_df,
        )

        storage.save_result(
            model_type=model_type,
            candidate=candidate,
            version=candidate_version,
            split_name=split_name,
            pipeline=artifacts["pipeline"],
            metrics=artifacts["metrics"],
            parameters=artifacts["parameters"],
            tune_predictions=artifacts.get("tune_predictions"),
            test_predictions=artifacts.get("test_predictions"),
        )
        logger.info(f"Wrote result {model_type}/{candidate}/v{candidate_version}/{split_name}")

    return candidate_version


def _join_upstream(
    storage: SnapshotStorage,
    upstream: Dict[str, str],
    split_name: str,
    train_df: pl.DataFrame,
    tune_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Join upstream score.parquet onto each frame.

    For each upstream {model_type: candidate}, look up the latest version
    that has a score.parquet for ``split_name`` and left-join on game_id.
    """
    for upstream_type, upstream_candidate in upstream.items():
        versions = storage.list_candidate_versions(upstream_type, upstream_candidate)
        if not versions:
            raise FileNotFoundError(
                f"Upstream {upstream_type}/{upstream_candidate} has no versions in this snapshot"
            )
        v = versions[-1]
        score = storage.load_score_predictions(upstream_type, upstream_candidate, v, split_name)
        if score is None:
            raise FileNotFoundError(
                f"Upstream {upstream_type}/{upstream_candidate}/v{v} has no "
                f"score.parquet for split {split_name!r}"
            )
        # Drop columns already present (other than game_id) to avoid join collisions
        join_cols = [c for c in score.columns if c == "game_id" or c not in train_df.columns]
        score = score.select(join_cols)
        train_df = train_df.join(score, on="game_id", how="left")
        tune_df = tune_df.join(score, on="game_id", how="left")
        test_df = test_df.join(score, on="game_id", how="left")
    return train_df, tune_df, test_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--snapshot-version", type=int, required=True)
    parser.add_argument("--splits", type=str, default="standard",
                        help="Comma-separated split names")
    parser.add_argument("--upstream", type=str, default=None,
                        help="Comma-separated overrides like 'complexity=ard-complexity'")
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    setup_logging()

    candidate_config = find_candidate(model_type=args.model, candidate=args.candidate)
    upstream = dict(candidate_config.get("upstream") or {})
    if args.upstream:
        for pair in args.upstream.split(","):
            k, v = pair.split("=", 1)
            upstream[k.strip()] = v.strip()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    version = train(
        snapshot_version=args.snapshot_version,
        model_type=args.model,
        candidate=args.candidate,
        candidate_config=candidate_config,
        splits=splits,
        upstream=upstream,
        base_dir=args.base_dir,
    )
    print(f"experiment: {args.model}/{args.candidate}/v{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
