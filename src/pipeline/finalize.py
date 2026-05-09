"""Snapshot-aware finalize orchestrator.

Refits a candidate's pipeline on the full snapshot universe (filtered
through ``finalize_through`` if provided) and writes ``finalized.pkl``
at the candidate level. Operational scoring downstream uses this
artifact.

CLI::

    uv run python -m src.pipeline.finalize \\
        --model complexity --candidate ard-complexity \\
        --snapshot-version 1 [--candidate-version N] [--finalize-through 2024]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import polars as pl
from sklearn.base import clone

from src.models.outcomes.data import select_X_y
from src.models.outcomes.train import get_model_class
from src.models.snapshot_storage import DEFAULT_BASE_DIR, SnapshotStorage
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def finalize(
    snapshot_version: int,
    model_type: str,
    candidate: str,
    candidate_version: Optional[int] = None,
    finalize_through: Optional[int] = None,
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
) -> Path:
    """Refit candidate on snapshot universe (≤ finalize_through) and save finalized.pkl."""
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)
    universe = storage.load_universe()
    if universe is None:
        raise FileNotFoundError(f"No snapshot v{snapshot_version}")

    if candidate_version is None:
        versions = storage.list_candidate_versions(model_type, candidate)
        if not versions:
            raise FileNotFoundError(
                f"No versions for {model_type}/{candidate}"
            )
        candidate_version = versions[-1]

    # Use any existing per-split pipeline to produce a clone for refitting
    cand_dir = storage.experiment_dir(model_type, candidate, candidate_version) / "results"
    if not cand_dir.exists() or not any(cand_dir.iterdir()):
        raise FileNotFoundError(
            f"No results for {model_type}/{candidate}/v{candidate_version}; train first"
        )
    any_split = next(cand_dir.iterdir()).name
    base_result = storage.load_result(model_type, candidate, candidate_version, any_split)
    if base_result is None:
        raise FileNotFoundError(f"Failed to load any result for {model_type}/{candidate}")
    template_pipeline = base_result["pipeline"]

    df = universe
    if finalize_through is not None:
        df = df.filter(pl.col("year_published") <= int(finalize_through))

    target_column = get_model_class(model_type)().target_column
    X, y = select_X_y(df, target_column)

    finalized_pipeline = clone(template_pipeline)
    finalized_pipeline.fit(X, y)

    storage.save_finalized_pipeline(model_type, candidate, candidate_version, finalized_pipeline)

    reg = storage.load_candidate_registration(model_type, candidate, candidate_version) or {}
    reg["finalize_through"] = int(finalize_through) if finalize_through is not None else None
    reg["finalized_at"] = datetime.now().isoformat()
    storage.save_candidate_registration(model_type, candidate, candidate_version, reg)

    finalized_path = (
        storage.experiment_dir(model_type, candidate, candidate_version) / "finalized.pkl"
    )
    logger.info(f"Finalized {model_type}/{candidate}/v{candidate_version} → {finalized_path}")
    return finalized_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--snapshot-version", type=int, required=True)
    parser.add_argument("--candidate-version", type=int, default=None)
    parser.add_argument("--finalize-through", type=int, default=None)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    setup_logging()
    finalize(
        snapshot_version=args.snapshot_version,
        model_type=args.model,
        candidate=args.candidate,
        candidate_version=args.candidate_version,
        finalize_through=args.finalize_through,
        base_dir=args.base_dir,
    )
    print(f"finalized: {args.model}/{args.candidate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
