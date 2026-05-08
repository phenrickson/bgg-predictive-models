"""Derive a named split from a snapshot.

Run::

    uv run python -m src.models.build_split \\
        --snapshot-version N --split-name standard \\
        [--train-through 2022 --tune-start 2023 --tune-through 2023 \\
         --test-start 2024 --test-through 2024]

For YoY mode see :func:`build_yoy_splits`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

import polars as pl

from src.models.snapshot_storage import DEFAULT_BASE_DIR, SnapshotStorage
from src.models.splitting import time_based_split
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def build_split(
    snapshot_version: int,
    split_name: str,
    train_through: int,
    tune_start: int,
    tune_through: int,
    test_start: int,
    test_through: int,
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
    time_col: str = "year_published",
) -> dict:
    """Build a single named split from a snapshot."""
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)
    universe = storage.load_universe()
    if universe is None:
        raise FileNotFoundError(
            f"No snapshot v{snapshot_version} at {storage.snapshot_dir}/universe.parquet"
        )

    if not (tune_start <= tune_through < test_start <= test_through):
        raise ValueError(
            f"Invalid year ranges: tune {tune_start}..{tune_through} "
            f"must precede test {test_start}..{test_through}"
        )
    if tune_start <= train_through:
        raise ValueError(
            f"tune_start ({tune_start}) must be greater than train_through ({train_through})"
        )

    validation_window = tune_through - tune_start + 1
    test_window = test_through - test_start + 1

    train_df, tune_df, test_df = time_based_split(
        df=universe,
        train_through=train_through,
        prediction_window=validation_window,
        test_window=test_window,
        time_col=time_col,
        return_dict=False,
    )

    metadata = {
        "split_name": split_name,
        "snapshot_version": snapshot_version,
        "train_through": train_through,
        "tune_start": tune_start,
        "tune_through": tune_through,
        "test_start": test_start,
        "test_through": test_through,
        "time_col": time_col,
        "n_train": train_df.height,
        "n_tune": tune_df.height,
        "n_test": test_df.height,
        "created_at": datetime.now().isoformat(),
    }

    paths = storage.save_split(split_name, train_df, tune_df, test_df, metadata)
    logger.info(
        f"Built split {split_name} on v{snapshot_version}: "
        f"train={train_df.height}, tune={tune_df.height}, test={test_df.height}"
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--snapshot-version", type=int, required=True)
    parser.add_argument("--split-name", type=str, default="standard")
    parser.add_argument("--train-through", type=int, default=None)
    parser.add_argument("--tune-start", type=int, default=None)
    parser.add_argument("--tune-through", type=int, default=None)
    parser.add_argument("--test-start", type=int, default=None)
    parser.add_argument("--test-through", type=int, default=None)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    setup_logging()

    # Defaults from config.yaml years.training
    if any(v is None for v in [args.train_through, args.tune_start, args.tune_through,
                               args.test_start, args.test_through]):
        config = load_config()
        ycfg = config.years.training
        args.train_through = args.train_through or ycfg.train_through
        args.tune_start = args.tune_start or ycfg.tune_start
        args.tune_through = args.tune_through or ycfg.tune_through
        args.test_start = args.test_start or ycfg.test_start
        args.test_through = args.test_through or ycfg.test_through

    build_split(
        snapshot_version=args.snapshot_version,
        split_name=args.split_name,
        train_through=args.train_through,
        tune_start=args.tune_start,
        tune_through=args.tune_through,
        test_start=args.test_start,
        test_through=args.test_through,
        base_dir=args.base_dir,
    )
    print(f"split: v{args.snapshot_version}/{args.split_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
