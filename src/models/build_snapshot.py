"""Build a versioned data snapshot for use by the snapshot+split training framework.

Run::

    uv run python -m src.models.build_snapshot \\
        [--use-embeddings] [--local-data PATH]

Writes ``models/experiments/_snapshots/v{N}/universe.parquet`` and
``metadata.json``. The version number is auto-assigned to the next available
integer. Once built, a snapshot is immutable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import polars as pl

from src.models.snapshot_storage import DEFAULT_BASE_DIR, SnapshotStorage
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def build_snapshot(
    local_data: Optional[Union[str, Path]] = None,
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
    use_embeddings: bool = False,
    snapshot_version: Optional[int] = None,
) -> int:
    """Build a new snapshot version. Returns the assigned version number."""
    if snapshot_version is None:
        snapshot_version = SnapshotStorage.next_version(base_dir=base_dir)

    if local_data is not None:
        df = pl.read_parquet(local_data)
        logger.info(f"Loaded {df.height} rows from local parquet: {local_data}")
    else:
        # BigQuery path
        from src.data.loader import BGGDataLoader
        config = load_config()
        loader = BGGDataLoader(config.get_data_warehouse_config())
        if use_embeddings:
            df = loader.load_data_with_embeddings(where_clause="")
        else:
            df = loader.load_data(where_clause="")
        logger.info(f"Loaded {df.height} rows from BigQuery")

    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)
    storage.save_universe(df)
    storage.save_metadata({
        "snapshot_version": snapshot_version,
        "created_at": datetime.now().isoformat(),
        "n_rows": df.height,
        "columns": df.columns,
        "use_embeddings": use_embeddings,
        "source": "local" if local_data is not None else "bigquery",
    })

    logger.info(f"Built snapshot v{snapshot_version}")
    return snapshot_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--snapshot-version", type=int, default=None,
                        help="Explicit version (default: next available)")
    parser.add_argument("--use-embeddings", action="store_true", default=False)
    parser.add_argument("--local-data", type=str, default=None,
                        help="Local parquet path (skips BigQuery)")
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    setup_logging()
    version = build_snapshot(
        local_data=args.local_data,
        base_dir=args.base_dir,
        use_embeddings=args.use_embeddings,
        snapshot_version=args.snapshot_version,
    )
    print(f"snapshot_version: {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
