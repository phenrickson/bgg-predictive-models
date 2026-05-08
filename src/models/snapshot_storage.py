"""Storage layer for model snapshots, splits, and experiment results.

Owns the path layout and I/O for ``models/experiments/_snapshots/v{N}/``.
Two experiments under the same ``(snapshot_version, split_name)`` are
guaranteed to have seen identical bytes for train/tune/test.

Path layout::

    {base_dir}/v{N}/
        universe.parquet                            # full feature+outcome+id frame
        metadata.json
        splits/{split_name}/
            train.parquet, tune.parquet, test.parquet
            metadata.json
        experiments/{model_type}/{candidate}/v{M}/
            config.json
            registration.json
            finalized.pkl                           # candidate-level
            results/{split_name}/
                pipeline.pkl
                metrics.json, parameters.json
                feature_importance.csv
                predictions/{tune,test,score}.parquet
            summary.json
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = "models/experiments/_snapshots"


class SnapshotStorage:
    """Handles snapshot/split/experiment artifact storage for the new layout."""

    def __init__(
        self,
        snapshot_version: int,
        base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
    ):
        self.snapshot_version = int(snapshot_version)
        self.base_dir = Path(base_dir)
        self.snapshot_dir: Path = self.base_dir / f"v{self.snapshot_version}"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def latest_version(cls, base_dir: Union[str, Path] = DEFAULT_BASE_DIR) -> Optional[int]:
        """Highest existing snapshot version number, or None if none exist."""
        base = Path(base_dir)
        if not base.exists():
            return None
        versions: List[int] = []
        for child in base.iterdir():
            if not child.is_dir() or not child.name.startswith("v"):
                continue
            try:
                versions.append(int(child.name[1:]))
            except ValueError:
                continue
        return max(versions) if versions else None

    @classmethod
    def next_version(cls, base_dir: Union[str, Path] = DEFAULT_BASE_DIR) -> int:
        """Next available snapshot version number (latest + 1, or 1 if none)."""
        latest = cls.latest_version(base_dir=base_dir)
        return (latest or 0) + 1

    # --- Universe ---

    def save_universe(self, df: pl.DataFrame) -> Path:
        """Write the snapshot's full feature+outcome frame."""
        path = self.snapshot_dir / "universe.parquet"
        df.write_parquet(path)
        logger.info(f"Saved universe ({df.height} rows) to {path}")
        return path

    def load_universe(self) -> Optional[pl.DataFrame]:
        """Load the snapshot's universe, or None if not yet built."""
        path = self.snapshot_dir / "universe.parquet"
        if not path.exists():
            return None
        return pl.read_parquet(path)

    # --- Metadata ---

    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Write the snapshot's metadata.json."""
        path = self.snapshot_dir / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2, default=str))
        return path

    def load_metadata(self) -> Optional[Dict[str, Any]]:
        path = self.snapshot_dir / "metadata.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())
