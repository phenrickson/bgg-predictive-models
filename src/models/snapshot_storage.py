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

    # --- Splits ---

    def _split_dir(self, split_name: str) -> Path:
        return self.snapshot_dir / "splits" / split_name

    def save_split(
        self,
        split_name: str,
        train_df: pl.DataFrame,
        tune_df: pl.DataFrame,
        test_df: pl.DataFrame,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Write the three folds plus split metadata."""
        split_dir = self._split_dir(split_name)
        split_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Any] = {"split_name": split_name}
        for name, df in [("train", train_df), ("tune", tune_df), ("test", test_df)]:
            target = split_dir / f"{name}.parquet"
            df.write_parquet(target)
            paths[name] = str(target)
            logger.info(f"Saved split {split_name}/{name} ({df.height} rows)")

        meta_path = split_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        paths["metadata"] = str(meta_path)
        return paths

    def load_split(self, split_name: str) -> Optional[Dict[str, Any]]:
        split_dir = self._split_dir(split_name)
        if not split_dir.exists():
            return None
        result: Dict[str, Any] = {"split_name": split_name}
        for name in ["train", "tune", "test"]:
            path = split_dir / f"{name}.parquet"
            if not path.exists():
                logger.warning(f"Split {split_name} is missing fold {name!r}")
                return None
            result[name] = pl.read_parquet(path)
        meta_path = split_dir / "metadata.json"
        result["metadata"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return result

    def list_splits(self) -> List[str]:
        splits_root = self.snapshot_dir / "splits"
        if not splits_root.exists():
            return []
        return sorted(p.name for p in splits_root.iterdir() if p.is_dir())
