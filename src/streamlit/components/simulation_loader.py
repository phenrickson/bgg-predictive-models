"""Loader for simulation run results from local disk."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import polars as pl

logger = logging.getLogger(__name__)

SIMULATION_DIR = Path("models/simulation")
OUTCOMES = ["complexity", "rating", "users_rated", "geek_rating"]


def discover_runs(base_dir: Path = SIMULATION_DIR) -> List[Dict[str, Any]]:
    """Scan for simulation runs containing run_metadata.json.

    Returns list of dicts with keys: name, path, timestamp, metadata.
    Sorted by timestamp descending (newest first).
    """
    runs = []
    if not base_dir.exists():
        return runs

    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            continue
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            runs.append({
                "name": metadata.get("run_name", run_dir.name),
                "path": run_dir,
                "timestamp": metadata.get("timestamp", ""),
                "metadata": metadata,
            })
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping {run_dir.name}: {e}")

    runs.sort(key=lambda r: r["timestamp"], reverse=True)
    return runs


def load_predictions(run_path: Path) -> Optional[pl.DataFrame]:
    """Load combined predictions.parquet for a run."""
    path = run_path / "predictions.parquet"
    if path.exists():
        return pl.read_parquet(path)
    return None


def load_summary_metrics(run_path: Path) -> Optional[pl.DataFrame]:
    """Load summary_metrics.csv for a run."""
    path = run_path / "summary_metrics.csv"
    if path.exists():
        return pl.read_csv(path)
    return None
