"""Loader for experiment results from local disk."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import polars as pl

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("models/experiments")
SKIP_DIRS = {"predictions"}


def discover_experiments(base_dir: Path = EXPERIMENTS_DIR) -> List[Dict[str, Any]]:
    """Scan for experiments containing metadata.json.

    Returns list of dicts with keys: outcome, name, version, path, metadata,
    model_info, is_eval, is_finalized.
    Sorted by outcome then name.
    """
    experiments = []
    if not base_dir.exists():
        return experiments

    for outcome_dir in sorted(base_dir.iterdir()):
        if not outcome_dir.is_dir() or outcome_dir.name in SKIP_DIRS:
            continue

        for exp_dir in sorted(outcome_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            for version_dir in sorted(exp_dir.iterdir()):
                if not version_dir.is_dir():
                    continue

                metadata_path = version_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                    model_info = None
                    model_info_path = version_dir / "model_info.json"
                    if model_info_path.exists():
                        with open(model_info_path) as f:
                            model_info = json.load(f)

                    experiments.append({
                        "outcome": outcome_dir.name,
                        "name": exp_dir.name,
                        "version": version_dir.name,
                        "path": version_dir,
                        "metadata": metadata,
                        "model_info": model_info,
                        "is_eval": exp_dir.name.startswith("eval-"),
                        "is_finalized": (version_dir / "finalized").is_dir(),
                    })
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Skipping {version_dir}: {e}")

    return experiments


def load_metrics(exp_path: Path, dataset: str = "test") -> Optional[Dict[str, Any]]:
    """Load metrics JSON for a dataset split."""
    path = exp_path / f"{dataset}_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_predictions(exp_path: Path, dataset: str = "test") -> Optional[pl.DataFrame]:
    """Load predictions parquet for a dataset split.

    Returns DataFrame with only the essential columns for display.
    """
    path = exp_path / f"{dataset}_predictions.parquet"
    if not path.exists():
        return None

    df = pl.read_parquet(path)

    keep_cols = []
    for col in ["game_id", "name", "year_published", "users_rated",
                 "prediction", "actual",
                 "predicted_proba_class_0", "predicted_proba_class_1"]:
        if col in df.columns:
            keep_cols.append(col)

    return df.select(keep_cols) if keep_cols else df


def load_coefficients(exp_path: Path) -> Optional[pl.DataFrame]:
    """Load coefficients.csv if it exists."""
    path = exp_path / "coefficients.csv"
    if path.exists():
        return pl.read_csv(path)
    return None
