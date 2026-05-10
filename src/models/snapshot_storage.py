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

    # --- Experiment paths ---

    def experiment_dir(self, model_type: str, candidate: str, version: int) -> Path:
        return (
            self.snapshot_dir / "experiments" / model_type / candidate / f"v{version}"
        )

    def result_dir(
        self, model_type: str, candidate: str, version: int, split_name: str,
    ) -> Path:
        return self.experiment_dir(model_type, candidate, version) / "results" / split_name

    def list_candidate_versions(self, model_type: str, candidate: str) -> List[int]:
        cand_dir = self.snapshot_dir / "experiments" / model_type / candidate
        if not cand_dir.exists():
            return []
        out: List[int] = []
        for child in cand_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("v"):
                continue
            try:
                out.append(int(child.name[1:]))
            except ValueError:
                continue
        return sorted(out)

    def next_candidate_version(self, model_type: str, candidate: str) -> int:
        existing = self.list_candidate_versions(model_type, candidate)
        return (existing[-1] if existing else 0) + 1

    # --- Candidate-level artifacts ---

    def _ensure(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_candidate_config(
        self, model_type: str, candidate: str, version: int, config: Dict[str, Any]
    ) -> Path:
        path = self._ensure(self.experiment_dir(model_type, candidate, version) / "config.json")
        path.write_text(json.dumps(config, indent=2, default=str))
        return path

    def load_candidate_config(
        self, model_type: str, candidate: str, version: int
    ) -> Optional[Dict[str, Any]]:
        path = self.experiment_dir(model_type, candidate, version) / "config.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def save_candidate_registration(
        self, model_type: str, candidate: str, version: int, registration: Dict[str, Any]
    ) -> Path:
        path = self._ensure(
            self.experiment_dir(model_type, candidate, version) / "registration.json"
        )
        path.write_text(json.dumps(registration, indent=2, default=str))
        return path

    def load_candidate_registration(
        self, model_type: str, candidate: str, version: int
    ) -> Optional[Dict[str, Any]]:
        path = self.experiment_dir(model_type, candidate, version) / "registration.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def save_finalized_pipeline(
        self, model_type: str, candidate: str, version: int, pipeline: Any
    ) -> Path:
        path = self._ensure(
            self.experiment_dir(model_type, candidate, version) / "finalized.pkl"
        )
        path.write_bytes(pickle.dumps(pipeline))
        return path

    def load_finalized_pipeline(
        self, model_type: str, candidate: str, version: int
    ) -> Optional[Any]:
        path = self.experiment_dir(model_type, candidate, version) / "finalized.pkl"
        if not path.exists():
            return None
        return pickle.loads(path.read_bytes())

    # --- Per-result artifacts ---

    def save_result(
        self,
        model_type: str,
        candidate: str,
        version: int,
        split_name: str,
        pipeline: Any,
        metrics: Dict[str, Any],
        parameters: Dict[str, Any],
        tune_predictions: Optional[pl.DataFrame] = None,
        test_predictions: Optional[pl.DataFrame] = None,
        score_predictions: Optional[pl.DataFrame] = None,
        feature_importance: Optional[pl.DataFrame] = None,
    ) -> Path:
        rdir = self.result_dir(model_type, candidate, version, split_name)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "pipeline.pkl").write_bytes(pickle.dumps(pipeline))
        (rdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        (rdir / "parameters.json").write_text(json.dumps(parameters, indent=2, default=str))

        preds_dir = rdir / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        if tune_predictions is not None:
            tune_predictions.write_parquet(preds_dir / "tune.parquet")
        if test_predictions is not None:
            test_predictions.write_parquet(preds_dir / "test.parquet")
        if score_predictions is not None:
            score_predictions.write_parquet(preds_dir / "score.parquet")
        if feature_importance is not None:
            feature_importance.write_csv(rdir / "feature_importance.csv")
        return rdir

    def load_result(
        self, model_type: str, candidate: str, version: int, split_name: str,
    ) -> Optional[Dict[str, Any]]:
        rdir = self.result_dir(model_type, candidate, version, split_name)
        if not rdir.exists():
            return None
        out: Dict[str, Any] = {}
        out["pipeline"] = pickle.loads((rdir / "pipeline.pkl").read_bytes())
        out["metrics"] = json.loads((rdir / "metrics.json").read_text())
        out["parameters"] = json.loads((rdir / "parameters.json").read_text())
        for fold in ["tune", "test", "score"]:
            p = rdir / "predictions" / f"{fold}.parquet"
            if p.exists():
                out[f"{fold}_predictions"] = pl.read_parquet(p)
        return out

    def load_score_predictions(
        self, model_type: str, candidate: str, version: int, split_name: str,
    ) -> Optional[pl.DataFrame]:
        p = self.result_dir(model_type, candidate, version, split_name) / "predictions" / "score.parquet"
        if not p.exists():
            return None
        return pl.read_parquet(p)

    # --- Simulation artifacts ---

    def simulation_dir(self, simulation_name: str, split_name: str, version: int) -> Path:
        return self.snapshot_dir / "simulations" / simulation_name / split_name / f"v{version}"

    def list_simulation_versions(self, simulation_name: str, split_name: str) -> List[int]:
        sim_dir = self.snapshot_dir / "simulations" / simulation_name / split_name
        if not sim_dir.exists():
            return []
        out: List[int] = []
        for child in sim_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("v"):
                continue
            try:
                out.append(int(child.name[1:]))
            except ValueError:
                continue
        return sorted(out)

    def next_simulation_version(self, simulation_name: str, split_name: str) -> int:
        existing = self.list_simulation_versions(simulation_name, split_name)
        return (existing[-1] if existing else 0) + 1

    def save_simulation(
        self,
        simulation_name: str,
        split_name: str,
        version: int,
        registration: Dict[str, Any],
        metrics: Dict[str, Any],
        predictions: pl.DataFrame,
    ) -> Path:
        rdir = self.simulation_dir(simulation_name, split_name, version)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "registration.json").write_text(json.dumps(registration, indent=2, default=str))
        (rdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        predictions.write_parquet(rdir / "predictions.parquet")
        return rdir

    def load_simulation(
        self, simulation_name: str, split_name: str, version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if version is None:
            versions = self.list_simulation_versions(simulation_name, split_name)
            if not versions:
                return None
            version = versions[-1]
        rdir = self.simulation_dir(simulation_name, split_name, version)
        if not rdir.exists():
            return None
        return {
            "registration": json.loads((rdir / "registration.json").read_text()),
            "metrics": json.loads((rdir / "metrics.json").read_text()),
            "predictions": pl.read_parquet(rdir / "predictions.parquet"),
        }
