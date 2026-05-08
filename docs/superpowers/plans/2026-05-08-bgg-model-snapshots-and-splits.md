# BGG Model Snapshots and Splits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorient bgg-rating-models training around a versioned data snapshot with named/versioned splits underneath, so experiments are honestly comparable on disk-location alone. Cascading dependencies (rating reads predicted_complexity) resolve within a fixed (snapshot, split) surface. K-fold OOF scoring for upstream models eliminates training-time leakage in downstream features.

**Architecture:** Two clean layers.

- **Pure model layer (`src/models/outcomes/`)**: takes data frames in, returns fitted pipelines + metrics + predictions out. No CLI, no IO, no knowledge of snapshots/splits/upstream cascades. The existing helpers (`tune_model`, `evaluate_model`, model classes' `configure_model`, `find_optimal_threshold`, etc.) already live here and are kept; what gets factored out of `outcomes/train.py` is a function `train_one(model, candidate_config, train_df, tune_df, test_df, ...)` that returns `{pipeline, metrics, parameters, tune_predictions, test_predictions}`.
- **Orchestration layer (`src/pipeline/`)**: knows about snapshots, splits, candidates, upstream cascades, and IO. `pipeline.train` becomes the snapshot-aware orchestrator — parses `--snapshot-version --candidate --splits --upstream`, loads from `SnapshotStorage`, joins upstream `score.parquet`, calls `train_one` per split, writes results back. Same name `pipeline.train` as today; the Makefile keeps working.

`SnapshotStorage` (Stage 1) owns paths and I/O for the new tree at `models/experiments/_snapshots/v{N}/`. OOF is layered on after the basic flow works.

**Tech Stack:** Python 3.12, Polars, scikit-learn, pytest, pandas (interop), uv. Existing project conventions: `tmp_path` for hermetic tests, `uv run python -m` for module execution.

**Branch:** Work on a redesign branch (`redesign/snapshots-and-splits` or similar). Production training continues on `main` until this is ready to merge.

**Spec:** `docs/superpowers/specs/2026-05-08-bgg-model-snapshots-and-splits-design.md`

---

## File Structure

**New files (all under `src/models/`):**

- `snapshot_storage.py` — `SnapshotStorage` class. Owns paths, IO, version helpers for `_snapshots/v{N}/{universe.parquet, splits/, experiments/}`.
- `build_snapshot.py` — CLI. Loads features (BQ via `BGGDataLoader` or local parquet), writes `_snapshots/v{N}/universe.parquet` + `metadata.json`.
- `build_split.py` — CLI. Reads a snapshot, slices by year via `time_based_split`, writes split parquets + `metadata.json`. Supports `--yoy` mode.

**Modified files:**

- `src/models/outcomes/train.py` — strip CLI/IO/argparse/ExperimentTracker concerns. Keep model wiring (preprocessor, configure_model, tune_model, refit on train+tune, threshold optimization, additional_metrics). Expose `train_one(model, candidate_config, train_df, tune_df, test_df, ...)` returning a dict of artifacts.
- `src/pipeline/train.py` — rewrite from a one-line shim into the snapshot-aware orchestrator. Parses snapshot/split/candidate/upstream args, loads from `SnapshotStorage`, joins upstream score columns, calls `train_one` per split, writes results via `SnapshotStorage`.
- `src/pipeline/score.py` — rewrite to score the snapshot universe and write `score.parquet` per (candidate, split) into `SnapshotStorage`.
- `src/pipeline/finalize.py` — rewrite to refit a candidate on the full snapshot universe and write `finalized.pkl` at the candidate level (not per split).
- `config.yaml` — add `candidates` list per model type (Task 11, already done).

**Test files (new):**

- `tests/test_snapshot_storage.py` — hermetic, tmp_path-based.
- `tests/test_build_snapshot.py` — local parquet input, asserts on file output.
- `tests/test_build_split.py` — round-trip a synthetic snapshot, verify split contents.
- `tests/test_candidate_config.py` — YAML-based candidate lookup.
- `tests/test_train_one.py` — pure model layer: takes synthetic frames, returns artifacts.
- `tests/test_pipeline_train_snapshot.py` — orchestration: build snapshot, build split, run pipeline.train, verify SnapshotStorage tree contents.
- `tests/test_oof_scoring.py` — added in Stage 4, validates K-fold OOF behavior.

**Out of scope for this plan (per spec):**

- Streamlit/Quarto adaptation
- GCS sync
- Migration of legacy artifacts
- Snapshot diffing tools

---

## Stage 1: Snapshot and Split Storage

The foundation. No model training yet — just the storage layer and the two builder CLIs. End state: you can build a snapshot, derive splits from it, and round-trip them through `SnapshotStorage`.

### Task 1: Create branch and initial snapshot storage scaffolding

**Files:**
- Create: `src/models/snapshot_storage.py`

- [ ] **Step 1: Create the redesign branch**

```bash
git checkout -b redesign/snapshots-and-splits
```

- [ ] **Step 2: Create the empty module file**

Create `src/models/snapshot_storage.py` with this skeleton:

```python
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
```

- [ ] **Step 3: Verify it imports**

Run: `uv run python -c "from src.models.snapshot_storage import SnapshotStorage; print(SnapshotStorage(1, '/tmp/snap_test'))"`
Expected: prints a `<src.models.snapshot_storage.SnapshotStorage object at 0x...>` line.

- [ ] **Step 4: Commit**

```bash
git add src/models/snapshot_storage.py
git commit -m "feat: add SnapshotStorage skeleton on redesign branch"
```

### Task 2: Snapshot version helpers

**Files:**
- Modify: `src/models/snapshot_storage.py`
- Test: `tests/test_snapshot_storage.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_snapshot_storage.py`:

```python
"""Tests for SnapshotStorage.

Hermetic: uses pytest's ``tmp_path`` for all I/O. No BigQuery, no network.
"""

from pathlib import Path

import polars as pl

from src.models.snapshot_storage import SnapshotStorage


def test_latest_version_with_no_snapshots(tmp_path: Path) -> None:
    # When no snapshots exist, latest_version returns None.
    base = tmp_path / "snapshots"
    base.mkdir()
    assert SnapshotStorage.latest_version(base_dir=base) is None


def test_next_version_with_no_snapshots(tmp_path: Path) -> None:
    base = tmp_path / "snapshots"
    base.mkdir()
    assert SnapshotStorage.next_version(base_dir=base) == 1


def test_latest_version_picks_highest(tmp_path: Path) -> None:
    base = tmp_path / "snapshots"
    (base / "v1").mkdir(parents=True)
    (base / "v3").mkdir(parents=True)
    (base / "v2").mkdir(parents=True)
    assert SnapshotStorage.latest_version(base_dir=base) == 3
    assert SnapshotStorage.next_version(base_dir=base) == 4
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_snapshot_storage.py -v`
Expected: FAIL with `AttributeError: type object 'SnapshotStorage' has no attribute 'latest_version'`.

- [ ] **Step 3: Implement classmethods**

Add to `src/models/snapshot_storage.py`:

```python
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
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/test_snapshot_storage.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/snapshot_storage.py tests/test_snapshot_storage.py
git commit -m "feat: SnapshotStorage version discovery helpers"
```

### Task 3: Universe save/load + metadata round-trip

**Files:**
- Modify: `src/models/snapshot_storage.py`
- Test: `tests/test_snapshot_storage.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_snapshot_storage.py`:

```python
def test_save_and_load_universe_roundtrip(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    df = pl.DataFrame({
        "game_id": [1, 2, 3],
        "year_published": [2018, 2019, 2020],
        "rating": [7.0, 8.0, 6.5],
    })
    storage.save_universe(df)
    loaded = storage.load_universe()
    assert loaded is not None
    assert loaded.equals(df)


def test_save_and_load_metadata_roundtrip(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    meta = {"created_at": "2026-05-08", "n_rows": 3, "use_embeddings": True}
    storage.save_metadata(meta)
    assert storage.load_metadata() == meta


def test_load_universe_when_missing(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    assert storage.load_universe() is None
```

- [ ] **Step 2: Run tests, expect failures**

Run: `uv run pytest tests/test_snapshot_storage.py -v`
Expected: 3 new test failures with `AttributeError: 'SnapshotStorage' object has no attribute 'save_universe'` (etc).

- [ ] **Step 3: Implement save/load methods**

Add to `SnapshotStorage`:

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_snapshot_storage.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/snapshot_storage.py tests/test_snapshot_storage.py
git commit -m "feat: SnapshotStorage universe + metadata IO"
```

### Task 4: Split save/load + listing

**Files:**
- Modify: `src/models/snapshot_storage.py`
- Test: `tests/test_snapshot_storage.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_snapshot_storage.py`:

```python
def test_save_and_load_split_roundtrip(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    train = pl.DataFrame({"game_id": [1, 2], "year_published": [2018, 2019]})
    tune = pl.DataFrame({"game_id": [3], "year_published": [2020]})
    test = pl.DataFrame({"game_id": [4], "year_published": [2021]})
    meta = {"train_through": 2019, "tune_start": 2020, "tune_through": 2020,
            "test_start": 2021, "test_through": 2021, "time_col": "year_published"}

    storage.save_split("standard", train, tune, test, meta)
    loaded = storage.load_split("standard")
    assert loaded is not None
    assert loaded["train"].equals(train)
    assert loaded["tune"].equals(tune)
    assert loaded["test"].equals(test)
    assert loaded["metadata"] == meta


def test_load_split_when_missing(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    assert storage.load_split("standard") is None


def test_list_splits(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    df = pl.DataFrame({"game_id": [1], "year_published": [2018]})
    meta = {"x": 1}
    storage.save_split("standard", df, df, df, meta)
    storage.save_split("yoy_2018", df, df, df, meta)
    storage.save_split("yoy_2019", df, df, df, meta)
    assert sorted(storage.list_splits()) == ["standard", "yoy_2018", "yoy_2019"]
```

- [ ] **Step 2: Run tests, expect failure**

Run: `uv run pytest tests/test_snapshot_storage.py -v`
Expected: 3 new failures.

- [ ] **Step 3: Implement split methods**

Add to `SnapshotStorage`:

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_snapshot_storage.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/snapshot_storage.py tests/test_snapshot_storage.py
git commit -m "feat: SnapshotStorage split IO + listing"
```

### Task 5: build_snapshot.py CLI

**Files:**
- Create: `src/models/build_snapshot.py`
- Test: `tests/test_build_snapshot.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_build_snapshot.py`:

```python
"""Hermetic test for build_snapshot CLI using a local-parquet input path."""

from pathlib import Path

import polars as pl

from src.models.build_snapshot import build_snapshot
from src.models.snapshot_storage import SnapshotStorage


def test_build_snapshot_from_local_parquet(tmp_path: Path) -> None:
    # Synthetic source data
    source = pl.DataFrame({
        "game_id": [1, 2, 3, 4],
        "year_published": [2018, 2019, 2020, 2021],
        "rating": [7.0, 8.0, 6.5, 7.2],
        "users_rated": [100, 200, 50, 150],
    })
    source_path = tmp_path / "source.parquet"
    source.write_parquet(source_path)

    base_dir = tmp_path / "snaps"

    version = build_snapshot(
        local_data=source_path,
        base_dir=base_dir,
        use_embeddings=False,
    )
    assert version == 1

    storage = SnapshotStorage(snapshot_version=version, base_dir=base_dir)
    universe = storage.load_universe()
    assert universe is not None
    assert universe.equals(source)

    meta = storage.load_metadata()
    assert meta is not None
    assert meta["n_rows"] == 4
    assert meta["use_embeddings"] is False
    assert "created_at" in meta
    assert meta["columns"] == source.columns


def test_build_snapshot_increments_version(tmp_path: Path) -> None:
    base_dir = tmp_path / "snaps"
    source = pl.DataFrame({"game_id": [1], "year_published": [2018]})
    source_path = tmp_path / "src.parquet"
    source.write_parquet(source_path)

    v1 = build_snapshot(local_data=source_path, base_dir=base_dir, use_embeddings=False)
    v2 = build_snapshot(local_data=source_path, base_dir=base_dir, use_embeddings=False)
    assert v1 == 1
    assert v2 == 2
```

- [ ] **Step 2: Run tests, expect failure**

Run: `uv run pytest tests/test_build_snapshot.py -v`
Expected: ImportError — module does not exist.

- [ ] **Step 3: Implement build_snapshot.py**

Create `src/models/build_snapshot.py`:

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_build_snapshot.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/build_snapshot.py tests/test_build_snapshot.py
git commit -m "feat: build_snapshot CLI"
```

### Task 6: build_split.py CLI (single split mode)

**Files:**
- Create: `src/models/build_split.py`
- Test: `tests/test_build_split.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_build_split.py`:

```python
"""Hermetic test for build_split CLI."""

from pathlib import Path

import polars as pl
import pytest

from src.models.build_snapshot import build_snapshot
from src.models.build_split import build_split
from src.models.snapshot_storage import SnapshotStorage


def _make_snapshot(tmp_path: Path) -> int:
    df = pl.DataFrame({
        "game_id": list(range(1, 21)),
        "year_published": [2018]*5 + [2019]*5 + [2020]*5 + [2021]*5,
        "rating": [7.0] * 20,
    })
    src = tmp_path / "src.parquet"
    df.write_parquet(src)
    return build_snapshot(
        local_data=src, base_dir=tmp_path / "snaps", use_embeddings=False,
    )


def test_build_standard_split(tmp_path: Path) -> None:
    base = tmp_path / "snaps"
    v = _make_snapshot(tmp_path)

    build_split(
        snapshot_version=v,
        split_name="standard",
        train_through=2019,
        tune_start=2020,
        tune_through=2020,
        test_start=2021,
        test_through=2021,
        base_dir=base,
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    split = storage.load_split("standard")
    assert split is not None
    assert split["train"].height == 10  # 2018 + 2019
    assert split["tune"].height == 5     # 2020
    assert split["test"].height == 5     # 2021

    meta = split["metadata"]
    assert meta["train_through"] == 2019
    assert meta["tune_start"] == 2020
    assert meta["tune_through"] == 2020


def test_build_split_errors_on_missing_snapshot(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_split(
            snapshot_version=99,
            split_name="standard",
            train_through=2019, tune_start=2020, tune_through=2020,
            test_start=2021, test_through=2021,
            base_dir=tmp_path / "snaps",
        )
```

- [ ] **Step 2: Run tests, expect failure**

Run: `uv run pytest tests/test_build_split.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement build_split.py (single mode only)**

Create `src/models/build_split.py`:

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_build_split.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/build_split.py tests/test_build_split.py
git commit -m "feat: build_split CLI (single mode)"
```

### Task 7: YoY mode for build_split

**Files:**
- Modify: `src/models/build_split.py`
- Test: `tests/test_build_split.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_build_split.py`:

```python
def test_build_yoy_splits(tmp_path: Path) -> None:
    from src.models.build_split import build_yoy_splits

    base = tmp_path / "snaps"

    df = pl.DataFrame({
        "game_id": list(range(1, 41)),
        "year_published": sum([[y]*5 for y in range(2014, 2022)], []),
        "rating": [7.0] * 40,
    })
    src = tmp_path / "src.parquet"
    df.write_parquet(src)
    v = build_snapshot(local_data=src, base_dir=base, use_embeddings=False)

    # Years 2018..2020 → splits yoy_2018, yoy_2019, yoy_2020
    # Each test year y → train through y-2, tune y-1, test y
    build_yoy_splits(
        snapshot_version=v,
        yoy_start=2018,
        yoy_end=2020,
        base_dir=base,
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    splits = storage.list_splits()
    assert "yoy_2018" in splits
    assert "yoy_2019" in splits
    assert "yoy_2020" in splits

    # yoy_2019: train≤2017, tune=2018, test=2019
    s = storage.load_split("yoy_2019")
    assert s["metadata"]["train_through"] == 2017
    assert s["metadata"]["tune_start"] == 2018
    assert s["metadata"]["test_start"] == 2019
    # train rows = 2014..2017 = 4 years × 5 = 20
    assert s["train"].height == 20
    assert s["tune"].height == 5
    assert s["test"].height == 5
```

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/test_build_split.py::test_build_yoy_splits -v`
Expected: ImportError on `build_yoy_splits`.

- [ ] **Step 3: Implement YoY builder**

Append to `src/models/build_split.py`:

```python
def build_yoy_splits(
    snapshot_version: int,
    yoy_start: int,
    yoy_end: int,
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
    time_col: str = "year_published",
) -> list:
    """Generate the YoY family of splits.

    For each test year y in [yoy_start, yoy_end], creates split ``yoy_{y}``
    with train through y-2, tune on y-1, test on y. Mirrors the logic in
    ``src/models/time_based_evaluation.py::generate_time_splits``.
    """
    paths = []
    for test_year in range(yoy_start, yoy_end + 1):
        result = build_split(
            snapshot_version=snapshot_version,
            split_name=f"yoy_{test_year}",
            train_through=test_year - 2,
            tune_start=test_year - 1,
            tune_through=test_year - 1,
            test_start=test_year,
            test_through=test_year,
            base_dir=base_dir,
            time_col=time_col,
        )
        paths.append(result)
    logger.info(f"Built {len(paths)} YoY splits ({yoy_start}..{yoy_end})")
    return paths
```

Then extend `main()` to support `--yoy`. Find this block in `main()`:

```python
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()
```

Replace it with:

```python
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--yoy", action="store_true", default=False,
                        help="Build a family of YoY splits instead of one named split")
    parser.add_argument("--yoy-start", type=int, default=None)
    parser.add_argument("--yoy-end", type=int, default=None)
    args = parser.parse_args()

    if args.yoy:
        setup_logging()
        if args.yoy_start is None or args.yoy_end is None:
            config = load_config()
            args.yoy_start = args.yoy_start or config.years.eval.start
            args.yoy_end = args.yoy_end or config.years.eval.end
        build_yoy_splits(
            snapshot_version=args.snapshot_version,
            yoy_start=args.yoy_start,
            yoy_end=args.yoy_end,
            base_dir=args.base_dir,
        )
        print(f"yoy splits: v{args.snapshot_version}/yoy_{args.yoy_start}..yoy_{args.yoy_end}")
        return 0

```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_build_split.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/build_split.py tests/test_build_split.py
git commit -m "feat: build_split --yoy mode"
```

---

## Stage 2: Experiment Storage and Single-Candidate Training

End state: you can train one model (no upstream dependencies — start with `hurdle`) on one split using the snapshot framework. Cascading and OOF come later.

### Task 8: Experiment path helpers in SnapshotStorage

**Files:**
- Modify: `src/models/snapshot_storage.py`
- Test: `tests/test_snapshot_storage.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_snapshot_storage.py`:

```python
def test_experiment_paths(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")

    exp_dir = storage.experiment_dir("hurdle", "logistic-hurdle", 1)
    assert str(exp_dir).endswith(
        "v1/experiments/hurdle/logistic-hurdle/v1"
    )

    result_dir = storage.result_dir("hurdle", "logistic-hurdle", 1, "standard")
    assert str(result_dir).endswith(
        "v1/experiments/hurdle/logistic-hurdle/v1/results/standard"
    )


def test_next_candidate_version(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")

    assert storage.next_candidate_version("hurdle", "logistic-hurdle") == 1
    # Manually create v1 and v2 dirs
    storage.experiment_dir("hurdle", "logistic-hurdle", 1).mkdir(parents=True)
    storage.experiment_dir("hurdle", "logistic-hurdle", 2).mkdir(parents=True)
    assert storage.next_candidate_version("hurdle", "logistic-hurdle") == 3
```

- [ ] **Step 2: Run tests, expect failure**

Run: `uv run pytest tests/test_snapshot_storage.py::test_experiment_paths tests/test_snapshot_storage.py::test_next_candidate_version -v`
Expected: AttributeError.

- [ ] **Step 3: Implement path helpers**

Add to `SnapshotStorage`:

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_snapshot_storage.py -v`
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add src/models/snapshot_storage.py tests/test_snapshot_storage.py
git commit -m "feat: SnapshotStorage experiment path helpers"
```

### Task 9: Experiment artifact persistence in SnapshotStorage

**Files:**
- Modify: `src/models/snapshot_storage.py`
- Test: `tests/test_snapshot_storage.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_snapshot_storage.py`:

```python
def test_save_and_load_candidate_config_and_registration(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")

    config = {"name": "logistic-hurdle", "algorithm": "logistic", "use_embeddings": True}
    registration = {"snapshot_version": 1, "candidate": "logistic-hurdle",
                    "version": 1, "upstream_experiments": {}}

    storage.save_candidate_config("hurdle", "logistic-hurdle", 1, config)
    storage.save_candidate_registration("hurdle", "logistic-hurdle", 1, registration)

    loaded_cfg = storage.load_candidate_config("hurdle", "logistic-hurdle", 1)
    loaded_reg = storage.load_candidate_registration("hurdle", "logistic-hurdle", 1)
    assert loaded_cfg == config
    assert loaded_reg == registration


def test_save_and_load_candidate_finalized(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")

    obj = {"my": "pipeline"}  # any picklable object
    storage.save_finalized_pipeline("hurdle", "logistic-hurdle", 1, obj)
    loaded = storage.load_finalized_pipeline("hurdle", "logistic-hurdle", 1)
    assert loaded == obj
```

- [ ] **Step 2: Run tests, expect failure**

Run: `uv run pytest tests/test_snapshot_storage.py -v`
Expected: AttributeError on the new methods.

- [ ] **Step 3: Implement persistence helpers**

Add to `SnapshotStorage`:

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_snapshot_storage.py -v`
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add src/models/snapshot_storage.py tests/test_snapshot_storage.py
git commit -m "feat: SnapshotStorage candidate-level config/registration/finalized IO"
```

### Task 10: Per-result artifact persistence

**Files:**
- Modify: `src/models/snapshot_storage.py`
- Test: `tests/test_snapshot_storage.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_snapshot_storage.py`:

```python
def test_save_and_load_result_artifacts(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")

    pipeline_obj = {"pipeline": "obj"}
    metrics = {"train": {"rmse": 0.5}, "tune": {"rmse": 0.6}, "test": {"rmse": 0.7}}
    params = {"alpha": 1.0}
    tune_preds = pl.DataFrame({"game_id": [1, 2], "prediction": [0.5, 0.6], "actual": [0.4, 0.7]})
    test_preds = pl.DataFrame({"game_id": [3], "prediction": [0.8], "actual": [0.7]})
    score_preds = pl.DataFrame({"game_id": [1, 2, 3, 4], "predicted_complexity": [2.0, 2.5, 3.0, 3.5]})

    storage.save_result(
        model_type="complexity",
        candidate="ard-complexity",
        version=1,
        split_name="standard",
        pipeline=pipeline_obj,
        metrics=metrics,
        parameters=params,
        tune_predictions=tune_preds,
        test_predictions=test_preds,
        score_predictions=score_preds,
    )

    loaded = storage.load_result("complexity", "ard-complexity", 1, "standard")
    assert loaded["pipeline"] == pipeline_obj
    assert loaded["metrics"] == metrics
    assert loaded["parameters"] == params
    assert loaded["tune_predictions"].equals(tune_preds)
    assert loaded["test_predictions"].equals(test_preds)
    assert loaded["score_predictions"].equals(score_preds)


def test_load_score_predictions_helper(tmp_path: Path) -> None:
    storage = SnapshotStorage(snapshot_version=1, base_dir=tmp_path / "snaps")
    score = pl.DataFrame({"game_id": [1, 2], "predicted_complexity": [2.0, 2.5]})
    storage.save_result(
        model_type="complexity", candidate="ard-complexity", version=1,
        split_name="standard", pipeline={}, metrics={}, parameters={},
        score_predictions=score,
    )
    loaded = storage.load_score_predictions("complexity", "ard-complexity", 1, "standard")
    assert loaded.equals(score)
```

- [ ] **Step 2: Run tests, expect failure**

Run: `uv run pytest tests/test_snapshot_storage.py -v`
Expected: AttributeError.

- [ ] **Step 3: Implement save_result / load_result**

Add to `SnapshotStorage`:

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_snapshot_storage.py -v`
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add src/models/snapshot_storage.py tests/test_snapshot_storage.py
git commit -m "feat: SnapshotStorage per-result artifact IO"
```

### Task 11: Add candidates lists to config.yaml

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add candidate lists to existing model blocks**

Find the existing `models:` block (around line 49 in `config.yaml`). For each of `hurdle`, `complexity`, `rating`, `users_rated`, `geek_rating`, add a `candidates` list. Replace the existing `models:` block with:

```yaml
# Model Configuration
models:
  predictions_dir: "./models/experiments/predictions"
  hurdle:
    type: logistic
    experiment_name: logistic-hurdle
    use_embeddings: true
    candidates:
      - name: logistic-hurdle
        algorithm: logistic
        use_embeddings: true
  complexity:
    type: ard
    experiment_name: ard-complexity
    use_sample_weights: false
    use_embeddings: true
    candidates:
      - name: ard-complexity
        algorithm: ard
        use_sample_weights: false
        use_embeddings: true
  rating:
    type: ard
    experiment_name: ard-ridge-rating
    use_sample_weights: false
    min_ratings: 5
    use_embeddings: true
    candidates:
      - name: ard-ridge-rating
        algorithm: ard
        use_sample_weights: false
        min_ratings: 5
        use_embeddings: true
        upstream:
          complexity: ard-complexity
  users_rated:
    type: ard
    experiment_name: ard-ridge-users_rated
    use_sample_weights: false
    min_ratings: 0
    use_embeddings: true
    candidates:
      - name: ard-ridge-users_rated
        algorithm: ard
        use_sample_weights: false
        min_ratings: 0
        use_embeddings: true
        upstream:
          complexity: ard-complexity
  geek_rating:
    type: ard
    experiment_name: ard-geek_rating
    mode: direct
    min_ratings: 25
    use_embeddings: true
    include_predictions: true
    candidates:
      - name: ard-geek_rating
        algorithm: ard
        mode: direct
        min_ratings: 25
        use_embeddings: true
        include_predictions: true
        upstream:
          complexity: ard-complexity
          rating: ard-ridge-rating
          users_rated: ard-ridge-users_rated
```

- [ ] **Step 2: Verify config still loads**

Run: `uv run python -c "from src.utils.config import load_config; c = load_config(); print(c.models['rating'])"`
Expected: prints rating config without errors. The new `candidates` field is preserved (config loader is dict-based).

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "feat: add candidates lists to config.yaml model blocks"
```

### Task 12: Candidate config helper

**Files:**
- Create: `src/models/candidate_config.py`
- Test: `tests/test_candidate_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_candidate_config.py`:

```python
"""Tests for candidate config resolution."""

from pathlib import Path

import pytest
import yaml

from src.models.candidate_config import find_candidate, list_candidates


def _write_config(tmp_path: Path, contents: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(contents))
    return p


def test_find_candidate_returns_block(tmp_path: Path) -> None:
    p = _write_config(tmp_path, {
        "models": {
            "rating": {
                "candidates": [
                    {"name": "ard-ridge-rating", "algorithm": "ard"},
                    {"name": "catboost-rating", "algorithm": "catboost"},
                ],
            },
        },
    })
    cfg = find_candidate(config_path=p, model_type="rating", candidate="catboost-rating")
    assert cfg["algorithm"] == "catboost"
    assert cfg["name"] == "catboost-rating"


def test_find_candidate_raises_when_missing(tmp_path: Path) -> None:
    p = _write_config(tmp_path, {"models": {"rating": {"candidates": []}}})
    with pytest.raises(KeyError):
        find_candidate(config_path=p, model_type="rating", candidate="nope")


def test_list_candidates(tmp_path: Path) -> None:
    p = _write_config(tmp_path, {
        "models": {
            "rating": {
                "candidates": [
                    {"name": "a", "algorithm": "x"},
                    {"name": "b", "algorithm": "y"},
                ],
            },
        },
    })
    assert list_candidates(config_path=p, model_type="rating") == ["a", "b"]
```

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/test_candidate_config.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement candidate_config.py**

Create `src/models/candidate_config.py`:

```python
"""Candidate config resolution from config.yaml.

A candidate is a recipe: algorithm, preprocessor settings, optional
upstream model choices. Defined in ``models.{type}.candidates`` in
config.yaml. This module reads that config without going through the
project's typed Config object so the candidate block remains a free-form
dict the trainer copies verbatim into the experiment's ``config.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


def _load_raw(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    p = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return yaml.safe_load(p.read_text())


def list_candidates(
    model_type: str, config_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    raw = _load_raw(config_path)
    cands = (raw.get("models") or {}).get(model_type, {}).get("candidates") or []
    return [c["name"] for c in cands]


def find_candidate(
    model_type: str, candidate: str, config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    raw = _load_raw(config_path)
    cands = (raw.get("models") or {}).get(model_type, {}).get("candidates") or []
    for c in cands:
        if c.get("name") == candidate:
            return c
    raise KeyError(
        f"No candidate {candidate!r} under models.{model_type}.candidates "
        f"(available: {[c.get('name') for c in cands]})"
    )
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_candidate_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/candidate_config.py tests/test_candidate_config.py
git commit -m "feat: candidate config resolution"
```

---

## Stage 3: Pure-function refactor + snapshot-aware orchestration

Stage 1 (storage) and Stage 2 (candidate config) are complete. Stage 3 extracts a pure `train_one(...)` function from `outcomes/train.py`, then rewrites `src/pipeline/train.py`, `src/pipeline/score.py`, and (in Stage 5) `src/pipeline/finalize.py` as snapshot-aware orchestrators.

After this stage, `train_one` takes data frames in and returns artifacts out — no CLI, no IO, no `ExperimentTracker`. The orchestration layer (`pipeline.train`, `pipeline.score`) owns argv, snapshot/split loading, upstream cascade resolution, and writes to `SnapshotStorage`.

This refactor preserves all existing training behavior: preprocessor construction, `configure_model`, `tune_model`, refit-on-train+tune, threshold optimization for hurdle, sample-weight handling, additional metrics, etc. It just stops writing to disk and stops parsing argv.

The legacy `main()` and `parse_arguments()` in `outcomes/train.py` are deleted on this branch — production stays on `main` until merge.

### Task 13: Extract train_one as a pure function

**Files:**
- Modify: `src/models/outcomes/train.py`
- Test: `tests/test_train_one.py`

The existing `outcomes/train.py:train_model` does too many things. Refactor it: keep the model-wiring logic, drop everything that touches argv, config files, or `ExperimentTracker`. Expose a `train_one` function that takes frames and a candidate dict, returns artifacts.

- [ ] **Step 1: Read the current train.py to understand its shape**

Open [src/models/outcomes/train.py](src/models/outcomes/train.py) and read `train_model` carefully (around lines 340–620). Identify the boundary between data loading + arg parsing (which goes away) and the model-wiring core (which stays).

The core includes:
- Instantiate model class via `model_class(**model_kwargs)`
- Determine algorithm
- Call `model.prepare_features(...)` for each fold (filters rows for some models)
- `model.configure_model(algorithm, algorithm_params)` → `(estimator, param_grid)`
- Build `preprocessor_kwargs` (preserve_columns logic, embeddings flag, count features)
- `create_preprocessing_pipeline(model_type=..., model_name=algorithm, **kwargs)`
- `model.create_pipeline(estimator, preprocessor, algorithm, args)`
- Sample weights via `calculate_sample_weights(...)`
- `tune_model(...)` → fitted pipeline + best params
- `evaluate_model(train_pipeline, train_X, train_y, "training")` for train metrics
- For classification: `model.find_optimal_threshold(...)` if available
- `evaluate_model(train_pipeline, tune_X, tune_y, "tuning")` for tune metrics
- Refit final pipeline on train + tune (with sample weights if applicable)
- `evaluate_model(final_pipeline, test_X, test_y, "test")` for test metrics
- `model.compute_additional_metrics(test_y.values, test_pred, "test")` → merge into test metrics
- Predictions on tune and test sets

That entire pipeline body is what `train_one` keeps. Everything else — argparse, BQ data loading, year-range splits, `ExperimentTracker`, `log_experiment` — gets deleted.

- [ ] **Step 2: Write the failing test**

Create `tests/test_train_one.py`:

```python
"""Tests for train_one — the pure model-training function.

Hermetic: builds frames in-memory, calls train_one, asserts on returned artifacts.
"""

from pathlib import Path

import polars as pl
import pytest

from src.models.outcomes.train import train_one
from src.models.outcomes.hurdle import HurdleModel


def _synthetic_hurdle_frames():
    """Build minimal frames the hurdle pipeline can train on.

    Hurdle target: derived from `users_rated >= min_ratings_for_hurdle`.
    Need a few feature columns the preprocessor expects. Bare minimum is
    year_published + a numeric column or two, depending on what the
    BGG preprocessor pipeline tolerates with no additional features.
    """
    n_train, n_tune, n_test = 60, 20, 20
    train = pl.DataFrame({
        "game_id": list(range(1, n_train + 1)),
        "year_published": [2018] * n_train,
        "users_rated": [50] * (n_train // 2) + [10] * (n_train - n_train // 2),
        "num_weights": [5] * n_train,
        "complexity": [2.5] * n_train,
        "rating": [7.0] * n_train,
        "min_players": [2] * n_train,
        "max_players": [4] * n_train,
        "playing_time": [60] * n_train,
        "name": [f"game_{i}" for i in range(n_train)],
    })
    tune = train.head(n_tune).with_columns(pl.lit(2019).alias("year_published"))
    test = train.head(n_test).with_columns(pl.lit(2020).alias("year_published"))
    return train, tune, test


def test_train_one_returns_expected_artifacts(monkeypatch):
    train_df, tune_df, test_df = _synthetic_hurdle_frames()

    candidate_config = {
        "name": "logistic-hurdle",
        "algorithm": "logistic",
        "use_embeddings": False,
        "use_sample_weights": False,
    }

    out = train_one(
        model_type="hurdle",
        candidate_config=candidate_config,
        train_df=train_df,
        tune_df=tune_df,
        test_df=test_df,
    )

    assert "pipeline" in out
    assert "metrics" in out
    assert "parameters" in out
    assert "tune_predictions" in out
    assert "test_predictions" in out
    assert set(out["metrics"].keys()) >= {"train", "tune", "test"}
```

Note: this test depends on the actual hurdle pipeline being able to train on a tiny synthetic frame. If the existing preprocessor rejects something, observe the failure in step 4 and add the necessary columns (e.g. designer/category features may be needed). DO NOT modify the production preprocessor to accommodate the test — adapt the fixture instead.

- [ ] **Step 3: Run test to verify failure**

Run: `uv run pytest tests/test_train_one.py -v`
Expected: ImportError on `train_one` (it doesn't exist yet).

- [ ] **Step 4: Refactor train.py**

Open [src/models/outcomes/train.py](src/models/outcomes/train.py) and:

1. **Delete `parse_arguments` entirely.**
2. **Delete `main()` entirely.**
3. **Delete `main_finalize()`** if it exists at the bottom of the file (Stage 5 will introduce a snapshot-aware replacement).
4. **Replace `train_model(model_class, args)` with `train_one(model_type, candidate_config, train_df, tune_df, test_df, ...)`.**
5. **Remove all imports that become unused after deletion** (`argparse`, `setup_logging`, `load_config`, `ExperimentTracker`, `log_experiment`, the data-loader imports for BQ).
6. Keep imports for: model registry (`get_model_class`, `MODEL_REGISTRY`, `register_model`, `_populate_registry`), `clone`, `pd`, `np`, the model classes' helpers, `create_preprocessing_pipeline`, `tune_model`, `evaluate_model`, `calculate_sample_weights`, `select_X_y`.

Write `train_one` like this. Make it the *only* training entry point in this file:

```python
"""Pure model training functions for outcome models.

This module exposes ``train_one``, a function that takes data frames and
a candidate-config dict, and returns the training artifacts. It does not
parse argv, load data, or write to disk — those are orchestration
concerns owned by ``src/pipeline/train.py``.

The module also hosts the model registry (``get_model_class``,
``MODEL_REGISTRY``, ``register_model``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone

from src.models.outcomes.base import TrainableModel
from src.models.outcomes.data import select_X_y
from src.models.training import (
    create_preprocessing_pipeline,
    tune_model,
    evaluate_model,
    calculate_sample_weights,
)


logger = logging.getLogger(__name__)


# Registry of available model classes
MODEL_REGISTRY: Dict[str, Type[TrainableModel]] = {}


def register_model(model_class: Type[TrainableModel]) -> Type[TrainableModel]:
    MODEL_REGISTRY[model_class.model_type] = model_class
    return model_class


def get_model_class(model_type: str) -> Type[TrainableModel]:
    if not MODEL_REGISTRY:
        _populate_registry()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]


def _populate_registry() -> None:
    from src.models.outcomes.hurdle import HurdleModel
    from src.models.outcomes.complexity import ComplexityModel
    from src.models.outcomes.rating import RatingModel
    from src.models.outcomes.users_rated import UsersRatedModel
    from src.models.outcomes.geek_rating import GeekRatingModel

    register_model(HurdleModel)
    register_model(ComplexityModel)
    register_model(RatingModel)
    register_model(UsersRatedModel)
    register_model(GeekRatingModel)


def train_one(
    model_type: str,
    candidate_config: Dict[str, Any],
    train_df: pl.DataFrame,
    tune_df: pl.DataFrame,
    test_df: pl.DataFrame,
    metric: Optional[str] = None,
    patience: int = 15,
    preprocessor_type: str = "auto",
) -> Dict[str, Any]:
    """Train one candidate on one (train, tune, test) triple.

    Inputs are frames (already loaded from snapshot+split, already joined
    with any upstream score columns). Output is a dict of artifacts:
    pipeline, metrics, parameters, tune_predictions, test_predictions,
    and (for classification) optimal_threshold.

    Args:
        model_type: Registered model type (e.g. "hurdle", "complexity").
        candidate_config: Candidate recipe dict (from ``config.yaml``'s
            ``models.{type}.candidates`` list, possibly with overrides).
            Recognized keys:

            - ``algorithm``: estimator name passed to ``model.configure_model``
            - ``use_embeddings``: include description embeddings in the preprocessor
            - ``use_sample_weights``: weight train rows during tuning + final fit
            - ``sample_weight_column``: column to weight by (default depends on model)
            - ``min_ratings``, ``mode``, ``include_predictions``: model-specific
            - ``preprocessor_kwargs``: extra kwargs forwarded to ``create_preprocessing_pipeline``
            - ``algorithm_params``: extra kwargs forwarded to ``model.configure_model``
            - ``include_count_features``: forwarded to preprocessor

        train_df/tune_df/test_df: Polars frames already containing target column.
        metric: Tuning metric override; defaults to log_loss (classification)
            or rmse (regression).
        patience: Early-stopping patience for ``tune_model``.
        preprocessor_type: "auto", "linear", or "tree".

    Returns:
        Dict with keys: pipeline, metrics (dict of train/tune/test sub-dicts),
        parameters, tune_predictions (pl.DataFrame), test_predictions (pl.DataFrame).
        For classification: also "optimal_threshold".
    """
    model_class = get_model_class(model_type)

    # Build model_kwargs from candidate config (model-specific knobs)
    model_kwargs: Dict[str, Any] = {}
    if "min_ratings" in candidate_config:
        model_kwargs["min_ratings"] = candidate_config["min_ratings"]
    if "mode" in candidate_config:
        model_kwargs["mode"] = candidate_config["mode"]
    if "include_predictions" in candidate_config:
        model_kwargs["include_predictions"] = candidate_config["include_predictions"]
    model = model_class(**model_kwargs)

    algorithm = candidate_config.get("algorithm")
    if algorithm is None:
        algorithm = "ridge" if model.model_task == "regression" else "lightgbm"

    logger.info(f"train_one: {model.model_type} / {algorithm}")

    # X / y
    train_X, train_y = select_X_y(train_df, model.target_column)
    tune_X, tune_y = select_X_y(tune_df, model.target_column)
    test_X, test_y = select_X_y(test_df, model.target_column)
    tune_X_original = tune_X.copy()

    # Allow models to prepare features (e.g. geek_rating's stacking)
    # The model's prepare_features signature historically took an `args`
    # namespace. Build a small SimpleNamespace-equivalent so we don't have
    # to change the model classes.
    from types import SimpleNamespace
    prep_args = SimpleNamespace(
        use_embeddings=bool(candidate_config.get("use_embeddings", False)),
        sub_model_experiments=candidate_config.get("sub_model_experiments", {}),
        mode=candidate_config.get("mode"),
        include_predictions=candidate_config.get("include_predictions", True),
    )
    train_X, train_y = model.prepare_features(train_X, train_y, "train", prep_args)
    tune_X, tune_y = model.prepare_features(tune_X, tune_y, "tune", prep_args)
    test_X, test_y = model.prepare_features(test_X, test_y, "test", prep_args)

    # Filter polars frames to match if prepare_features dropped rows
    if len(train_X) < len(train_df):
        train_df = train_df[train_X.index.tolist()]
    if len(tune_X) < len(tune_df):
        tune_df = tune_df[tune_X.index.tolist()]
    if len(test_X) < len(test_df):
        test_df = test_df[test_X.index.tolist()]

    # Configure model + estimator
    algorithm_params = candidate_config.get("algorithm_params", {}) or {}
    estimator, param_grid = model.configure_model(algorithm, algorithm_params)

    # Build preprocessor
    preserve_columns = ["year_published"]
    if model.data_config.requires_complexity_predictions:
        preserve_columns.append("predicted_complexity")
    if model_type == "geek_rating" and prep_args.mode == "direct":
        preserve_columns.append("predicted_complexity")
        if prep_args.include_predictions:
            preserve_columns.extend(["predicted_rating", "predicted_users_rated_log"])

    preprocessor_kwargs = dict(candidate_config.get("preprocessor_kwargs", {}) or {})
    preprocessor_kwargs.update(
        preserve_columns=preserve_columns,
        include_description_embeddings=prep_args.use_embeddings,
        include_count_features=bool(candidate_config.get("include_count_features", False)),
    )

    preprocessor = create_preprocessing_pipeline(
        model_type=preprocessor_type,
        model_name=algorithm,
        **preprocessor_kwargs,
    )
    pipeline = model.create_pipeline(estimator, preprocessor, algorithm, prep_args)

    # Sample weights
    sample_weights = None
    use_sample_weights = bool(candidate_config.get("use_sample_weights", False))
    weight_column = candidate_config.get("sample_weight_column")
    if use_sample_weights:
        if weight_column is None:
            weight_column = "num_weights" if model.model_type == "complexity" else "users_rated"
        sample_weights = calculate_sample_weights(train_df, weight_column=weight_column)

    # Tuning metric
    if metric is None:
        metric = "log_loss" if model.model_task == "classification" else "rmse"

    tuned_pipeline, best_params, _ = tune_model(
        pipeline=pipeline,
        train_X=train_X,
        train_y=train_y,
        tune_X=tune_X,
        tune_y=tune_y,
        param_grid=param_grid,
        metric=metric,
        patience=patience,
        sample_weights=sample_weights,
    )

    # Train-set metrics from a clone fit on train only
    train_pipeline = clone(tuned_pipeline).fit(train_X, train_y)
    train_metrics = evaluate_model(train_pipeline, train_X, train_y, "training")

    # Optional threshold optimization (classification only)
    optimal_threshold: Optional[float] = None
    if hasattr(model, "find_optimal_threshold") and model.model_task == "classification":
        tune_pred_proba = train_pipeline.predict_proba(tune_X)[:, 1]
        threshold_results = model.find_optimal_threshold(tune_y, tune_pred_proba)
        optimal_threshold = float(threshold_results["threshold"])

    tune_metrics = evaluate_model(train_pipeline, tune_X, tune_y, "tuning")

    # Refit on train + tune (matches existing behavior)
    if hasattr(model, "filter_for_refit"):
        tune_X_refit, tune_y_refit = model.filter_for_refit(tune_X, tune_y, tune_X_original)
    else:
        tune_X_refit, tune_y_refit = tune_X, tune_y

    X_combined = pd.concat([train_X, tune_X_refit])
    y_combined = pd.concat([train_y, tune_y_refit])

    if use_sample_weights:
        combined_weights = calculate_sample_weights(
            pl.concat([train_df, tune_df]),
            weight_column=weight_column,
        )
        final_pipeline = clone(tuned_pipeline).fit(
            X_combined, y_combined,
            model__sample_weight=np.asarray(combined_weights),
        )
    else:
        final_pipeline = clone(tuned_pipeline).fit(X_combined, y_combined)

    test_metrics = evaluate_model(final_pipeline, test_X, test_y, "test")
    test_pred = final_pipeline.predict(test_X)
    additional = model.compute_additional_metrics(test_y.values, test_pred, "test")
    test_metrics.update(additional)

    # Predictions frames (polars, suitable for SnapshotStorage.save_result)
    tune_preds = _build_predictions_frame(
        train_pipeline, tune_X, tune_y, tune_df, model.model_task,
    )
    test_preds = _build_predictions_frame(
        final_pipeline, test_X, test_y, test_df, model.model_task,
    )

    out: Dict[str, Any] = {
        "pipeline": final_pipeline,
        "metrics": {"train": train_metrics, "tune": tune_metrics, "test": test_metrics},
        "parameters": best_params,
        "tune_predictions": tune_preds,
        "test_predictions": test_preds,
    }
    if optimal_threshold is not None:
        out["optimal_threshold"] = optimal_threshold
    return out


def _build_predictions_frame(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    df: pl.DataFrame,
    model_task: str,
) -> pl.DataFrame:
    """Produce a polars frame matching df's rows + ``prediction``/``actual`` columns."""
    preds = pipeline.predict(X)
    out = df.clone().with_columns([
        pl.Series("prediction", preds),
        pl.Series("actual", y.values),
    ])
    if model_task == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                out = out.with_columns(pl.Series("predicted_proba", proba[:, 1]))
        except Exception:
            pass
    return out
```

- [ ] **Step 5: Run the new test**

Run: `uv run pytest tests/test_train_one.py -v`

Expected: PASS — but if the synthetic fixture is too thin for the preprocessor (e.g. it needs designer/category list-columns), the test will fail with a feature-shape error. Adapt the fixture: add list-typed columns matching what the preprocessor expects, or use a larger fixture sourced from the snapshot tests' helper. Do NOT modify the production preprocessor.

If the test passes, run the full suite to confirm nothing else broke:

Run: `uv run pytest tests/ -v`

Expected: all snapshot/storage/build tests still pass; no production-side test exists for the old `outcomes/train.py:main` (which is deleted).

- [ ] **Step 6: Commit**

```bash
git add src/models/outcomes/train.py tests/test_train_one.py
git commit -m "refactor: extract train_one as pure data-frame function in outcomes/train.py"
```

### Task 14: Rewrite src/pipeline/train.py as snapshot-aware orchestrator

**Files:**
- Modify: `src/pipeline/train.py` (currently a 12-line shim)
- Test: `tests/test_pipeline_train_snapshot.py`

The orchestrator owns: argv, snapshot/split loading, upstream cascade resolution, looping over splits, writing to `SnapshotStorage`. It calls `train_one` per (snapshot, split, candidate).

This task does single-split, no upstream. Multi-split and upstream join come in Task 15. Per-candidate finalize comes in Task 21.

- [ ] **Step 1: Write failing test**

Create `tests/test_pipeline_train_snapshot.py`:

```python
"""Smoke test for the snapshot-aware pipeline.train orchestrator.

Hurdle has no upstream dependencies — start there.
"""

from pathlib import Path

import polars as pl

from src.models.build_snapshot import build_snapshot
from src.models.build_split import build_split
from src.models.snapshot_storage import SnapshotStorage
from src.pipeline.train import train as run_pipeline_train


def _synthetic_universe(tmp_path: Path) -> tuple[Path, int]:
    base = tmp_path / "snaps"
    n = 200
    df = pl.DataFrame({
        "game_id": list(range(1, n + 1)),
        "year_published": [2018]*50 + [2019]*50 + [2020]*50 + [2021]*50,
        "users_rated": [50] * (n // 2) + [10] * (n - n // 2),
        "num_weights": [5] * n,
        "complexity": [2.5] * n,
        "rating": [7.0] * n,
        "min_players": [2] * n,
        "max_players": [4] * n,
        "playing_time": [60] * n,
        "name": [f"game_{i}" for i in range(n)],
    })
    src = tmp_path / "src.parquet"
    df.write_parquet(src)
    v = build_snapshot(local_data=src, base_dir=base, use_embeddings=False)
    build_split(
        snapshot_version=v, split_name="standard",
        train_through=2019, tune_start=2020, tune_through=2020,
        test_start=2021, test_through=2021,
        base_dir=base,
    )
    return base, v


def test_pipeline_train_writes_result_artifacts(tmp_path: Path) -> None:
    base, v = _synthetic_universe(tmp_path)

    candidate_config = {
        "name": "logistic-hurdle",
        "algorithm": "logistic",
        "use_embeddings": False,
        "use_sample_weights": False,
    }

    run_pipeline_train(
        snapshot_version=v,
        model_type="hurdle",
        candidate="logistic-hurdle",
        candidate_config=candidate_config,
        splits=["standard"],
        upstream={},
        base_dir=base,
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    result = storage.load_result("hurdle", "logistic-hurdle", 1, "standard")
    assert result is not None
    assert "pipeline" in result
    assert "metrics" in result
    assert "tune_predictions" in result
    assert "test_predictions" in result

    # Candidate-level config + registration written
    cfg = storage.load_candidate_config("hurdle", "logistic-hurdle", 1)
    assert cfg == candidate_config

    reg = storage.load_candidate_registration("hurdle", "logistic-hurdle", 1)
    assert reg["snapshot_version"] == v
    assert reg["candidate"] == "logistic-hurdle"
    assert reg["splits"] == ["standard"]
    assert reg["upstream_experiments"] == {}
```

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/test_pipeline_train_snapshot.py -v`
Expected: ImportError on `from src.pipeline.train import train`.

- [ ] **Step 3: Rewrite src/pipeline/train.py**

Replace the entire file content of [src/pipeline/train.py](src/pipeline/train.py) with:

```python
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
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_pipeline_train_snapshot.py -v`
Expected: PASS. If the synthetic data is rejected by the hurdle preprocessor, adapt the fixture (don't modify the preprocessor).

Then run the full suite: `uv run pytest tests/ -v`
Expected: everything still passing.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/train.py tests/test_pipeline_train_snapshot.py
git commit -m "feat: rewrite pipeline.train as snapshot-aware orchestrator"
```

### Task 15: Multi-split + upstream cascade in pipeline.train

**Files:**
- Test: `tests/test_pipeline_train_snapshot.py` (append)

`pipeline.train` already supports multi-split iteration (Task 14's loop) and upstream join (`_join_upstream`). This task adds tests that exercise both — together with a complexity-then-rating cascade.

- [ ] **Step 1: Append tests**

In `tests/test_pipeline_train_snapshot.py` add:

```python
def test_train_multi_split_with_upstream(tmp_path: Path) -> None:
    """Train complexity, then rating with complexity as upstream, on two splits."""
    base, v = _synthetic_universe(tmp_path)
    # Add a yoy_2020 split (train≤2018, tune=2019, test=2020)
    build_split(
        snapshot_version=v, split_name="yoy_2020",
        train_through=2018, tune_start=2019, tune_through=2019,
        test_start=2020, test_through=2020,
        base_dir=base,
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)

    # Train complexity on both splits — produces score.parquet for each
    complexity_cfg = {
        "name": "ard-complexity",
        "algorithm": "ridge",  # ARD can be slow; ridge is fine for tests
        "use_embeddings": False,
        "use_sample_weights": False,
    }
    # Hand-write score.parquet for each split since pipeline.score doesn't exist yet (Task 16).
    # This task validates the upstream-JOIN code path; Task 16 wires the actual scorer.
    run_pipeline_train(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_config=complexity_cfg,
        splits=["standard", "yoy_2020"], upstream={}, base_dir=base,
    )
    # Synthesize score.parquet for both splits manually so rating training has a column to join
    universe = storage.load_universe()
    for split_name in ["standard", "yoy_2020"]:
        score_df = universe.select(["game_id"]).with_columns(
            pl.lit(2.5).alias("predicted_complexity")
        )
        result_dir = storage.result_dir("complexity", "ard-complexity", 1, split_name)
        result_dir.mkdir(parents=True, exist_ok=True)
        preds_dir = result_dir / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        score_df.write_parquet(preds_dir / "score.parquet")

    # Train rating with complexity upstream
    rating_cfg = {
        "name": "ard-ridge-rating",
        "algorithm": "ridge",
        "use_embeddings": False,
        "use_sample_weights": False,
        "min_ratings": 0,  # synthetic data is too small for the default of 5
    }
    run_pipeline_train(
        snapshot_version=v, model_type="rating",
        candidate="ard-ridge-rating", candidate_config=rating_cfg,
        splits=["standard", "yoy_2020"],
        upstream={"complexity": "ard-complexity"},
        base_dir=base,
    )

    # Both splits got results
    standard_result = storage.load_result("rating", "ard-ridge-rating", 1, "standard")
    yoy_result = storage.load_result("rating", "ard-ridge-rating", 1, "yoy_2020")
    assert standard_result is not None and yoy_result is not None

    reg = storage.load_candidate_registration("rating", "ard-ridge-rating", 1)
    assert reg["upstream_experiments"] == {"complexity": "ard-complexity"}
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_pipeline_train_snapshot.py -v`

Expected: 2 passed (the original Task 14 test plus this new one). If the rating model rejects the synthetic data because of feature requirements specific to its preprocessor, adapt the fixture.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_train_snapshot.py
git commit -m "test: multi-split + upstream cascade in pipeline.train"
```

### Task 16: Rewrite src/pipeline/score.py for snapshot-tree scoring

**Files:**
- Modify: `src/pipeline/score.py` (currently calls into legacy code)
- Test: `tests/test_pipeline_score_snapshot.py`

Scoring writes `score.parquet` for a (candidate, split) — predictions on every row of the snapshot universe. Today's flow: complexity must be scored before rating training, because rating's training-time feature `predicted_complexity` lives in `score.parquet`.

For Stage 2/3, the simple approach: use the train-fold pipeline (the pipeline already saved to `results/{split_name}/pipeline.pkl`) to score every row of the universe. Stage 4 will replace this with K-fold OOF on train rows.

- [ ] **Step 1: Write failing test**

Create `tests/test_pipeline_score_snapshot.py`:

```python
"""Tests for pipeline.score writing score.parquet to the snapshot tree."""

from pathlib import Path

import polars as pl

from src.models.build_snapshot import build_snapshot
from src.models.build_split import build_split
from src.models.snapshot_storage import SnapshotStorage
from src.pipeline.train import train as run_pipeline_train
from src.pipeline.score import score as run_pipeline_score


def _synthetic_universe(tmp_path: Path) -> tuple[Path, int]:
    base = tmp_path / "snaps"
    n = 200
    df = pl.DataFrame({
        "game_id": list(range(1, n + 1)),
        "year_published": [2018]*50 + [2019]*50 + [2020]*50 + [2021]*50,
        "users_rated": [50] * (n // 2) + [10] * (n - n // 2),
        "num_weights": [5] * n,
        "complexity": [2.5] * n,
        "rating": [7.0] * n,
        "min_players": [2] * n,
        "max_players": [4] * n,
        "playing_time": [60] * n,
        "name": [f"game_{i}" for i in range(n)],
    })
    src = tmp_path / "src.parquet"
    df.write_parquet(src)
    v = build_snapshot(local_data=src, base_dir=base, use_embeddings=False)
    build_split(
        snapshot_version=v, split_name="standard",
        train_through=2019, tune_start=2020, tune_through=2020,
        test_start=2021, test_through=2021,
        base_dir=base,
    )
    return base, v


def test_pipeline_score_writes_score_parquet(tmp_path: Path) -> None:
    base, v = _synthetic_universe(tmp_path)
    storage = SnapshotStorage(snapshot_version=v, base_dir=base)

    # Train complexity first
    complexity_cfg = {
        "name": "ard-complexity", "algorithm": "ridge",
        "use_embeddings": False, "use_sample_weights": False,
    }
    run_pipeline_train(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_config=complexity_cfg,
        splits=["standard"], upstream={}, base_dir=base,
    )

    # Score
    run_pipeline_score(
        snapshot_version=v,
        model_type="complexity",
        candidate="ard-complexity",
        candidate_version=1,
        splits=["standard"],
        upstream={},
        base_dir=base,
    )

    score = storage.load_score_predictions("complexity", "ard-complexity", 1, "standard")
    assert score is not None
    assert score.height == 200  # full universe
    assert "game_id" in score.columns
    assert "predicted_complexity" in score.columns
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_pipeline_score_snapshot.py -v`
Expected: ImportError on `from src.pipeline.score import score`.

- [ ] **Step 3: Replace src/pipeline/score.py**

Read what's currently in [src/pipeline/score.py](src/pipeline/score.py) — likely a thin shim like train.py was. Replace its entire contents with:

```python
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

    upstream: Dict[str, str] = {}
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
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_pipeline_score_snapshot.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/score.py tests/test_pipeline_score_snapshot.py
git commit -m "feat: rewrite pipeline.score for snapshot tree"
```

### Task 17: Cross-split summary writer

**Files:**
- Modify: `src/pipeline/train.py`
- Test: `tests/test_pipeline_train_snapshot.py` (append)

After `pipeline.train` finishes all splits for a candidate, write a candidate-level `summary.json` rolling up per-split metrics. Useful for cross-split comparison without having to walk the result tree.

- [ ] **Step 1: Append failing test**

In `tests/test_pipeline_train_snapshot.py` add:

```python
def test_summary_json_written_after_multi_split_training(tmp_path: Path) -> None:
    import json
    base, v = _synthetic_universe(tmp_path)
    build_split(
        snapshot_version=v, split_name="yoy_2020",
        train_through=2018, tune_start=2019, tune_through=2019,
        test_start=2020, test_through=2020,
        base_dir=base,
    )

    cfg = {
        "name": "logistic-hurdle", "algorithm": "logistic",
        "use_embeddings": False, "use_sample_weights": False,
    }
    run_pipeline_train(
        snapshot_version=v, model_type="hurdle",
        candidate="logistic-hurdle", candidate_config=cfg,
        splits=["standard", "yoy_2020"], upstream={}, base_dir=base,
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    summary_path = storage.experiment_dir("hurdle", "logistic-hurdle", 1) / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert sorted(summary["per_split"].keys()) == ["standard", "yoy_2020"]
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_pipeline_train_snapshot.py::test_summary_json_written_after_multi_split_training -v`
Expected: FAIL — `summary.json` does not exist.

- [ ] **Step 3: Add summary writer to pipeline.train**

In [src/pipeline/train.py](src/pipeline/train.py), modify the `train()` function. Find the end of the per-split loop:

```python
        logger.info(f"Wrote result {model_type}/{candidate}/v{candidate_version}/{split_name}")

    return candidate_version
```

Replace with:

```python
        logger.info(f"Wrote result {model_type}/{candidate}/v{candidate_version}/{split_name}")

    _write_summary(storage, model_type, candidate, candidate_version, splits)
    return candidate_version


def _write_summary(
    storage: SnapshotStorage,
    model_type: str,
    candidate: str,
    version: int,
    splits: List[str],
) -> Path:
    import json as _json
    per_split: Dict[str, Any] = {}
    for split_name in splits:
        result = storage.load_result(model_type, candidate, version, split_name)
        if result is None:
            continue
        per_split[split_name] = result["metrics"]
    summary = {
        "model_type": model_type,
        "candidate": candidate,
        "version": version,
        "snapshot_version": storage.snapshot_version,
        "splits": splits,
        "per_split": per_split,
        "created_at": datetime.now().isoformat(),
    }
    path = storage.experiment_dir(model_type, candidate, version) / "summary.json"
    path.write_text(_json.dumps(summary, indent=2, default=str))
    return path
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/test_pipeline_train_snapshot.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/train.py tests/test_pipeline_train_snapshot.py
git commit -m "feat: write summary.json rolling up cross-split metrics"
```

### Task 18: Manual end-to-end chain

**Files:**
- None (manual verification on real data)

This step is not a code change — it confirms the new flow works end-to-end against the real BQ data and produces metrics in the same ballpark as today's legacy training.

- [ ] **Step 1: Build a real snapshot from BigQuery**

```bash
uv run python -m src.models.build_snapshot --use-embeddings
```

Expected: prints `snapshot_version: 1` (or higher if you've previously experimented). Creates `models/experiments/_snapshots/v{N}/universe.parquet`.

- [ ] **Step 2: Build splits**

```bash
uv run python -m src.models.build_split --snapshot-version 1 --split-name standard
uv run python -m src.models.build_split --snapshot-version 1 --yoy --yoy-start 2018 --yoy-end 2024
```

- [ ] **Step 3: Train the chain on the standard split first**

```bash
uv run python -m src.pipeline.train --model hurdle --candidate logistic-hurdle \
  --snapshot-version 1 --splits standard

uv run python -m src.pipeline.train --model complexity --candidate ard-complexity \
  --snapshot-version 1 --splits standard

uv run python -m src.pipeline.score --model complexity --candidate ard-complexity \
  --snapshot-version 1 --splits standard

uv run python -m src.pipeline.train --model rating --candidate ard-ridge-rating \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity

uv run python -m src.pipeline.train --model users_rated --candidate ard-ridge-users_rated \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity

uv run python -m src.pipeline.score --model rating --candidate ard-ridge-rating \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity

uv run python -m src.pipeline.score --model users_rated --candidate ard-ridge-users_rated \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity

uv run python -m src.pipeline.train --model geek_rating --candidate ard-geek_rating \
  --snapshot-version 1 --splits standard \
  --upstream complexity=ard-complexity,rating=ard-ridge-rating,users_rated=ard-ridge-users_rated
```

- [ ] **Step 4: Sanity-check metrics**

```bash
for m in hurdle complexity rating users_rated geek_rating; do
  echo "=== $m ==="
  cat models/experiments/_snapshots/v1/experiments/$m/*/v1/results/standard/metrics.json | head -30
done
```

Expected: each model's test metrics should be in the same ballpark as the legacy `models/experiments/{model_type}/...` results. Big regressions = a real bug; investigate before continuing.

- [ ] **Step 5: Repeat across YoY splits (optional but recommended)**

```bash
uv run python -m src.pipeline.train --model hurdle --candidate logistic-hurdle \
  --snapshot-version 1 --splits yoy_2018,yoy_2019,yoy_2020,yoy_2021,yoy_2022,yoy_2023,yoy_2024
# ... and similarly for the rest of the chain
```

`summary.json` should now contain per-split metrics for the headline + all YoY splits.

---

## Stage 4: K-fold OOF Scoring for Upstream Models

End state: `score.parquet` for upstream models (complexity, rating, users_rated) uses K-fold OOF predictions for the train rows. Downstream models train on honest features.

### Task 19: K-fold OOF predictor utility

**Files:**
- Create: `src/models/oof.py`
- Test: `tests/test_oof_scoring.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_oof_scoring.py`:

```python
"""Tests for K-fold OOF prediction utility."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.oof import kfold_oof_predict


def _make_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])


def test_oof_predictions_have_same_length_as_input():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    y = pd.Series((rng.normal(size=100) > 0).astype(int))

    preds = kfold_oof_predict(_make_pipeline(), X, y, k=5, seed=42)
    assert preds.shape == (100,)


def test_oof_is_deterministic_with_seed():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))
    y = pd.Series((rng.normal(size=50) > 0).astype(int))

    p1 = kfold_oof_predict(_make_pipeline(), X, y, k=5, seed=7)
    p2 = kfold_oof_predict(_make_pipeline(), X, y, k=5, seed=7)
    np.testing.assert_array_equal(p1, p2)


def test_oof_differs_with_different_seed():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))
    y = pd.Series((rng.normal(size=50) > 0).astype(int))

    p1 = kfold_oof_predict(_make_pipeline(), X, y, k=5, seed=1)
    p2 = kfold_oof_predict(_make_pipeline(), X, y, k=5, seed=2)
    assert not np.array_equal(p1, p2)


def test_oof_with_proba_returns_class_one_proba():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=list("abc"))
    y = pd.Series((rng.normal(size=80) > 0).astype(int))

    preds = kfold_oof_predict(_make_pipeline(), X, y, k=4, seed=0, predict_proba=True)
    assert preds.shape == (80,)
    assert preds.min() >= 0 and preds.max() <= 1
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_oof_scoring.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement oof.py**

Create `src/models/oof.py`:

```python
"""K-fold out-of-fold prediction utility.

Given an unfitted pipeline and a (X, y) frame, produce predictions for
every row of X using only models that did not see that row. Used to
generate honest training-time features for downstream cascaded models
(complexity → rating, etc).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


def kfold_oof_predict(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5,
    seed: int = 42,
    predict_proba: bool = False,
) -> np.ndarray:
    """Return out-of-fold predictions for every row of X.

    Args:
        pipeline: Unfitted sklearn pipeline. Cloned per fold.
        X: Feature frame.
        y: Target series.
        k: Number of folds.
        seed: Random seed for fold assignment.
        predict_proba: If True, return probability of the positive class
            (binary classification). Default False.
    """
    n = len(X)
    out = np.zeros(n, dtype=float)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        if predict_proba:
            proba = fold_pipeline.predict_proba(X_val)
            out[val_idx] = proba[:, 1]
        else:
            out[val_idx] = fold_pipeline.predict(X_val)
    return out
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/test_oof_scoring.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/oof.py tests/test_oof_scoring.py
git commit -m "feat: K-fold OOF prediction utility"
```

### Task 20: Wire OOF into pipeline.score for upstream models

**Files:**
- Modify: `src/pipeline/score.py`
- Test: `tests/test_pipeline_score_snapshot.py`

For upstream model types (complexity, rating, users_rated), `score.parquet` for train rows should be OOF rather than in-sample. Tune/test rows are already OOF in the sense that the train-fold pipeline never saw them. Rows outside train+tune+test continue to be predicted by the train-fold pipeline (Stage 5 swaps in the finalized model for those).

- [ ] **Step 1: Append failing test**

In `tests/test_pipeline_score_snapshot.py` add:

```python
def test_score_uses_oof_for_upstream_train_rows(tmp_path: Path, monkeypatch) -> None:
    """When scoring an upstream model, kfold_oof_predict is invoked for the train rows."""
    base, v = _synthetic_universe(tmp_path)

    # Train complexity with a small k so the test is fast
    cfg = {
        "name": "ard-complexity", "algorithm": "ridge",
        "use_embeddings": False, "use_sample_weights": False,
        "oof_folds": 3,
    }
    run_pipeline_train(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_config=cfg,
        splits=["standard"], upstream={}, base_dir=base,
    )

    calls = []
    from src.models import oof as _oof
    real = _oof.kfold_oof_predict

    def spy(*args, **kwargs):
        calls.append(kwargs.get("k"))
        return real(*args, **kwargs)

    monkeypatch.setattr(_oof, "kfold_oof_predict", spy)

    run_pipeline_score(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_version=1,
        splits=["standard"], upstream={}, base_dir=base,
    )

    assert calls, "kfold_oof_predict was not called"
    assert calls[0] == 3
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_pipeline_score_snapshot.py::test_score_uses_oof_for_upstream_train_rows -v`
Expected: FAIL — assertion fails because kfold_oof_predict was not called.

- [ ] **Step 3: Implement OOF in score**

In [src/pipeline/score.py](src/pipeline/score.py), modify the per-split loop in `score()`. Find this block:

```python
        X = scoring_universe.to_pandas()
        preds = pipeline.predict(X)
        score_df = scoring_universe.select(["game_id"]).clone().with_columns(
            pl.Series(pred_col, preds)
        )
```

Replace with:

```python
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
```

Then add this helper function at the bottom of `src/pipeline/score.py` (before `main`):

```python
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
    from src.models.oof import kfold_oof_predict
    from src.models.outcomes.train import get_model_class
    from src.models.outcomes.data import select_X_y

    cfg = storage.load_candidate_config(model_type, candidate, candidate_version) or {}
    oof_folds = int(cfg.get("oof_folds", 5))

    # Re-construct an unfitted pipeline equivalent to the trained one.
    # The simplest route: clone the trained pipeline (clears fitted state).
    from sklearn.base import clone
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

    oof_preds = kfold_oof_predict(
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_pipeline_score_snapshot.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/score.py tests/test_pipeline_score_snapshot.py
git commit -m "feat: K-fold OOF on train rows in pipeline.score for upstream models"
```

### Task 21: Manual OOF verification

**Files:**
- None

- [ ] **Step 1: Re-run complexity scoring on the real snapshot**

```bash
# Bump candidate version by re-training (or delete and rebuild — but training is cheap enough)
uv run python -m src.pipeline.train --model complexity --candidate ard-complexity \
  --snapshot-version 1 --splits standard
uv run python -m src.pipeline.score --model complexity --candidate ard-complexity \
  --snapshot-version 1 --splits standard
```

- [ ] **Step 2: Inspect score.parquet train rows**

```bash
uv run python -c "
import polars as pl
from src.models.snapshot_storage import SnapshotStorage

s = SnapshotStorage(1)
score = s.load_score_predictions('complexity', 'ard-complexity', s.list_candidate_versions('complexity', 'ard-complexity')[-1], 'standard')
split = s.load_split('standard')
joined = score.join(split['train'].select(['game_id', 'complexity']), on='game_id', how='inner')
print('train rows in score.parquet:', joined.height)
print(joined.head(10))
print('predicted_complexity range:', joined.select('predicted_complexity').min(), joined.select('predicted_complexity').max())
"
```

Expected: predicted_complexity for train rows should look reasonable (in [1, 5] range, varying), and should differ from a hypothetical in-sample prediction. The variation across rows is the visible sign that OOF is producing real held-out predictions.

- [ ] **Step 3: Re-run rating with the new OOF complexity**

```bash
uv run python -m src.pipeline.train --model rating --candidate ard-ridge-rating \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity
```

The rating model's tune RMSE may rise compared to the in-sample-feature run — that's expected and correct. The model is no longer learning from a leaked feature.

---

## Stage 5: Finalize

End state: `pipeline.finalize` produces a candidate-level `finalized.pkl`, refit on the full snapshot universe through some `final_end_year`. This is the deployable artifact for operational scoring.

### Task 22: Rewrite src/pipeline/finalize.py for snapshot tree

**Files:**
- Modify: `src/pipeline/finalize.py`
- Test: `tests/test_pipeline_finalize_snapshot.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_pipeline_finalize_snapshot.py`:

```python
"""Tests for pipeline.finalize writing finalized.pkl at the candidate level."""

from pathlib import Path

import polars as pl

from src.models.build_snapshot import build_snapshot
from src.models.build_split import build_split
from src.models.snapshot_storage import SnapshotStorage
from src.pipeline.train import train as run_pipeline_train
from src.pipeline.finalize import finalize as run_pipeline_finalize


def _synthetic_universe(tmp_path: Path) -> tuple[Path, int]:
    base = tmp_path / "snaps"
    n = 200
    df = pl.DataFrame({
        "game_id": list(range(1, n + 1)),
        "year_published": [2018]*50 + [2019]*50 + [2020]*50 + [2021]*50,
        "users_rated": [50] * (n // 2) + [10] * (n - n // 2),
        "num_weights": [5] * n,
        "complexity": [2.5] * n,
        "rating": [7.0] * n,
        "min_players": [2] * n,
        "max_players": [4] * n,
        "playing_time": [60] * n,
        "name": [f"game_{i}" for i in range(n)],
    })
    src = tmp_path / "src.parquet"
    df.write_parquet(src)
    v = build_snapshot(local_data=src, base_dir=base, use_embeddings=False)
    build_split(
        snapshot_version=v, split_name="standard",
        train_through=2019, tune_start=2020, tune_through=2020,
        test_start=2021, test_through=2021,
        base_dir=base,
    )
    return base, v


def test_finalize_writes_candidate_level_pipeline(tmp_path: Path) -> None:
    base, v = _synthetic_universe(tmp_path)
    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    cfg = {
        "name": "ard-complexity", "algorithm": "ridge",
        "use_embeddings": False, "use_sample_weights": False,
    }

    run_pipeline_train(
        snapshot_version=v, model_type="complexity",
        candidate="ard-complexity", candidate_config=cfg,
        splits=["standard"], upstream={}, base_dir=base,
    )

    run_pipeline_finalize(
        snapshot_version=v,
        model_type="complexity",
        candidate="ard-complexity",
        candidate_version=1,
        finalize_through=2021,
        base_dir=base,
    )

    finalized = storage.load_finalized_pipeline("complexity", "ard-complexity", 1)
    assert finalized is not None

    reg = storage.load_candidate_registration("complexity", "ard-complexity", 1)
    assert reg["finalize_through"] == 2021
    assert "finalized_at" in reg
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_pipeline_finalize_snapshot.py -v`
Expected: ImportError.

- [ ] **Step 3: Rewrite src/pipeline/finalize.py**

Replace the entire content of [src/pipeline/finalize.py](src/pipeline/finalize.py) with:

```python
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
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/test_pipeline_finalize_snapshot.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/finalize.py tests/test_pipeline_finalize_snapshot.py
git commit -m "feat: rewrite pipeline.finalize for snapshot tree (candidate-level finalized.pkl)"
```

### Task 23: Self-review checklist (manual)

**Files:**
- None (review pass)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: all passing.

- [ ] **Step 2: Run lint**

Run: `make lint`
Expected: no errors.

- [ ] **Step 3: End-to-end manual chain on real data**

```bash
# If a v1 snapshot doesn't already exist, build one:
uv run python -m src.models.build_snapshot --use-embeddings

uv run python -m src.models.build_split --snapshot-version 1 --split-name standard
uv run python -m src.models.build_split --snapshot-version 1 --yoy --yoy-start 2018 --yoy-end 2024

# Train + score the cascade. Each upstream model needs to be scored
# before the next layer trains.
uv run python -m src.pipeline.train --model hurdle --candidate logistic-hurdle \
  --snapshot-version 1 --splits standard
uv run python -m src.pipeline.train --model complexity --candidate ard-complexity \
  --snapshot-version 1 --splits standard
uv run python -m src.pipeline.score --model complexity --candidate ard-complexity \
  --snapshot-version 1 --splits standard

uv run python -m src.pipeline.train --model rating --candidate ard-ridge-rating \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity
uv run python -m src.pipeline.train --model users_rated --candidate ard-ridge-users_rated \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity
uv run python -m src.pipeline.score --model rating --candidate ard-ridge-rating \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity
uv run python -m src.pipeline.score --model users_rated --candidate ard-ridge-users_rated \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity

uv run python -m src.pipeline.train --model geek_rating --candidate ard-geek_rating \
  --snapshot-version 1 --splits standard \
  --upstream complexity=ard-complexity,rating=ard-ridge-rating,users_rated=ard-ridge-users_rated

# Finalize each candidate
for m in hurdle complexity rating users_rated geek_rating; do
  cand=$(uv run python -c "from src.models.candidate_config import list_candidates; print(list_candidates('$m')[0])")
  uv run python -m src.pipeline.finalize --model $m --candidate $cand \
    --snapshot-version 1 --finalize-through 2024
done
```

- [ ] **Step 4: Inspect the tree**

```bash
tree -L 5 models/experiments/_snapshots/v1/ | head -60
```

Expected: matches the layout in the spec.

- [ ] **Step 5: Sanity-check metrics**

Compare metrics from the new layout's `summary.json` files against the existing `models/experiments/{model_type}/...` legacy results for the same train/tune/test years. They should be in the same ballpark. Big differences indicate either: a real bug, or a feature column the new code path is missing — investigate.

---

## Done criteria

- All Stages 1–5 tasks complete and committed.
- Full test suite passes.
- The end-to-end manual chain in Task 23 produces a `_snapshots/v1/` tree matching the spec layout.
- `make lint` clean.
- Branch ready for merge — but merge happens out of band, when you've validated the redesign against current production behavior.

## Self-review notes (writer)

- Spec covers: snapshot storage, split storage, in-snapshot training, cascading dependencies, K-fold OOF, finalize-at-candidate-level, candidates-in-config — all have tasks.
- Out of scope per spec (streamlit/Quarto adaptation, GCS sync, legacy migration, snapshot diffing) — correctly omitted from plan.
- Architectural shape: `outcomes/train.py` becomes the pure model layer (`train_one(model, candidate_config, frames) -> artifacts`); `src/pipeline/{train,score,finalize}.py` becomes the snapshot-aware orchestration layer. The Makefile targets (`make hurdle`, `make complexity`, etc.) keep pointing at `src.pipeline.*` and continue to work; only their CLI args change.
- The legacy `outcomes/train.py:main`, `parse_arguments`, `main_finalize` are deleted on this branch. Production behavior continues on `main` until merge.
