# BGG Model Snapshots and Splits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorient bgg-rating-models training around a versioned data snapshot with named/versioned splits underneath, so experiments are honestly comparable on disk-location alone. Cascading dependencies (rating reads predicted_complexity) resolve within a fixed (snapshot, split) surface. K-fold OOF scoring for upstream models eliminates training-time leakage in downstream features.

**Architecture:** A single `SnapshotStorage` class owns paths and I/O for the new tree at `models/experiments/_snapshots/v{N}/`. Three new CLIs (`build_snapshot`, `build_split`, plus a snapshot-aware refactor of `outcomes/train.py`) drive the workflow. The existing `Experiment` artifact-logging code is reused; only path resolution changes. OOF is layered on after the basic flow works.

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

- `src/models/outcomes/data.py` — replace `create_data_splits` inline-year-loading with `load_split_from_storage(snapshot_version, split_name)`. `load_training_data` for new code paths reads from snapshot, not BQ.
- `src/models/outcomes/train.py` — replace year CLI flags with `--snapshot-version`, `--candidate`, `--splits`, `--upstream`. Iterate over splits, train per split, write to snapshot tree. Per-candidate finalize at end.
- `src/models/outcomes/finalize.py` — adapt to refit on full snapshot universe and write `finalized.pkl` at the candidate level.
- `src/models/experiments.py` — `ExperimentTracker` gains a `snapshot_storage` mode where `base_dir` is resolved per-split as `_snapshots/v{N}/experiments/{model_type}/{candidate}/v{M}/results/{split_name}/`. Existing `Experiment.log_*` methods unchanged.
- `config.yaml` — add `candidates` list per model type, mirroring `collections.candidates`.

**Test files (new):**

- `tests/test_snapshot_storage.py` — hermetic, tmp_path-based, mirrors `test_collection_artifact_storage_local.py`.
- `tests/test_build_snapshot.py` — uses local parquet input, asserts on file output.
- `tests/test_build_split.py` — round-trip a synthetic snapshot, verify split contents.
- `tests/test_outcomes_train_snapshot.py` — end-to-end training of a tiny synthetic candidate, asserts result paths and `score.parquet` shape.
- `tests/test_oof_scoring.py` — added in Stage 2, validates K-fold OOF behavior.

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

### Task 13: New training entry point — single split, no upstream

**Files:**
- Create: `src/models/train_snapshot.py`
- Test: `tests/test_train_snapshot.py`

This task introduces a new orchestration function that uses snapshot storage and trains one candidate on one split. It reuses the existing `train_model` machinery in `src/models/outcomes/train.py` for the actual training, but bypasses the year-based CLI.

- [ ] **Step 1: Write a smoke test (kept narrow)**

Create `tests/test_train_snapshot.py`:

```python
"""Smoke test for the snapshot-aware single-split training flow.

Uses the hurdle model class because it has no upstream dependencies.
Synthetic data; tests structural correctness, not model quality.
"""

from pathlib import Path

import polars as pl
import pytest

from src.models.build_snapshot import build_snapshot
from src.models.build_split import build_split
from src.models.snapshot_storage import SnapshotStorage
from src.models.train_snapshot import train_candidate


@pytest.fixture
def synthetic_snapshot(tmp_path: Path) -> tuple[Path, int]:
    base = tmp_path / "snaps"
    df = pl.DataFrame({
        "game_id": list(range(1, 101)),
        "year_published": [2018]*25 + [2019]*25 + [2020]*25 + [2021]*25,
        "users_rated": [50] * 100,
        "num_weights": [5] * 100,
        "complexity": [2.5] * 100,
        "rating": [7.0] * 100,
        # Hurdle target is derived; here we just need columns the loader sees.
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


def test_train_candidate_writes_result_artifacts(synthetic_snapshot, tmp_path):
    base, v = synthetic_snapshot

    # Use a test-only candidate config that doesn't need real BGG features
    candidate_config = {
        "name": "logistic-hurdle",
        "algorithm": "logistic",
        "use_embeddings": False,
    }

    train_candidate(
        snapshot_version=v,
        model_type="hurdle",
        candidate="logistic-hurdle",
        candidate_config=candidate_config,
        splits=["standard"],
        base_dir=base,
        upstream={},
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    result = storage.load_result("hurdle", "logistic-hurdle", 1, "standard")
    assert result is not None
    assert "pipeline" in result
    assert "metrics" in result
    assert "tune_predictions" in result
    assert "test_predictions" in result
    # Score predictions = predictions on the full universe
    assert "score_predictions" in result
    assert result["score_predictions"].height == 100  # full universe
```

Note: This test depends on the actual `HurdleModel` and preprocessing pipeline being able to handle a small synthetic dataset. If the existing pipeline rejects rows with too few features, this fixture will need extra columns. Adjust by inspecting the failure in step 2.

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/test_train_snapshot.py -v`
Expected: ImportError on `train_candidate`.

- [ ] **Step 3: Implement train_snapshot.py with the train_candidate function**

Create `src/models/train_snapshot.py`:

```python
"""Snapshot-aware training entry point.

Trains one candidate on one or more splits within a snapshot. Uses the
existing model classes from ``src/models/outcomes/`` and the existing
preprocessing/tuning pipeline from ``src/models/training.py``. The
difference is path resolution: artifacts go under
``_snapshots/v{N}/experiments/{model_type}/{candidate}/v{M}/``.

Usage::

    uv run python -m src.models.train_snapshot \\
        --model rating --candidate ard-ridge-rating \\
        --snapshot-version 1 --splits standard,yoy_2018 \\
        --upstream complexity=ard-complexity
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import polars as pl

from src.models.candidate_config import find_candidate
from src.models.outcomes.data import select_X_y
from src.models.outcomes.train import get_model_class
from src.models.snapshot_storage import DEFAULT_BASE_DIR, SnapshotStorage
from src.models.training import (
    create_preprocessing_pipeline,
    tune_model,
    evaluate_model,
)
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def train_candidate(
    snapshot_version: int,
    model_type: str,
    candidate: str,
    candidate_config: Dict[str, Any],
    splits: List[str],
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
    upstream: Optional[Dict[str, str]] = None,
) -> int:
    """Train one candidate on one or more splits. Returns the assigned candidate version."""
    upstream = upstream or {}
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)
    universe = storage.load_universe()
    if universe is None:
        raise FileNotFoundError(f"No snapshot v{snapshot_version}")

    candidate_version = storage.next_candidate_version(model_type, candidate)

    # Persist candidate-level artifacts (config + registration)
    storage.save_candidate_config(model_type, candidate, candidate_version, candidate_config)
    registration = {
        "snapshot_version": snapshot_version,
        "model_type": model_type,
        "candidate": candidate,
        "version": candidate_version,
        "created_at": datetime.now().isoformat(),
        "upstream_experiments": upstream,
        "splits": splits,
    }
    storage.save_candidate_registration(model_type, candidate, candidate_version, registration)

    # Look up the model class
    model_class = get_model_class(model_type)
    target_column = model_class().target_column

    for split_name in splits:
        logger.info(f"Training {model_type}/{candidate}/v{candidate_version} on split {split_name}")
        split = storage.load_split(split_name)
        if split is None:
            raise FileNotFoundError(f"Split {split_name!r} not found in v{snapshot_version}")

        train_df = split["train"]
        tune_df = split["tune"]
        test_df = split["test"]

        # Apply upstream score columns by joining (no-op if upstream is empty)
        train_df, tune_df, test_df, universe_with_upstream = _join_upstream(
            storage, snapshot_version, split_name, upstream,
            train_df, tune_df, test_df, universe,
        )

        # X/y extraction
        train_X, train_y = select_X_y(train_df, target_column)
        tune_X, tune_y = select_X_y(tune_df, target_column)
        test_X, test_y = select_X_y(test_df, target_column)

        # Build preprocessor + model (existing code path)
        algorithm = candidate_config.get("algorithm")
        preprocessor = create_preprocessing_pipeline(
            X=train_X, model_type=model_type, algorithm=algorithm,
        )
        # Tune (returns fitted pipeline + best params + per-set metrics)
        tuned = tune_model(
            preprocessor=preprocessor,
            model_type=model_type,
            algorithm=algorithm,
            train_X=train_X, train_y=train_y,
            tune_X=tune_X, tune_y=tune_y,
        )
        pipeline = tuned["pipeline"]
        best_params = tuned["best_params"]
        train_metrics = tuned.get("train_metrics", {})
        tune_metrics = tuned.get("tune_metrics", {})
        test_metrics = evaluate_model(pipeline, test_X, test_y, model_type=model_type)

        metrics = {"train": train_metrics, "tune": tune_metrics, "test": test_metrics}

        # Predictions
        tune_preds = _predict_to_df(pipeline, tune_X, tune_y, tune_df, model_type)
        test_preds = _predict_to_df(pipeline, test_X, test_y, test_df, model_type)

        # Score predictions: in-sample for now (Stage 3 replaces with OOF for upstream models)
        universe_X = universe_with_upstream.drop(target_column).to_pandas() \
            if target_column in universe_with_upstream.columns else universe_with_upstream.to_pandas()
        score_preds = _score_universe(pipeline, universe_with_upstream, model_type)

        storage.save_result(
            model_type=model_type,
            candidate=candidate,
            version=candidate_version,
            split_name=split_name,
            pipeline=pipeline,
            metrics=metrics,
            parameters=best_params,
            tune_predictions=tune_preds,
            test_predictions=test_preds,
            score_predictions=score_preds,
        )
        logger.info(f"Wrote result for {model_type}/{candidate}/v{candidate_version}/{split_name}")

    return candidate_version


def _join_upstream(
    storage: SnapshotStorage,
    snapshot_version: int,
    split_name: str,
    upstream: Dict[str, str],
    train_df: pl.DataFrame,
    tune_df: pl.DataFrame,
    test_df: pl.DataFrame,
    universe: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Join upstream score predictions onto each frame.

    For each upstream {model_type: candidate}, look up that candidate's latest
    version's score.parquet for the same split, and left-join on game_id.
    """
    for upstream_type, upstream_candidate in upstream.items():
        versions = storage.list_candidate_versions(upstream_type, upstream_candidate)
        if not versions:
            raise FileNotFoundError(
                f"Upstream {upstream_type}/{upstream_candidate} has no runs in v{snapshot_version}"
            )
        upstream_v = versions[-1]
        score = storage.load_score_predictions(
            upstream_type, upstream_candidate, upstream_v, split_name,
        )
        if score is None:
            raise FileNotFoundError(
                f"Upstream {upstream_type}/{upstream_candidate}/v{upstream_v} "
                f"has no score.parquet for split {split_name!r}"
            )
        # Strip duplicate columns other than game_id
        join_cols = [c for c in score.columns if c == "game_id" or c not in train_df.columns]
        score = score.select(join_cols)
        train_df = train_df.join(score, on="game_id", how="left")
        tune_df = tune_df.join(score, on="game_id", how="left")
        test_df = test_df.join(score, on="game_id", how="left")
        universe = universe.join(score, on="game_id", how="left")
    return train_df, tune_df, test_df, universe


def _predict_to_df(
    pipeline, X: pd.DataFrame, y: pd.Series, df: pl.DataFrame, model_type: str,
) -> pl.DataFrame:
    preds = pipeline.predict(X)
    out = df.clone().with_columns([
        pl.Series("prediction", preds),
        pl.Series("actual", y.values),
    ])
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                out = out.with_columns(pl.Series("predicted_proba", proba[:, 1]))
        except Exception:
            pass
    return out


def _score_universe(pipeline, universe: pl.DataFrame, model_type: str) -> pl.DataFrame:
    """In-sample score on every row of the universe. Stage 3 will replace this
    with K-fold OOF for upstream models."""
    universe_X = universe.to_pandas()
    preds = pipeline.predict(universe_X)
    out = universe.select(["game_id"]).clone()
    pred_col = {
        "complexity": "predicted_complexity",
        "rating": "predicted_rating",
        "users_rated": "predicted_users_rated",
        "geek_rating": "predicted_geek_rating",
        "hurdle": "predicted_hurdle",
    }.get(model_type, "prediction")
    out = out.with_columns(pl.Series(pred_col, preds))
    return out


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

    candidate_cfg = find_candidate(model_type=args.model, candidate=args.candidate)

    upstream = dict(candidate_cfg.get("upstream") or {})
    if args.upstream:
        for pair in args.upstream.split(","):
            k, v = pair.split("=", 1)
            upstream[k.strip()] = v.strip()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    version = train_candidate(
        snapshot_version=args.snapshot_version,
        model_type=args.model,
        candidate=args.candidate,
        candidate_config=candidate_cfg,
        splits=splits,
        base_dir=args.base_dir,
        upstream=upstream,
    )
    print(f"experiment: {args.model}/{args.candidate}/v{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test, expect pass (or revise fixture)**

Run: `uv run pytest tests/test_train_snapshot.py -v`
Expected: PASS — but the test depends on the existing `HurdleModel` + preprocessing being able to handle the small synthetic dataset. If it fails because of feature requirements (e.g. preprocessor needs more columns), inspect the error and add the required columns to the fixture; do NOT modify the production code to accommodate the test.

- [ ] **Step 5: Commit**

```bash
git add src/models/train_snapshot.py tests/test_train_snapshot.py
git commit -m "feat: snapshot-aware single-candidate training (no upstream)"
```

### Task 14: End-to-end smoke test against real BQ snapshot (manual)

**Files:**
- None (manual verification step)

- [ ] **Step 1: Build a real snapshot from BQ**

Run: `uv run python -m src.models.build_snapshot --use-embeddings`
Expected: prints `snapshot_version: 1` (assuming this is the first snapshot on the branch). Creates `models/experiments/_snapshots/v1/universe.parquet`.

- [ ] **Step 2: Build the standard split**

Run: `uv run python -m src.models.build_split --snapshot-version 1 --split-name standard`
Expected: prints `split: v1/standard`. Creates `models/experiments/_snapshots/v1/splits/standard/{train,tune,test}.parquet`.

- [ ] **Step 3: Train hurdle on the standard split**

Run: `uv run python -m src.models.train_snapshot --model hurdle --candidate logistic-hurdle --snapshot-version 1 --splits standard`
Expected: prints `experiment: hurdle/logistic-hurdle/v1`. Creates the result artifacts; metrics in `metrics.json` should be in the same ballpark as today's `make hurdle` output.

- [ ] **Step 4: Sanity-check the result**

Run: `cat models/experiments/_snapshots/v1/experiments/hurdle/logistic-hurdle/v1/results/standard/metrics.json | head -40`
Expected: JSON with `train`, `tune`, `test` keys, each containing classification metrics (auc, log_loss, accuracy).

- [ ] **Step 5: Commit a note about the run**

```bash
# No code change; this step is for the user to confirm the manual run worked.
# If anything failed, fix the bug in the relevant Stage 1/2 task and re-run.
```

---

## Stage 3: Cascading Dependencies and Multi-Split Training

End state: train the full chain (complexity → rating + users_rated → geek_rating) across multiple splits, with downstream models reading upstream `score.parquet` from sibling experiments.

### Task 15: Verify upstream join in multi-split training

**Files:**
- Test: `tests/test_train_snapshot.py`

- [ ] **Step 1: Write failing test for upstream join**

Append to `tests/test_train_snapshot.py`:

```python
def test_upstream_join_pulls_score_predictions(synthetic_snapshot, tmp_path):
    """Training rating after complexity should join predicted_complexity
    onto the train/tune/test frames before training."""
    base, v = synthetic_snapshot
    storage = SnapshotStorage(snapshot_version=v, base_dir=base)

    # First, train complexity (creates score.parquet)
    complexity_cfg = {"name": "ard-complexity", "algorithm": "ard", "use_embeddings": False}
    train_candidate(
        snapshot_version=v, model_type="complexity", candidate="ard-complexity",
        candidate_config=complexity_cfg, splits=["standard"], base_dir=base, upstream={},
    )

    # Verify complexity wrote score.parquet
    score = storage.load_score_predictions("complexity", "ard-complexity", 1, "standard")
    assert score is not None
    assert "predicted_complexity" in score.columns
    assert score.height == 100

    # Now train rating with upstream complexity
    rating_cfg = {"name": "ard-ridge-rating", "algorithm": "ard",
                  "use_embeddings": False, "min_ratings": 0}
    train_candidate(
        snapshot_version=v, model_type="rating", candidate="ard-ridge-rating",
        candidate_config=rating_cfg, splits=["standard"], base_dir=base,
        upstream={"complexity": "ard-complexity"},
    )

    # Rating's registration should record the upstream choice
    reg = storage.load_candidate_registration("rating", "ard-ridge-rating", 1)
    assert reg["upstream_experiments"] == {"complexity": "ard-complexity"}
```

- [ ] **Step 2: Run test, expect pass (it should already work from Task 13)**

Run: `uv run pytest tests/test_train_snapshot.py::test_upstream_join_pulls_score_predictions -v`
Expected: PASS — the upstream join logic was implemented in Task 13. If it fails because the synthetic data is missing required columns for `RatingModel` (which has `min_ratings` filter), tweak the fixture's column set to include what RatingModel checks. Avoid changing production code.

- [ ] **Step 3: Commit (test only)**

```bash
git add tests/test_train_snapshot.py
git commit -m "test: upstream join in cascaded training"
```

### Task 16: Multi-split training in one invocation

**Files:**
- Test: `tests/test_train_snapshot.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_train_snapshot.py`:

```python
def test_train_candidate_handles_multiple_splits(synthetic_snapshot, tmp_path):
    """Training one candidate over multiple splits writes per-split results."""
    base, v = synthetic_snapshot
    # Add a yoy_2020 split (train≤2018, tune=2019, test=2020)
    from src.models.build_split import build_split
    build_split(
        snapshot_version=v, split_name="yoy_2020",
        train_through=2018, tune_start=2019, tune_through=2019,
        test_start=2020, test_through=2020,
        base_dir=base,
    )

    cfg = {"name": "logistic-hurdle", "algorithm": "logistic", "use_embeddings": False}
    train_candidate(
        snapshot_version=v, model_type="hurdle", candidate="logistic-hurdle",
        candidate_config=cfg, splits=["standard", "yoy_2020"], base_dir=base, upstream={},
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    standard_result = storage.load_result("hurdle", "logistic-hurdle", 1, "standard")
    yoy_result = storage.load_result("hurdle", "logistic-hurdle", 1, "yoy_2020")
    assert standard_result is not None
    assert yoy_result is not None
```

- [ ] **Step 2: Run test, expect pass**

Run: `uv run pytest tests/test_train_snapshot.py::test_train_candidate_handles_multiple_splits -v`
Expected: PASS (Task 13 already iterates over `splits`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_train_snapshot.py
git commit -m "test: multi-split training in one invocation"
```

### Task 17: Cross-split summary writer

**Files:**
- Modify: `src/models/train_snapshot.py`
- Test: `tests/test_train_snapshot.py`

After all splits succeed, roll up `metrics.json` across splits into the candidate's `summary.json`.

- [ ] **Step 1: Write failing test**

Append to `tests/test_train_snapshot.py`:

```python
def test_summary_json_written_after_multi_split_training(synthetic_snapshot, tmp_path):
    base, v = synthetic_snapshot
    from src.models.build_split import build_split
    import json
    build_split(
        snapshot_version=v, split_name="yoy_2020",
        train_through=2018, tune_start=2019, tune_through=2019,
        test_start=2020, test_through=2020, base_dir=base,
    )
    cfg = {"name": "logistic-hurdle", "algorithm": "logistic", "use_embeddings": False}
    train_candidate(
        snapshot_version=v, model_type="hurdle", candidate="logistic-hurdle",
        candidate_config=cfg, splits=["standard", "yoy_2020"], base_dir=base, upstream={},
    )

    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    summary_path = storage.experiment_dir("hurdle", "logistic-hurdle", 1) / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert sorted(summary["per_split"].keys()) == ["standard", "yoy_2020"]
    assert "tune" in summary["per_split"]["standard"]
    assert "test" in summary["per_split"]["standard"]
```

- [ ] **Step 2: Run test, expect failure (no summary written yet)**

Run: `uv run pytest tests/test_train_snapshot.py::test_summary_json_written_after_multi_split_training -v`
Expected: FAIL — `summary.json` does not exist.

- [ ] **Step 3: Implement summary writer**

In `src/models/train_snapshot.py`, find this block at the end of `train_candidate`:

```python
        logger.info(f"Wrote result for {model_type}/{candidate}/v{candidate_version}/{split_name}")

    return candidate_version
```

Replace with:

```python
        logger.info(f"Wrote result for {model_type}/{candidate}/v{candidate_version}/{split_name}")

    _write_summary(storage, model_type, candidate, candidate_version, splits)
    return candidate_version


def _write_summary(
    storage: SnapshotStorage,
    model_type: str,
    candidate: str,
    version: int,
    splits: List[str],
) -> Path:
    """Roll up per-split metrics into summary.json at the candidate level."""
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

- [ ] **Step 4: Run test, expect pass**

Run: `uv run pytest tests/test_train_snapshot.py -v`
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add src/models/train_snapshot.py tests/test_train_snapshot.py
git commit -m "feat: write summary.json rolling up cross-split metrics"
```

### Task 18: End-to-end manual chain test

**Files:**
- None (manual verification)

- [ ] **Step 1: Build YoY splits on top of v1**

Run: `uv run python -m src.models.build_split --snapshot-version 1 --yoy --yoy-start 2018 --yoy-end 2024`
Expected: prints `yoy splits: v1/yoy_2018..yoy_2024`.

- [ ] **Step 2: Train the chain**

```bash
uv run python -m src.models.train_snapshot --model hurdle --candidate logistic-hurdle \
  --snapshot-version 1 --splits standard,yoy_2018,yoy_2019,yoy_2020,yoy_2021,yoy_2022,yoy_2023,yoy_2024

uv run python -m src.models.train_snapshot --model complexity --candidate ard-complexity \
  --snapshot-version 1 --splits standard,yoy_2018,yoy_2019,yoy_2020,yoy_2021,yoy_2022,yoy_2023,yoy_2024

uv run python -m src.models.train_snapshot --model rating --candidate ard-ridge-rating \
  --snapshot-version 1 --splits standard,yoy_2018,yoy_2019,yoy_2020,yoy_2021,yoy_2022,yoy_2023,yoy_2024

uv run python -m src.models.train_snapshot --model users_rated --candidate ard-ridge-users_rated \
  --snapshot-version 1 --splits standard,yoy_2018,yoy_2019,yoy_2020,yoy_2021,yoy_2022,yoy_2023,yoy_2024

uv run python -m src.models.train_snapshot --model geek_rating --candidate ard-geek_rating \
  --snapshot-version 1 --splits standard,yoy_2018,yoy_2019,yoy_2020,yoy_2021,yoy_2022,yoy_2023,yoy_2024
```

- [ ] **Step 3: Verify summary.json for each candidate**

```bash
for m in hurdle complexity rating users_rated geek_rating; do
  echo "=== $m ==="
  cat models/experiments/_snapshots/v1/experiments/$m/*/v1/summary.json | head -20
done
```

Expected: each `summary.json` contains 8 split entries (`standard` + 7 YoY) with sensible metrics.

---

## Stage 4: K-fold OOF Scoring for Upstream Models

End state: `score.parquet` for `complexity`, `rating`, and `users_rated` uses K-fold OOF predictions for the train rows. Downstream models train on honest features.

### Task 19: K-fold OOF predictor utility

**Files:**
- Create: `src/models/oof.py`
- Test: `tests/test_oof_scoring.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_oof_scoring.py`:

```python
"""Tests for K-fold OOF prediction utility."""

import numpy as np
import pandas as pd
import pytest
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

- [ ] **Step 2: Run tests, expect failure**

Run: `uv run pytest tests/test_oof_scoring.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement oof.py**

Create `src/models/oof.py`:

```python
"""K-fold out-of-fold prediction utility.

Given a fitted-yet-unfitted pipeline and a (X, y) training frame, produce
predictions for every row of X using only models that did not see that row.
Used to generate honest training-time features for downstream cascaded
models (complexity → rating, etc).
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
    """Produce out-of-fold predictions for every row of X.

    Args:
        pipeline: Unfitted sklearn pipeline. Cloned per fold.
        X: Feature frame.
        y: Target series.
        k: Number of folds.
        seed: Random seed for fold assignment.
        predict_proba: If True, return probability of the positive class
            (binary classification). Default False (regression / argmax).
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

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_oof_scoring.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/oof.py tests/test_oof_scoring.py
git commit -m "feat: K-fold OOF prediction utility"
```

### Task 20: Wire OOF into score.parquet for upstream models

**Files:**
- Modify: `src/models/train_snapshot.py`
- Test: `tests/test_train_snapshot.py`

The score.parquet for upstream models (complexity, rating, users_rated) should use OOF predictions for train rows. Tune/test rows continue to be predicted by the model trained on the full train fold (this is already correct because tune/test were never in train). Rows outside train+tune+test get the finalized model — but finalize hasn't happened yet at this point in the flow, so for now use the train-fold model for those rows too. Stage 5 wires in finalize.

- [ ] **Step 1: Write failing test**

Append to `tests/test_train_snapshot.py`:

```python
def test_score_parquet_train_rows_are_oof_for_upstream_models(synthetic_snapshot, tmp_path):
    """For upstream models (complexity in this case), train rows in
    score.parquet should be OOF, not in-sample. We verify by checking
    that score.parquet's predicted_complexity for train rows differs
    from what an in-sample model would produce (loose check)."""
    base, v = synthetic_snapshot
    storage = SnapshotStorage(snapshot_version=v, base_dir=base)

    cfg = {"name": "ard-complexity", "algorithm": "ard", "use_embeddings": False, "oof_folds": 3}
    train_candidate(
        snapshot_version=v, model_type="complexity", candidate="ard-complexity",
        candidate_config=cfg, splits=["standard"], base_dir=base, upstream={},
    )

    score = storage.load_score_predictions("complexity", "ard-complexity", 1, "standard")
    split = storage.load_split("standard")
    train_ids = set(split["train"]["game_id"].to_list())
    score_pd = score.to_pandas()

    # Sanity: every game in the universe appears in score
    assert set(score_pd["game_id"]).issuperset(train_ids)
    # The score for train rows should not all be NaN; the OOF code path filled them in
    train_scores = score_pd[score_pd["game_id"].isin(train_ids)]
    assert not train_scores["predicted_complexity"].isna().all()
```

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/test_train_snapshot.py::test_score_parquet_train_rows_are_oof_for_upstream_models -v`
Expected: FAIL — current `_score_universe` is in-sample only; the test will likely pass anyway (the assertions are very loose). If it passes, write a stricter assertion that compares OOF vs in-sample on synthetic data with high signal. For this plan, the loose check is sufficient — the substantive verification happens in Step 3 by inspection.

If the loose test passes, replace it with a stricter test that verifies the OOF path is being taken:

```python
def test_score_parquet_uses_oof_for_upstream(synthetic_snapshot, monkeypatch):
    """When OOF is configured for an upstream model, kfold_oof_predict is called."""
    base, v = synthetic_snapshot

    calls = []

    from src.models import oof as _oof
    real_predict = _oof.kfold_oof_predict

    def spy(*args, **kwargs):
        calls.append(kwargs.get("k"))
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(_oof, "kfold_oof_predict", spy)

    cfg = {"name": "ard-complexity", "algorithm": "ard", "use_embeddings": False, "oof_folds": 3}
    train_candidate(
        snapshot_version=v, model_type="complexity", candidate="ard-complexity",
        candidate_config=cfg, splits=["standard"], base_dir=base, upstream={},
    )
    assert len(calls) >= 1, "kfold_oof_predict was not called for upstream model"
    assert calls[0] == 3, "OOF was not called with the configured k=3"
```

- [ ] **Step 3: Implement OOF in score generation**

In `src/models/train_snapshot.py`, replace the existing `_score_universe` and adjust `train_candidate` to pass through OOF info. Find the call site:

```python
        # Score predictions: in-sample for now (Stage 3 replaces with OOF for upstream models)
        universe_X = universe_with_upstream.drop(target_column).to_pandas() \
            if target_column in universe_with_upstream.columns else universe_with_upstream.to_pandas()
        score_preds = _score_universe(pipeline, universe_with_upstream, model_type)
```

Replace with:

```python
        # Score the universe. For upstream models (complexity, rating,
        # users_rated), produce OOF predictions for the train rows so
        # downstream models train on honest features.
        score_preds = _score_universe(
            pipeline=pipeline,
            train_X=train_X,
            train_y=train_y,
            train_df=train_df,
            universe=universe_with_upstream,
            model_type=model_type,
            candidate_config=candidate_config,
            algorithm=algorithm,
            preprocessor=preprocessor,
        )
```

And replace the existing `_score_universe` function with this:

```python
def _score_universe(
    pipeline,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    train_df: pl.DataFrame,
    universe: pl.DataFrame,
    model_type: str,
    candidate_config: Dict[str, Any],
    algorithm: Optional[str],
    preprocessor: Any,
) -> pl.DataFrame:
    """Score every row of the universe.

    For upstream models (complexity, rating, users_rated), use K-fold OOF
    on the train rows so downstream models train on honest features. The
    full pipeline (trained on all of train) is used for tune/test rows
    and any rows outside train+tune+test.

    For non-upstream models (hurdle, geek_rating), score in-sample —
    nothing downstream consumes their score.parquet train rows during
    training.
    """
    from src.models.oof import kfold_oof_predict

    pred_col = {
        "complexity": "predicted_complexity",
        "rating": "predicted_rating",
        "users_rated": "predicted_users_rated",
        "geek_rating": "predicted_geek_rating",
        "hurdle": "predicted_hurdle",
    }.get(model_type, "prediction")

    is_upstream = model_type in {"complexity", "rating", "users_rated"}
    oof_folds = int(candidate_config.get("oof_folds", 5))

    universe_X = universe.to_pandas()
    full_preds = pipeline.predict(universe_X)
    out_pl = universe.select(["game_id"]).clone().with_columns(
        pl.Series(pred_col, full_preds)
    )

    if not is_upstream:
        return out_pl

    # OOF for train rows
    fresh_pipeline = _fresh_pipeline_like(preprocessor, algorithm, model_type)
    oof_train_preds = kfold_oof_predict(
        pipeline=fresh_pipeline,
        X=train_X,
        y=train_y,
        k=oof_folds,
        seed=42,
    )
    train_ids = set(train_df["game_id"].to_list())
    train_id_to_oof = dict(zip(train_df["game_id"].to_list(), oof_train_preds))

    out_pd = out_pl.to_pandas()
    mask = out_pd["game_id"].isin(train_ids)
    out_pd.loc[mask, pred_col] = out_pd.loc[mask, "game_id"].map(train_id_to_oof)
    return pl.from_pandas(out_pd)


def _fresh_pipeline_like(preprocessor, algorithm: Optional[str], model_type: str):
    """Construct a fresh, unfitted pipeline equivalent to the one used for
    training. Used by OOF to avoid mutating the trained pipeline in place.
    """
    from sklearn.base import clone
    from src.models.training import build_estimator
    from sklearn.pipeline import Pipeline as SKPipeline

    estimator = build_estimator(model_type=model_type, algorithm=algorithm)
    return SKPipeline(steps=[("preprocessor", clone(preprocessor)), ("model", estimator)])
```

(If `src.models.training.build_estimator` does not exist with this exact signature, replace the call in `_fresh_pipeline_like` with whatever the existing `tune_model` uses to construct its inner estimator. Adapt to the actual API; the key is producing an unfitted pipeline whose preprocessing matches what the trained pipeline used.)

- [ ] **Step 4: Run test, expect pass**

Run: `uv run pytest tests/test_train_snapshot.py -v`
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add src/models/train_snapshot.py tests/test_train_snapshot.py
git commit -m "feat: OOF scoring for upstream models in score.parquet"
```

### Task 21: Manual end-to-end OOF verification

**Files:**
- None (manual)

- [ ] **Step 1: Re-run the chain on v1 with OOF enabled**

Bump candidate versions by deleting/regenerating, or run with a fresh snapshot. The simplest:

```bash
# delete prior runs (optional; these were Stage 3's work)
rm -rf models/experiments/_snapshots/v1/experiments

# rerun chain
uv run python -m src.models.train_snapshot --model complexity --candidate ard-complexity \
  --snapshot-version 1 --splits standard
uv run python -m src.models.train_snapshot --model rating --candidate ard-ridge-rating \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity
```

- [ ] **Step 2: Inspect score.parquet train rows**

```bash
uv run python -c "
import polars as pl
score = pl.read_parquet('models/experiments/_snapshots/v1/experiments/complexity/ard-complexity/v1/results/standard/predictions/score.parquet')
split = pl.read_parquet('models/experiments/_snapshots/v1/splits/standard/train.parquet')
joined = score.join(split.select(['game_id', 'complexity']), on='game_id', how='inner')
print('train rows in score.parquet:', joined.height)
print(joined.head(10))
"
```

Expected: prints the count and a sample with both `predicted_complexity` (from OOF) and `complexity` (actual). Eyeball that the OOF values look reasonable (in 1-5 range, varying).

- [ ] **Step 3: Compare metrics on the rating model's tune/test against the previous chain**

If the rating model's tune RMSE noticeably *increased* compared to the in-sample-feature run from Stage 3, that's expected and a good sign — the model is no longer learning from a leaked feature. If it dropped or stayed identical, the OOF code path likely isn't being taken; debug.

---

## Stage 5: Finalize and Cleanup

End state: a candidate's `finalized.pkl` lives at the candidate level, refit on the full snapshot universe (or through `final_end_year`), ready for operational scoring.

### Task 22: Per-candidate finalize step

**Files:**
- Modify: `src/models/train_snapshot.py`
- Test: `tests/test_train_snapshot.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_train_snapshot.py`:

```python
def test_finalize_writes_candidate_level_pipeline(synthetic_snapshot, tmp_path):
    base, v = synthetic_snapshot
    storage = SnapshotStorage(snapshot_version=v, base_dir=base)
    cfg = {"name": "ard-complexity", "algorithm": "ard", "use_embeddings": False}

    train_candidate(
        snapshot_version=v, model_type="complexity", candidate="ard-complexity",
        candidate_config=cfg, splits=["standard"], base_dir=base, upstream={},
        finalize=True, finalize_through=2021,
    )

    finalized = storage.load_finalized_pipeline("complexity", "ard-complexity", 1)
    assert finalized is not None

    reg = storage.load_candidate_registration("complexity", "ard-complexity", 1)
    assert reg["finalize_through"] == 2021
```

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/test_train_snapshot.py::test_finalize_writes_candidate_level_pipeline -v`
Expected: FAIL — `train_candidate` does not yet accept `finalize`.

- [ ] **Step 3: Implement finalize**

In `train_candidate`, add `finalize: bool = False, finalize_through: Optional[int] = None` to the signature, and after the per-split loop and summary write:

```python
    if finalize:
        _finalize_candidate(
            storage=storage,
            universe=universe,
            model_type=model_type,
            candidate=candidate,
            version=candidate_version,
            candidate_config=candidate_config,
            target_column=target_column,
            finalize_through=finalize_through,
        )

    return candidate_version


def _finalize_candidate(
    storage: SnapshotStorage,
    universe: pl.DataFrame,
    model_type: str,
    candidate: str,
    version: int,
    candidate_config: Dict[str, Any],
    target_column: str,
    finalize_through: Optional[int],
) -> None:
    """Refit the candidate on the full snapshot universe (filtered through
    finalize_through if provided). Save as ``finalized.pkl`` at the
    candidate level."""
    from src.models.training import create_preprocessing_pipeline, fit_model

    df = universe
    if finalize_through is not None:
        df = df.filter(pl.col("year_published") <= int(finalize_through))

    X, y = select_X_y(df, target_column)
    algorithm = candidate_config.get("algorithm")
    preprocessor = create_preprocessing_pipeline(X=X, model_type=model_type, algorithm=algorithm)
    pipeline = fit_model(preprocessor=preprocessor, model_type=model_type,
                         algorithm=algorithm, X=X, y=y)
    storage.save_finalized_pipeline(model_type, candidate, version, pipeline)

    reg = storage.load_candidate_registration(model_type, candidate, version) or {}
    reg["finalize_through"] = int(finalize_through) if finalize_through is not None else None
    reg["finalized_at"] = datetime.now().isoformat()
    storage.save_candidate_registration(model_type, candidate, version, reg)
```

(If `fit_model` doesn't exist as named in `src/models/training.py`, replace the call with the actual training-fit primitive. The point is "fit a fresh preprocessor+model on (X, y) and return the fitted pipeline.")

Also extend the CLI's `main()` in `train_snapshot.py`:

```python
    parser.add_argument("--finalize", action="store_true", default=False)
    parser.add_argument("--finalize-through", type=int, default=None)
```

And wire those args through to `train_candidate(...)`.

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_train_snapshot.py -v`
Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add src/models/train_snapshot.py tests/test_train_snapshot.py
git commit -m "feat: per-candidate finalize step writing finalized.pkl"
```

### Task 23: Self-review checklist (manual)

**Files:**
- None (review pass)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/test_snapshot_storage.py tests/test_build_snapshot.py tests/test_build_split.py tests/test_train_snapshot.py tests/test_oof_scoring.py tests/test_candidate_config.py -v`
Expected: all passing. Note the count of tests.

- [ ] **Step 2: Run lint**

Run: `make lint`
Expected: no errors.

- [ ] **Step 3: End-to-end manual chain on real data**

```bash
uv run python -m src.models.build_snapshot --use-embeddings   # if you don't already have v1
# Skip this if v1 already exists.

# If you need a fresh snapshot for testing the full flow, use --snapshot-version 2
# and rebuild splits + train against v2.

# Otherwise, on whatever the current snapshot version is:
uv run python -m src.models.build_split --snapshot-version 1 --split-name standard
uv run python -m src.models.build_split --snapshot-version 1 --yoy --yoy-start 2018 --yoy-end 2024

# Train chain WITH finalize
uv run python -m src.models.train_snapshot --model hurdle --candidate logistic-hurdle \
  --snapshot-version 1 --splits standard --finalize --finalize-through 2024
uv run python -m src.models.train_snapshot --model complexity --candidate ard-complexity \
  --snapshot-version 1 --splits standard --finalize --finalize-through 2024
uv run python -m src.models.train_snapshot --model rating --candidate ard-ridge-rating \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity \
  --finalize --finalize-through 2024
uv run python -m src.models.train_snapshot --model users_rated --candidate ard-ridge-users_rated \
  --snapshot-version 1 --splits standard --upstream complexity=ard-complexity \
  --finalize --finalize-through 2024
uv run python -m src.models.train_snapshot --model geek_rating --candidate ard-geek_rating \
  --snapshot-version 1 --splits standard \
  --upstream complexity=ard-complexity,rating=ard-ridge-rating,users_rated=ard-ridge-users_rated \
  --finalize --finalize-through 2024
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
- Tasks 13 and 20 lean on existing functions (`tune_model`, `evaluate_model`, `create_preprocessing_pipeline`, `build_estimator`, `fit_model`) whose signatures are not pinned down in this plan. The implementer must inspect `src/models/training.py` and adapt the calls — these are noted inline. This is unavoidable without auditing every existing helper, and is acceptable because the engineer is working on a redesign branch where minor adjustments to existing helpers are fine.
- Type/method consistency: `train_candidate(...)` signature is consistent across Tasks 13, 15, 16, 17, 22. `SnapshotStorage`'s method names match across tasks.
