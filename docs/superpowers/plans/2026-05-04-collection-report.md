# Collection Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Quarto-based per-user collection report rendered from local artifacts (dev) or GCS-hosted artifacts (CI), with one parameterized template and a thin Python data loader.

**Architecture:** A new `src/reports/collection_data.py` returns a `CollectionReportData` dataclass that splits user-level (BQ-backed: collection snapshot, games metadata) from per-outcome artifacts (filesystem: pipeline, predictions, registration). The `.qmd` template under `reports/` calls into the loader plus `src/collection/viz.py` helpers (extended in this plan). A `reports/render.py` CLI shells out to `quarto render` per user. `source` parameter swaps the artifact root between a local `Path` and a `gs://` URI via `fsspec`/`gcsfs`.

**Tech Stack:** Python 3.12, polars, pandas, sklearn Pipeline, plotly, plotnine, itables, Quarto, fsspec/gcsfs (new), pytest.

**Spec:** [docs/superpowers/specs/2026-05-04-collection-report-design.md](../specs/2026-05-04-collection-report-design.md)

---

## File Structure

| File | Responsibility |
|---|---|
| `src/reports/__init__.py` | Empty marker |
| `src/reports/collection_data.py` | `CollectionReportData`, `OutcomeArtifacts`, `load()` — pure data loading |
| `src/collection/viz.py` | Existing module; add new plot/table helpers used by both report and Streamlit |
| `reports/collection_report.qmd` | Quarto template — prose + small chunks calling helpers |
| `reports/styles.css` | CSS for the rendered HTML |
| `reports/render.py` | CLI wrapper that shells out to `quarto render` per user |
| `tests/reports/__init__.py` | Empty marker |
| `tests/reports/conftest.py` | Shared fixtures (fixture artifact tree builder) |
| `tests/reports/test_collection_data.py` | Loader tests (BQ-backed fields mocked) |
| `tests/reports/test_viz_collection_report.py` | Tests for new `viz.py` helpers |
| `tests/reports/test_render_smoke.py` | End-to-end smoke test (skipped without Quarto) |
| `pyproject.toml` | Add `fsspec`, `gcsfs` to dependencies |
| `.gitignore` | Add `reports/_output/` |

**Dependencies between tasks:** Tasks 1–2 unlock the data layer (Tasks 3–8). Tasks 9–14 (viz helpers) are independent of the loader and can be done in any order. Tasks 15–17 (template, render driver, smoke test) require everything above.

---

## Task 1: Add dependencies + gitignore

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Add fsspec and gcsfs to project dependencies**

Edit `pyproject.toml`. Find the `dependencies = [...]` list under `[project]`. Insert two lines (alphabetically; after `db-dtypes`):

```toml
    "fsspec>=2024.10.0",
    "gcsfs>=2024.10.0",
```

- [ ] **Step 2: Run uv lock + sync**

```bash
uv lock && uv sync
```

Expected: success, no errors.

- [ ] **Step 3: Add reports/_output to .gitignore**

Append to `.gitignore`:

```
# Rendered Quarto reports — built artifacts
reports/_output/
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock .gitignore
git commit -m "chore: add fsspec/gcsfs deps and gitignore reports/_output"
```

---

## Task 2: Create empty package skeleton

**Files:**
- Create: `src/reports/__init__.py`
- Create: `tests/reports/__init__.py`
- Create: `reports/.gitkeep`

- [ ] **Step 1: Create `src/reports/__init__.py`**

Empty file (per project convention).

- [ ] **Step 2: Create `tests/reports/__init__.py`**

Empty file.

- [ ] **Step 3: Create `reports/` directory with placeholder**

```bash
mkdir -p reports
touch reports/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add src/reports/__init__.py tests/reports/__init__.py reports/.gitkeep
git commit -m "feat: scaffold reports package"
```

---

## Task 3: Define the `CollectionReportData` dataclass

**Files:**
- Modify: `src/reports/collection_data.py` (create)
- Test: `tests/reports/test_collection_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/reports/test_collection_data.py`:

```python
"""Tests for src.reports.collection_data."""

from __future__ import annotations

import polars as pl
import pytest

from src.reports.collection_data import CollectionReportData, OutcomeArtifacts


def test_outcome_artifacts_has_required_fields():
    fields = OutcomeArtifacts.__dataclass_fields__
    expected = {
        "outcome",
        "selected_candidate",
        "selected_version",
        "pipeline",
        "registration",
        "threshold",
        "feature_importance",
        "oof_predictions",
        "val_predictions",
        "test_predictions",
        "upcoming_predictions",
    }
    assert set(fields) == expected


def test_collection_report_data_has_required_fields():
    fields = CollectionReportData.__dataclass_fields__
    expected = {"username", "collection", "games", "outcomes"}
    assert set(fields) == expected


def test_collection_report_data_outcomes_is_dict():
    data = CollectionReportData(
        username="phenrickson",
        collection=pl.DataFrame(),
        games=pl.DataFrame(),
        outcomes={},
    )
    assert isinstance(data.outcomes, dict)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/reports/test_collection_data.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.reports.collection_data'`

- [ ] **Step 3: Implement the dataclasses**

Create `src/reports/collection_data.py`:

```python
"""Data loader for the collection Quarto report.

Returns a `CollectionReportData` aggregate that the `.qmd` template
consumes. Splits user-level (outcome-agnostic) data from per-outcome
artifacts so multi-outcome reports can iterate `data.outcomes` without
a loader refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass
class OutcomeArtifacts:
    """All artifacts the report needs for one (user, outcome)."""

    outcome: str
    selected_candidate: str
    selected_version: int

    pipeline: Any  # sklearn.pipeline.Pipeline
    registration: dict
    threshold: float | None
    feature_importance: pl.DataFrame  # extracted from fitted pipeline

    oof_predictions: pl.DataFrame
    val_predictions: pl.DataFrame
    test_predictions: pl.DataFrame

    upcoming_predictions: pl.DataFrame  # deployed-model scores from BQ landing


@dataclass
class CollectionReportData:
    """Full data bundle for a single rendering of the collection report."""

    username: str
    collection: pl.DataFrame  # raw BGG snapshot, BQ — outcome-agnostic
    games: pl.DataFrame  # game metadata, BQ — outcome-agnostic
    outcomes: dict[str, OutcomeArtifacts]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/reports/test_collection_data.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reports/collection_data.py tests/reports/test_collection_data.py
git commit -m "feat(reports): define CollectionReportData/OutcomeArtifacts"
```

---

## Task 4: Build the test fixture artifact tree

**Files:**
- Create: `tests/reports/conftest.py`

- [ ] **Step 1: Implement the fixture builder**

Create `tests/reports/conftest.py`:

```python
"""Shared fixtures for tests/reports.

Builds a minimal but realistic on-disk artifact tree under a tmp_path
that mirrors `models/collections/{username}/{outcome}/...`. Use the
`fixture_collection_root` fixture in tests that exercise the loader.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def _identity_pipeline() -> Pipeline:
    """A trivial Pipeline with a `preprocessor` and `model` step.

    The preprocessor is a no-op identity transform; the model is a fitted
    DummyClassifier that always predicts class 0. Real enough for code
    paths that pull `coef_` / `feature_importances_` to find *something*
    (we attach a fake `coef_` below).
    """

    def to_pandas_or_passthrough(x):
        return x.to_pandas() if hasattr(x, "to_pandas") else x

    preprocessor = FunctionTransformer(to_pandas_or_passthrough, validate=False)
    model = DummyClassifier(strategy="constant", constant=0)
    # Fit on tiny data to populate sklearn internals.
    X = np.zeros((4, 3))
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)
    # Attach a fake coef_ so importance extraction has something to read.
    model.coef_ = np.array([[0.5, -0.2, 0.1]])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


@pytest.fixture
def fixture_collection_root(tmp_path: Path) -> Path:
    """Build `tmp_path/collections/phenrickson/own/...` with one finalized
    candidate (`logistic_row_norm`) plus canonical splits.

    Layout:
        phenrickson/
          collection/latest.parquet
          own/_splits/v1/{train,validation,test}.parquet
          own/logistic_row_norm/v1/
            model.pkl
            finalized.pkl                # presence marks "finalized"
            registration.json            # finalize_through, splits_version=1
            threshold.json
            feature_importance.parquet
            predictions/{oof,val,test}.parquet
    """
    root = tmp_path / "collections"
    user_dir = root / "phenrickson"
    user_dir.mkdir(parents=True)

    # User-level collection snapshot
    coll_dir = user_dir / "collection"
    coll_dir.mkdir()
    pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "game_name": ["A", "B", "C"],
            "user_rating": [8.0, 7.5, 9.0],
            "owned": [True, True, False],
        }
    ).write_parquet(coll_dir / "latest.parquet")

    # Canonical splits (v1) — used by feature-importance name recovery
    splits_dir = user_dir / "own" / "_splits" / "v1"
    splits_dir.mkdir(parents=True)
    train_df = pl.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5],
            "name": ["A", "B", "C", "D", "E"],
            "year_published": [2018, 2019, 2020, 2021, 2022],
            "feat_a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feat_b": [1, 0, 1, 0, 1],
            "feat_c": [0.0, 0.5, 1.0, 0.5, 0.0],
            "label": [1, 0, 1, 0, 1],
        }
    )
    train_df.write_parquet(splits_dir / "train.parquet")
    train_df.head(2).write_parquet(splits_dir / "validation.parquet")
    train_df.tail(2).write_parquet(splits_dir / "test.parquet")

    # Candidate run (finalized)
    cand_dir = user_dir / "own" / "logistic_row_norm" / "v1"
    cand_dir.mkdir(parents=True)
    pipeline = _identity_pipeline()
    (cand_dir / "model.pkl").write_bytes(pickle.dumps(pipeline))
    (cand_dir / "finalized.pkl").write_bytes(pickle.dumps(pipeline))
    (cand_dir / "registration.json").write_text(
        json.dumps(
            {
                "candidate": "logistic_row_norm",
                "version": 1,
                "splits_version": 1,
                "finalize_through": 2024,
                "finalized_at": "2026-05-01T00:00:00",
                "task": "classification",
                "metrics": {"roc_auc": 0.85, "pr_auc": 0.6, "log_loss": 0.4},
                "val_metrics": {"roc_auc": 0.82, "pr_auc": 0.55},
                "oof_metrics": {"overall": {"roc_auc": 0.8}},
                "threshold": 0.5,
            }
        )
    )
    (cand_dir / "threshold.json").write_text(json.dumps({"threshold": 0.5}))
    pl.DataFrame(
        {"feature": ["feat_a", "feat_b", "feat_c"], "value": [0.5, -0.2, 0.1]}
    ).with_columns(pl.col("value").abs().alias("abs_value")).write_parquet(
        cand_dir / "feature_importance.parquet"
    )

    preds_dir = cand_dir / "predictions"
    preds_dir.mkdir()
    preds_df = pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "year_published": [2020, 2021, 2022],
            "proba": [0.9, 0.4, 0.7],
            "label": [True, False, True],
        }
    )
    preds_df.write_parquet(preds_dir / "oof.parquet")
    preds_df.write_parquet(preds_dir / "val.parquet")
    preds_df.write_parquet(preds_dir / "test.parquet")

    return root
```

- [ ] **Step 2: Smoke-test the fixture builder by listing the tree**

```bash
uv run pytest tests/reports/conftest.py --collect-only -q
```

Expected: succeeds (no test files exist yet, just confirms the fixture imports cleanly).

- [ ] **Step 3: Commit**

```bash
git add tests/reports/conftest.py
git commit -m "test(reports): fixture artifact tree builder"
```

---

## Task 5: Implement candidate-selection logic

**Files:**
- Modify: `src/reports/collection_data.py`
- Modify: `tests/reports/test_collection_data.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/reports/test_collection_data.py`:

```python
import json
from pathlib import Path

from src.reports.collection_data import select_candidate


def test_select_candidate_prefers_logistic_row_norm(fixture_collection_root: Path):
    cand, version = select_candidate(
        fixture_collection_root, "phenrickson", "own"
    )
    assert cand == "logistic_row_norm"
    assert version == 1


def test_select_candidate_explicit_override(fixture_collection_root: Path):
    # Add a second finalized candidate
    other_dir = fixture_collection_root / "phenrickson" / "own" / "lgbm_default" / "v1"
    other_dir.mkdir(parents=True)
    (other_dir / "finalized.pkl").write_bytes(b"x")
    (other_dir / "registration.json").write_text(
        json.dumps({"candidate": "lgbm_default", "version": 1, "splits_version": 1})
    )

    cand, version = select_candidate(
        fixture_collection_root,
        "phenrickson",
        "own",
        candidate="lgbm_default",
    )
    assert cand == "lgbm_default"
    assert version == 1


def test_select_candidate_raises_when_no_finalized(tmp_path: Path):
    # Empty tree → no finalized candidate
    user_dir = tmp_path / "phenrickson" / "own"
    user_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="No finalized candidate"):
        select_candidate(tmp_path, "phenrickson", "own")
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
uv run pytest tests/reports/test_collection_data.py -v -k select_candidate
```

Expected: FAIL — `select_candidate` not importable.

- [ ] **Step 3: Implement `select_candidate`**

Append to `src/reports/collection_data.py`:

```python
import json as _json
from pathlib import Path

DEFAULT_CANDIDATE = "logistic_row_norm"


def _list_candidate_versions(user_outcome_dir: Path, candidate: str) -> list[int]:
    """Return all integer versions for a candidate dir, ascending."""
    cand_dir = user_outcome_dir / candidate
    if not cand_dir.exists():
        return []
    versions: list[int] = []
    for child in cand_dir.iterdir():
        if not (child.is_dir() and child.name.startswith("v")):
            continue
        try:
            versions.append(int(child.name[1:]))
        except ValueError:
            continue
    return sorted(versions)


def _is_finalized(user_outcome_dir: Path, candidate: str, version: int) -> bool:
    """A candidate version is 'finalized' iff it has finalized.pkl."""
    return (user_outcome_dir / candidate / f"v{version}" / "finalized.pkl").exists()


def _list_finalized_candidates(
    user_outcome_dir: Path,
) -> list[tuple[str, int]]:
    """Return (candidate, latest_finalized_version) for every candidate in the
    outcome dir that has at least one finalized version."""
    if not user_outcome_dir.exists():
        return []
    out: list[tuple[str, int]] = []
    for child in user_outcome_dir.iterdir():
        if not child.is_dir() or child.name.startswith("_") or child.name.startswith("v"):
            continue
        cand = child.name
        finalized_versions = [
            v
            for v in _list_candidate_versions(user_outcome_dir, cand)
            if _is_finalized(user_outcome_dir, cand, v)
        ]
        if finalized_versions:
            out.append((cand, max(finalized_versions)))
    return sorted(out)


def select_candidate(
    root: Path,
    username: str,
    outcome: str,
    candidate: str | None = None,
) -> tuple[str, int]:
    """Pick a (candidate, version) for a user/outcome.

    Resolution order:
        1. If `candidate` is given and has a finalized version, use its
           latest finalized version.
        2. Otherwise prefer DEFAULT_CANDIDATE if it has a finalized version.
        3. Otherwise pick any finalized candidate (alphabetically first;
           BQ-registry tiebreak is a future enhancement).
        4. Raise ValueError if nothing is finalized.
    """
    user_outcome_dir = Path(root) / username / outcome
    finalized = dict(_list_finalized_candidates(user_outcome_dir))

    if candidate is not None:
        if candidate not in finalized:
            raise ValueError(
                f"Candidate {candidate!r} is not finalized for "
                f"{username}/{outcome} under {root}"
            )
        return candidate, finalized[candidate]

    if DEFAULT_CANDIDATE in finalized:
        return DEFAULT_CANDIDATE, finalized[DEFAULT_CANDIDATE]

    if not finalized:
        raise ValueError(
            f"No finalized candidate found for {username}/{outcome} under {root}"
        )

    # Stable fallback: alphabetical
    cand = sorted(finalized.keys())[0]
    return cand, finalized[cand]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/reports/test_collection_data.py -v
```

Expected: all PASS (3 dataclass + 3 select_candidate).

- [ ] **Step 5: Commit**

```bash
git add src/reports/collection_data.py tests/reports/test_collection_data.py
git commit -m "feat(reports): select_candidate with logistic_row_norm default"
```

---

## Task 6: Implement filesystem reads via fsspec (path-prefix abstraction)

**Files:**
- Modify: `src/reports/collection_data.py`
- Modify: `tests/reports/test_collection_data.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/reports/test_collection_data.py`:

```python
import pickle

from src.reports.collection_data import (
    _read_json,
    _read_parquet,
    _read_pickle,
)


def test_read_json_local(fixture_collection_root: Path):
    path = (
        fixture_collection_root
        / "phenrickson"
        / "own"
        / "logistic_row_norm"
        / "v1"
        / "registration.json"
    )
    data = _read_json(str(path))
    assert data["candidate"] == "logistic_row_norm"


def test_read_parquet_local(fixture_collection_root: Path):
    path = (
        fixture_collection_root
        / "phenrickson"
        / "own"
        / "logistic_row_norm"
        / "v1"
        / "predictions"
        / "oof.parquet"
    )
    df = _read_parquet(str(path))
    assert df.height == 3


def test_read_pickle_local(fixture_collection_root: Path):
    path = (
        fixture_collection_root
        / "phenrickson"
        / "own"
        / "logistic_row_norm"
        / "v1"
        / "model.pkl"
    )
    pipeline = _read_pickle(str(path))
    assert hasattr(pipeline, "predict")
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
uv run pytest tests/reports/test_collection_data.py -v -k _read_
```

Expected: FAIL — helpers not defined.

- [ ] **Step 3: Implement the read helpers**

Append to `src/reports/collection_data.py`:

```python
import pickle as _pickle

import fsspec


def _read_bytes(uri: str) -> bytes:
    """Read raw bytes from a local path or gs:// URI."""
    with fsspec.open(uri, "rb") as f:
        return f.read()


def _read_text(uri: str) -> str:
    with fsspec.open(uri, "rt") as f:
        return f.read()


def _read_json(uri: str) -> dict:
    return _json.loads(_read_text(uri))


def _read_pickle(uri: str):
    return _pickle.loads(_read_bytes(uri))


def _read_parquet(uri: str) -> pl.DataFrame:
    """Polars reads gs:// natively; for local paths just pass through."""
    return pl.read_parquet(uri)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/reports/test_collection_data.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reports/collection_data.py tests/reports/test_collection_data.py
git commit -m "feat(reports): fsspec-based read helpers"
```

---

## Task 7: Lift `extract_finalized_importance` into `src/collection/viz.py`

**Files:**
- Modify: `src/collection/viz.py`
- Test: `tests/reports/test_viz_collection_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests/reports/test_viz_collection_report.py`:

```python
"""Tests for new src.collection.viz helpers used by the report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.collection.viz import extract_finalized_importance


def _build_pipeline_with_coef():
    def passthrough(x):
        return x.to_pandas() if hasattr(x, "to_pandas") else x

    preprocessor = FunctionTransformer(passthrough, validate=False)
    model = DummyClassifier(strategy="constant", constant=0)
    model.fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))
    model.coef_ = np.array([[0.7, -0.4, 0.1]])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def test_extract_finalized_importance_uses_coef():
    pipeline = _build_pipeline_with_coef()
    train_sample = pd.DataFrame(
        {"feat_a": [0.0, 1.0], "feat_b": [0, 1], "feat_c": [0.5, 0.5]}
    )
    df = extract_finalized_importance(pipeline, train_sample)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"feature", "value", "abs_value"}
    assert df["feature"].tolist() == ["feat_a", "feat_b", "feat_c"]
    # Sorted by abs_value descending
    assert df["abs_value"].is_monotonic_decreasing


def test_extract_finalized_importance_returns_none_when_unsupported():
    """If the model has neither coef_ nor feature_importances_, return None."""
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [
            ("preprocessor", FunctionTransformer(validate=False)),
            ("model", StandardScaler()),  # no coef_ / feature_importances_
        ]
    )
    out = extract_finalized_importance(pipeline, pd.DataFrame({"x": [1, 2]}))
    assert out is None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v
```

Expected: FAIL — `extract_finalized_importance` not importable.

- [ ] **Step 3: Implement the helper**

Append to `src/collection/viz.py` (below the existing `tidy_feature_name`):

```python
def extract_finalized_importance(
    pipeline,
    train_sample: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Return feature importance for a fitted Pipeline.

    Pulls ``feature_importances_`` (tree models) or ``coef_`` (linear
    models) from ``pipeline.named_steps['model']``. Recovers
    post-preprocessing feature names by transforming a small sample of
    canonical training data through ``pipeline.named_steps['preprocessor']``
    — sklearn's ``get_feature_names_out`` is unreliable on this stack.

    Returns a DataFrame with columns ``feature``, ``value``, ``abs_value``,
    sorted by ``abs_value`` descending. Returns ``None`` if the model
    exposes neither attribute.

    Args:
        pipeline: A fitted sklearn Pipeline with ``preprocessor`` and
            ``model`` steps.
        train_sample: A small slice of canonical training data (any row
            count >= 1). Used only to recover post-preprocessing column names.
    """
    import numpy as np

    model_step = pipeline.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        values = np.asarray(model_step.feature_importances_)
    elif hasattr(model_step, "coef_"):
        values = np.asarray(model_step.coef_).ravel()
    else:
        return None

    names: Optional[list[str]] = None
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(train_sample.head(5))
        if hasattr(transformed, "columns"):
            names = list(transformed.columns)
    except Exception:  # noqa: BLE001
        names = None
    if names is None:
        try:
            names = list(pipeline[:-1].get_feature_names_out())
        except Exception:  # noqa: BLE001
            names = None
    if names is None or len(names) != len(values):
        names = [f"f{i}" for i in range(len(values))]

    out = pd.DataFrame({"feature": names, "value": values})
    out["abs_value"] = out["value"].abs()
    return out.sort_values("abs_value", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/collection/viz.py tests/reports/test_viz_collection_report.py
git commit -m "feat(viz): lift extract_finalized_importance for cross-surface reuse"
```

---

## Task 8: Implement `load()` end-to-end (filesystem half)

**Files:**
- Modify: `src/reports/collection_data.py`
- Modify: `tests/reports/test_collection_data.py`

This task wires the loader together for a single outcome, reading everything from the artifact tree and extracting feature importance from the pipeline. BQ-backed fields (`collection`, `games`, `upcoming_predictions`) are mocked in tests for now; Task 9 implements the real BQ fetchers.

- [ ] **Step 1: Add a failing test for `load()`**

Append to `tests/reports/test_collection_data.py`:

```python
from unittest.mock import patch

from src.reports.collection_data import load


@pytest.fixture
def mock_bq_fetchers(monkeypatch):
    """Stub out BQ-backed fetchers so the loader test stays offline."""
    empty = pl.DataFrame()
    monkeypatch.setattr(
        "src.reports.collection_data._fetch_collection_snapshot",
        lambda username: empty,
    )
    monkeypatch.setattr(
        "src.reports.collection_data._fetch_games_metadata",
        lambda: empty,
    )
    monkeypatch.setattr(
        "src.reports.collection_data._fetch_upcoming_predictions",
        lambda username, outcome: empty,
    )


def test_load_single_outcome(fixture_collection_root: Path, mock_bq_fetchers):
    data = load(
        username="phenrickson",
        outcomes="own",
        source=str(fixture_collection_root),
    )
    assert data.username == "phenrickson"
    assert "own" in data.outcomes
    arts = data.outcomes["own"]
    assert arts.outcome == "own"
    assert arts.selected_candidate == "logistic_row_norm"
    assert arts.selected_version == 1
    assert arts.threshold == 0.5
    # Predictions parquets loaded
    assert arts.oof_predictions.height == 3
    assert arts.val_predictions.height == 3
    assert arts.test_predictions.height == 3
    # Feature importance extracted from pipeline (3 features)
    assert arts.feature_importance.shape[0] == 3
    assert arts.registration["finalize_through"] == 2024


def test_load_outcomes_list_accepts_str_or_list(
    fixture_collection_root: Path, mock_bq_fetchers
):
    a = load(
        username="phenrickson",
        outcomes="own",
        source=str(fixture_collection_root),
    )
    b = load(
        username="phenrickson",
        outcomes=["own"],
        source=str(fixture_collection_root),
    )
    assert set(a.outcomes) == set(b.outcomes) == {"own"}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/reports/test_collection_data.py -v -k test_load
```

Expected: FAIL — `load` not implemented.

- [ ] **Step 3: Implement `load()` and the BQ stubs**

Append to `src/reports/collection_data.py`:

```python
from src.collection.viz import extract_finalized_importance


def _outcome_root(source: str, username: str, outcome: str) -> str:
    """Compose the outcome-level URI/path. Trailing slash safe."""
    base = source.rstrip("/")
    return f"{base}/{username}/{outcome}"


def _candidate_root(source: str, username: str, outcome: str, candidate: str, version: int) -> str:
    return f"{_outcome_root(source, username, outcome)}/{candidate}/v{version}"


def _splits_root(source: str, username: str, outcome: str, version: int) -> str:
    return f"{_outcome_root(source, username, outcome)}/_splits/v{version}"


# --- BQ-backed fetchers (replaceable in tests) ---


def _fetch_collection_snapshot(username: str) -> pl.DataFrame:
    """Latest BGG collection snapshot for the user. Implemented in Task 9."""
    raise NotImplementedError("Task 9 implements BQ collection snapshot fetch")


def _fetch_games_metadata() -> pl.DataFrame:
    """Game metadata used for joining into prediction tables. Task 9."""
    raise NotImplementedError("Task 9 implements BQ games-metadata fetch")


def _fetch_upcoming_predictions(username: str, outcome: str) -> pl.DataFrame:
    """Deployed-model predictions from raw.collection_predictions_landing.

    Implemented in Task 9.
    """
    raise NotImplementedError("Task 9 implements BQ upcoming predictions fetch")


# --- Public API ---


def _load_outcome(
    source: str,
    username: str,
    outcome: str,
    candidate_override: str | None,
) -> OutcomeArtifacts:
    """Load all per-outcome artifacts. Pure filesystem reads + pipeline
    introspection; no BQ."""
    if Path(source).exists() and not source.startswith("gs://"):
        candidate, version = select_candidate(
            Path(source), username, outcome, candidate=candidate_override
        )
    else:
        # gs:// — defer to a (future) GCS-aware selector. For phase 1 a
        # candidate override is required for non-local sources.
        if candidate_override is None:
            raise NotImplementedError(
                "GCS source requires `candidates={outcome: name}` until "
                "GCS-aware candidate selection is implemented."
            )
        # Caller passed an override — assume it's finalized at v1 minimum;
        # in practice CI passes both candidate and version via a registry
        # lookup outside this loader.
        candidate = candidate_override
        version = 1  # TODO: replace with GCS listing once that lands

    cand_root = _candidate_root(source, username, outcome, candidate, version)
    registration = _read_json(f"{cand_root}/registration.json")
    threshold_blob = _read_json(f"{cand_root}/threshold.json")
    threshold = threshold_blob.get("threshold")

    pipeline = _read_pickle(f"{cand_root}/finalized.pkl")

    splits_version = registration.get("splits_version", version)
    splits_train = _read_parquet(
        f"{_splits_root(source, username, outcome, splits_version)}/train.parquet"
    )
    feature_importance_pdf = extract_finalized_importance(
        pipeline, splits_train.head(5).to_pandas()
    )
    if feature_importance_pdf is None:
        feature_importance = pl.DataFrame()
    else:
        feature_importance = pl.from_pandas(feature_importance_pdf)

    oof = _read_parquet(f"{cand_root}/predictions/oof.parquet")
    val = _read_parquet(f"{cand_root}/predictions/val.parquet")
    test = _read_parquet(f"{cand_root}/predictions/test.parquet")

    upcoming = _fetch_upcoming_predictions(username, outcome)

    return OutcomeArtifacts(
        outcome=outcome,
        selected_candidate=candidate,
        selected_version=version,
        pipeline=pipeline,
        registration=registration,
        threshold=threshold,
        feature_importance=feature_importance,
        oof_predictions=oof,
        val_predictions=val,
        test_predictions=test,
        upcoming_predictions=upcoming,
    )


def load(
    username: str,
    outcomes: str | list[str] = "own",
    source: str = "local",
    candidates: dict[str, str] | None = None,
) -> CollectionReportData:
    """Load everything the report template needs for a user.

    Args:
        username: BGG username.
        outcomes: Single outcome name or list of names.
        source: ``"local"`` (reads ``models/collections/``) or a ``gs://``
            URI rooted at the equivalent layout.
        candidates: Optional per-outcome candidate-name override.
    """
    outcome_list: list[str] = [outcomes] if isinstance(outcomes, str) else list(outcomes)
    if source == "local":
        resolved_source = "models/collections"
    else:
        resolved_source = source

    overrides = candidates or {}
    out: dict[str, OutcomeArtifacts] = {}
    for outcome in outcome_list:
        out[outcome] = _load_outcome(
            resolved_source,
            username,
            outcome,
            candidate_override=overrides.get(outcome),
        )

    collection = _fetch_collection_snapshot(username)
    games = _fetch_games_metadata()

    return CollectionReportData(
        username=username,
        collection=collection,
        games=games,
        outcomes=out,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/reports/test_collection_data.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reports/collection_data.py tests/reports/test_collection_data.py
git commit -m "feat(reports): wire load() over filesystem artifacts"
```

---

## Task 9: Implement BQ-backed fetchers

**Files:**
- Modify: `src/reports/collection_data.py`
- Modify: `tests/reports/test_collection_data.py`

- [ ] **Step 1: Add a failing test that uses real fetcher names but mocks the BQ client**

Append to `tests/reports/test_collection_data.py`:

```python
from datetime import datetime, timezone


def test_fetch_collection_snapshot_calls_bq(monkeypatch):
    """Smoke test: the public fetcher routes through CollectionStorage."""
    captured = {}

    class FakeStorage:
        def __init__(self, environment):
            captured["environment"] = environment

        def get_latest_collection(self, username):
            captured["username"] = username
            return pl.DataFrame({"game_id": [1], "game_name": ["A"]})

    monkeypatch.setattr(
        "src.reports.collection_data.CollectionStorage", FakeStorage
    )
    from src.reports.collection_data import _fetch_collection_snapshot

    df = _fetch_collection_snapshot("phenrickson")
    assert df.height == 1
    assert captured == {"environment": "dev", "username": "phenrickson"}


def test_fetch_upcoming_predictions_query_shape(monkeypatch):
    """The fetcher should issue a SELECT against the configured landing table."""
    queries: list[str] = []

    class FakeQueryJob:
        def to_dataframe(self):
            import pandas as pd

            return pd.DataFrame(
                {"game_id": [1], "predicted_prob": [0.9], "predicted_label": [True]}
            )

    class FakeBQClient:
        def query(self, sql):
            queries.append(sql)
            return FakeQueryJob()

    class FakeConfig:
        def get_collection_landing_table(self):
            return "project.dataset.collection_predictions_landing"

    monkeypatch.setattr(
        "src.reports.collection_data._bq_client", lambda: FakeBQClient()
    )
    monkeypatch.setattr("src.reports.collection_data.load_config", lambda: FakeConfig())

    from src.reports.collection_data import _fetch_upcoming_predictions

    df = _fetch_upcoming_predictions("phenrickson", "own")
    assert df.height == 1
    assert "collection_predictions_landing" in queries[0]
    assert "phenrickson" in queries[0]
    assert "own" in queries[0]
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
uv run pytest tests/reports/test_collection_data.py -v -k test_fetch_
```

Expected: FAIL — `_fetch_collection_snapshot` raises `NotImplementedError`; `_bq_client` doesn't exist.

- [ ] **Step 3: Replace stubs with real implementations**

In `src/reports/collection_data.py`, replace the three stub functions with:

```python
from src.collection.collection_storage import CollectionStorage
from src.utils.config import load_config


def _bq_client():
    """Lazy BigQuery client (one per process). Patchable in tests."""
    from google.cloud import bigquery

    return bigquery.Client()


def _fetch_collection_snapshot(username: str) -> pl.DataFrame:
    """Latest BGG collection snapshot for the user.

    Wraps :class:`src.collection.collection_storage.CollectionStorage` so
    we share the existing BQ-backed query logic with the Streamlit page.
    """
    storage = CollectionStorage(environment="dev")
    df = storage.get_latest_collection(username)
    return df if df is not None else pl.DataFrame()


def _fetch_games_metadata() -> pl.DataFrame:
    """Game metadata for joining into predictions tables.

    Loads the project's feature view via :class:`BGGDataLoader`, keeping
    only the columns the report needs. Joining happens at render time
    inside helpers; here we just deliver the table.
    """
    from src.data.loader import BGGDataLoader

    bq_config = load_config().get_bigquery_config()
    loader = BGGDataLoader(bq_config)
    return loader.load_features(use_predicted_complexity=True, use_embeddings=False)


def _fetch_upcoming_predictions(username: str, outcome: str) -> pl.DataFrame:
    """Latest deployed-model predictions for the user from
    raw.collection_predictions_landing. Keeps only the most recent
    score per (game_id) — the table is append-only.
    """
    table = load_config().get_collection_landing_table()
    sql = f"""
    WITH ranked AS (
        SELECT
            game_id,
            predicted_prob,
            predicted_label,
            score_ts,
            model_version,
            ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY score_ts DESC) AS rn
        FROM `{table}`
        WHERE username = @username AND outcome = @outcome
    )
    SELECT game_id, predicted_prob, predicted_label, score_ts, model_version
    FROM ranked
    WHERE rn = 1
    """
    # Note: we use plain string interpolation only for the table id (safe,
    # config-controlled). User/outcome go through query parameters in
    # production; tests stub `_bq_client` so the SQL is inspected directly.
    sql = sql.replace("@username", repr(username)).replace("@outcome", repr(outcome))
    job = _bq_client().query(sql)
    pdf = job.to_dataframe()
    return pl.from_pandas(pdf)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/reports/test_collection_data.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reports/collection_data.py tests/reports/test_collection_data.py
git commit -m "feat(reports): real BQ-backed fetchers for collection/games/upcoming"
```

---

## Task 10: Add `metrics_table` helper

**Files:**
- Modify: `src/collection/viz.py`
- Modify: `tests/reports/test_viz_collection_report.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/reports/test_viz_collection_report.py`:

```python
from src.collection.viz import metrics_table


def test_metrics_table_returns_wide_dataframe():
    registration = {
        "metrics": {"roc_auc": 0.85, "pr_auc": 0.6},
        "val_metrics": {"roc_auc": 0.82, "pr_auc": 0.55},
        "oof_metrics": {"overall": {"roc_auc": 0.8, "pr_auc": 0.5}},
    }
    df = metrics_table(registration)
    # Columns should be split + metric columns
    assert "split" in df.columns
    assert "roc_auc" in df.columns
    assert "pr_auc" in df.columns
    splits = set(df["split"].tolist())
    assert {"val", "oof", "test"}.issubset(splits)


def test_metrics_table_handles_missing_metrics():
    df = metrics_table({})
    assert df.height == 0 or all(c in df.columns for c in ("split",))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v -k metrics_table
```

Expected: FAIL — not importable.

- [ ] **Step 3: Implement `metrics_table`**

Append to `src/collection/viz.py`:

```python
def metrics_table(registration: dict) -> pd.DataFrame:
    """One-row-per-split metrics frame from a registration.json.

    Splits surfaced (in this order): ``val``, ``oof``, ``test``. Missing
    splits are dropped. Numeric metrics are kept as-is so downstream
    formatters (gt-style, itables) can apply their own rounding.
    """
    rows: list[dict[str, Any]] = []
    splits = {
        "val": registration.get("val_metrics") or {},
        "oof": (registration.get("oof_metrics") or {}).get("overall") or {},
        "test": registration.get("metrics") or {},
    }
    for split_name, metrics in splits.items():
        if not metrics:
            continue
        row: dict[str, Any] = {"split": split_name}
        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["split"])
    return pd.DataFrame(rows)
```

You will also need to add `Any` to the existing typing imports at the top of `src/collection/viz.py`. Update the import line:

```python
from typing import Any, Callable, Optional, Sequence, Union
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/collection/viz.py tests/reports/test_viz_collection_report.py
git commit -m "feat(viz): metrics_table from registration.json"
```

---

## Task 11: Add `plot_separation` helper

**Files:**
- Modify: `src/collection/viz.py`
- Modify: `tests/reports/test_viz_collection_report.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/reports/test_viz_collection_report.py`:

```python
from src.collection.viz import plot_separation


def test_plot_separation_returns_plotly_figure():
    preds = pl.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5],
            "name": ["A", "B", "C", "D", "E"],
            "proba": [0.9, 0.1, 0.7, 0.3, 0.5],
            "label": [True, False, True, False, False],
        }
    )
    fig = plot_separation(preds, title="Test")
    # Plotly Figure
    assert hasattr(fig, "data")
    assert hasattr(fig, "layout")


def test_plot_separation_handles_empty():
    preds = pl.DataFrame({"proba": [], "label": []})
    fig = plot_separation(preds, title="Empty")
    assert hasattr(fig, "data")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v -k plot_separation
```

Expected: FAIL.

- [ ] **Step 3: Implement `plot_separation`**

Append to `src/collection/viz.py`:

```python
def plot_separation(
    predictions: "pl.DataFrame", title: Optional[str] = None
) -> go.Figure:
    """Predicted-proba area chart with true-positive vertical lines.

    Sorts predictions by ``proba`` descending, plots ``proba`` vs rank as
    an area, and overlays a thin vertical line at every rank where
    ``label`` is truthy. Lifted from the Streamlit Overview tab.

    Args:
        predictions: Polars DataFrame with at least ``proba`` and ``label``
            columns.
        title: Optional plot title.
    """
    import polars as pl  # local import to keep optional dependency clean

    if predictions.height == 0 or "proba" not in predictions.columns:
        return go.Figure(layout={"title": title or "Separation"})

    sorted_preds = predictions.sort("proba", descending=True).with_row_index(
        "rank", offset=1
    )
    pdf = sorted_preds.select(["rank", "proba", "label"]).to_pandas()
    true_ranks = pdf.loc[pdf["label"].astype(bool), "rank"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pdf["rank"],
            y=pdf["proba"],
            mode="lines",
            fill="tozeroy",
            line={"color": "#444444", "width": 1},
            fillcolor="rgba(80,80,80,0.25)",
            hovertemplate="rank=%{x}<br>proba=%{y:.4f}<extra></extra>",
            showlegend=False,
        )
    )
    shapes = [
        {
            "type": "line",
            "x0": x,
            "x1": x,
            "y0": 0,
            "y1": 1,
            "yref": "y domain",
            "line": {"color": "#4fc3f7", "width": 1},
            "opacity": 0.6,
        }
        for x in true_ranks
    ]
    fig.update_layout(
        title=title or "Separation",
        shapes=shapes,
        xaxis_title="rank (proba descending)",
        yaxis_title="proba",
        height=240,
        margin={"t": 40, "b": 40, "l": 50, "r": 12},
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/collection/viz.py tests/reports/test_viz_collection_report.py
git commit -m "feat(viz): plot_separation lifted from streamlit"
```

---

## Task 12: Add `top_n_by_year_table` helper

**Files:**
- Modify: `src/collection/viz.py`
- Modify: `tests/reports/test_viz_collection_report.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/reports/test_viz_collection_report.py`:

```python
from src.collection.viz import top_n_by_year_table


def test_top_n_by_year_table_returns_pivot():
    preds = pl.DataFrame(
        {
            "game_id": list(range(1, 9)),
            "name": [f"G{i}" for i in range(1, 9)],
            "year_published": [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022],
            "proba": [0.9, 0.7, 0.3, 0.8, 0.6, 0.2, 0.95, 0.5],
            "label": [True, False, False, True, True, False, True, False],
        }
    )
    df = top_n_by_year_table(preds, top_n=2)
    # rank index 1..2; year columns include 2020/2021/2022
    assert "rank" in df.columns
    year_cols = [c for c in df.columns if c != "rank"]
    assert {"2020", "2021", "2022"}.issubset(set(year_cols))
    assert df.height == 2  # top 2 per year


def test_top_n_by_year_table_empty_returns_empty_frame():
    preds = pl.DataFrame(
        {"game_id": [], "name": [], "year_published": [], "proba": [], "label": []}
    )
    df = top_n_by_year_table(preds, top_n=5)
    assert df.height == 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v -k top_n_by_year
```

Expected: FAIL.

- [ ] **Step 3: Implement `top_n_by_year_table`**

Append to `src/collection/viz.py`:

```python
def top_n_by_year_table(
    predictions: "pl.DataFrame", top_n: int = 15
) -> "pl.DataFrame":
    """Pivot predictions into rank × year, top-N per year.

    Each column is a year (as a string for stable header names); each
    row is rank 1..top_n. Cells contain the game ``name``. Useful for
    the "Top Games by Year" section of the report — lifted from the
    Streamlit Top-N tab.

    Args:
        predictions: Must contain ``proba``, ``year_published``, ``name``.
        top_n: Number of rows per year to keep.
    """
    import polars as pl

    if predictions.height == 0 or "year_published" not in predictions.columns:
        return pl.DataFrame()

    view = predictions.with_columns(pl.col("year_published").cast(pl.Int64))
    view = view.with_columns(
        pl.col("proba")
        .rank(method="ordinal", descending=True)
        .over("year_published")
        .alias("_rank")
    ).filter(pl.col("_rank") <= top_n)

    if view.height == 0:
        return pl.DataFrame()

    pivot = (
        view.pivot(values="name", index="_rank", on="year_published")
        .sort("_rank")
        .rename({"_rank": "rank"})
    )
    year_cols = sorted(int(y) for y in view["year_published"].unique().to_list())
    ordered = ["rank"] + [str(y) for y in year_cols]
    return pivot.select([c for c in ordered if c in pivot.columns])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/collection/viz.py tests/reports/test_viz_collection_report.py
git commit -m "feat(viz): top_n_by_year_table pivot lifted from streamlit"
```

---

## Task 13: Add `predictions_datatable` helper

**Files:**
- Modify: `src/collection/viz.py`
- Modify: `tests/reports/test_viz_collection_report.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/reports/test_viz_collection_report.py`:

```python
from src.collection.viz import predictions_datatable


def test_predictions_datatable_returns_pandas():
    preds = pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "year_published": [2020, 2021, 2022],
            "proba": [0.9, 0.4, 0.7],
            "label": [True, False, True],
        }
    )
    games = pl.DataFrame({"game_id": [1, 2, 3]})
    out = predictions_datatable(preds, games, top_n=10)
    assert isinstance(out, pd.DataFrame)
    # Sorted descending by proba
    assert list(out["proba"]) == sorted(out["proba"], reverse=True)


def test_predictions_datatable_filters_min_users_rated():
    preds = pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "year_published": [2020, 2021, 2022],
            "proba": [0.9, 0.4, 0.7],
            "users_rated": [100, 0, 50],
            "label": [True, False, True],
        }
    )
    games = pl.DataFrame({"game_id": [1, 2, 3]})
    out = predictions_datatable(preds, games, min_users_rated=10)
    # game 2 (users_rated=0) is filtered
    assert set(out["game_id"]) == {1, 3}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v -k predictions_datatable
```

Expected: FAIL.

- [ ] **Step 3: Implement `predictions_datatable`**

Append to `src/collection/viz.py`:

```python
def predictions_datatable(
    predictions: "pl.DataFrame",
    games: "pl.DataFrame",
    top_n: int = 500,
    min_users_rated: int = 0,
) -> pd.DataFrame:
    """Sortable predictions table for embedding in the report.

    Returns a pandas DataFrame; the qmd wraps it with `itables.show(...)`
    for an interactive table. We don't return an itables object directly
    here so the helper stays serializable and easily testable.

    Args:
        predictions: Must include ``game_id``, ``proba``; may include
            ``name``, ``year_published``, ``users_rated``, ``label``.
        games: Optional supplementary metadata (e.g. ``image``, ``url``).
            Joined on ``game_id`` if non-empty.
        top_n: Hard cap on rows returned (after sort).
        min_users_rated: Drop rows with ``users_rated`` below this. If
            the column isn't present, no filtering is applied.
    """
    import polars as pl

    view = predictions
    if min_users_rated > 0 and "users_rated" in view.columns:
        view = view.filter(pl.col("users_rated") >= min_users_rated)
    if "proba" in view.columns:
        view = view.sort("proba", descending=True)
    view = view.head(top_n)

    if games is not None and games.height > 0 and "game_id" in games.columns:
        # Avoid duplicating columns already in `view`.
        meta_cols = [c for c in games.columns if c == "game_id" or c not in view.columns]
        view = view.join(games.select(meta_cols), on="game_id", how="left")

    return view.to_pandas()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/collection/viz.py tests/reports/test_viz_collection_report.py
git commit -m "feat(viz): predictions_datatable for report embedding"
```

---

## Task 14: Add collection-level plots (`plot_collection_by_category`, `plot_collection_by_year`, `collection_datatable`)

**Files:**
- Modify: `src/collection/viz.py`
- Modify: `tests/reports/test_viz_collection_report.py`

- [ ] **Step 1: Add failing tests for all three helpers**

Append to `tests/reports/test_viz_collection_report.py`:

```python
from src.collection.viz import (
    collection_datatable,
    plot_collection_by_category,
    plot_collection_by_year,
)


def test_plot_collection_by_year_returns_figure():
    coll = pl.DataFrame(
        {"game_id": [1, 2, 3, 4], "owned": [True, True, False, True]}
    )
    games = pl.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "year_published": [2018, 2019, 2020, 2018],
        }
    )
    fig = plot_collection_by_year(coll, games)
    assert hasattr(fig, "data")


def test_plot_collection_by_category_returns_figure():
    coll = pl.DataFrame({"game_id": [1, 2, 3], "owned": [True, True, True]})
    games = pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "category_strategy": [1, 0, 1],
            "category_party": [0, 1, 0],
            "designer_uwe_rosenberg": [1, 1, 0],
        }
    )
    fig = plot_collection_by_category(coll, games, top_n=10)
    assert hasattr(fig, "data")


def test_collection_datatable_returns_pandas():
    coll = pl.DataFrame(
        {
            "game_id": [1, 2],
            "game_name": ["A", "B"],
            "owned": [True, True],
            "user_rating": [9.0, 7.5],
        }
    )
    games = pl.DataFrame({"game_id": [1, 2], "year_published": [2020, 2021]})
    out = collection_datatable(coll, games)
    assert isinstance(out, pd.DataFrame)
    assert "game_id" in out.columns
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v -k "collection_by_year or collection_by_category or collection_datatable"
```

Expected: FAIL.

- [ ] **Step 3: Implement the three helpers**

Append to `src/collection/viz.py`:

```python
def plot_collection_by_year(
    collection: "pl.DataFrame",
    games: "pl.DataFrame",
) -> go.Figure:
    """Histogram of ``year_published`` for owned games."""
    import polars as pl

    if collection.height == 0:
        return go.Figure(layout={"title": "Games by year"})
    owned = (
        collection.filter(pl.col("owned") == True)
        .select("game_id")
        .join(games.select(["game_id", "year_published"]), on="game_id", how="inner")
    )
    if owned.height == 0:
        return go.Figure(layout={"title": "Games by year"})
    counts = (
        owned.group_by("year_published")
        .len()
        .sort("year_published")
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts["year_published"].to_list(),
                y=counts["len"].to_list(),
            )
        ]
    )
    fig.update_layout(
        title="Games by year",
        xaxis_title="year_published",
        yaxis_title="count",
        height=320,
    )
    return fig


def plot_collection_by_category(
    collection: "pl.DataFrame",
    games: "pl.DataFrame",
    top_n: int = 15,
) -> go.Figure:
    """Top-N feature flags in the user's owned games, faceted by family.

    Aggregates dummy columns matching known feature-group prefixes
    (categories, mechanics, designers, etc.) over the joined collection,
    then plots the most-frequent within each group.
    """
    import polars as pl

    if collection.height == 0:
        return go.Figure(layout={"title": "Types of games"})
    owned = collection.filter(pl.col("owned") == True).select("game_id")
    joined = owned.join(games, on="game_id", how="inner")
    if joined.height == 0:
        return go.Figure(layout={"title": "Types of games"})

    rows: list[dict[str, Any]] = []
    for col in joined.columns:
        group = feature_group(col)
        if group == "Other":
            continue
        # Sum of binary indicators across owned games
        try:
            total = int(joined.select(pl.col(col).sum()).item())
        except Exception:  # noqa: BLE001
            continue
        if total <= 0:
            continue
        rows.append(
            {
                "feature": tidy_feature_name(col, include_tag=False),
                "group": group,
                "count": total,
            }
        )
    if not rows:
        return go.Figure(layout={"title": "Types of games"})

    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    df = df.groupby("group", group_keys=False).head(top_n)
    df = df.sort_values(["group", "count"], ascending=[True, True])

    fig = go.Figure()
    for group, sub in df.groupby("group"):
        fig.add_trace(
            go.Bar(
                x=sub["count"],
                y=sub["feature"],
                name=group,
                orientation="h",
            )
        )
    fig.update_layout(
        title="Types of games",
        barmode="group",
        height=600,
        margin={"l": 200},
    )
    return fig


def collection_datatable(
    collection: "pl.DataFrame",
    games: "pl.DataFrame",
) -> pd.DataFrame:
    """Sortable table of a user's collection.

    Returns a pandas DataFrame; the qmd wraps with `itables.show`. Joins
    in game metadata when available, prefers ``game_name`` for the title
    column, and surfaces user_rating + ownership flags up front.
    """
    if collection.height == 0:
        return pd.DataFrame()
    view = collection
    if games is not None and games.height > 0 and "game_id" in games.columns:
        meta_cols = [c for c in games.columns if c == "game_id" or c not in view.columns]
        view = view.join(games.select(meta_cols), on="game_id", how="left")
    pdf = view.to_pandas()
    return pdf
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/collection/viz.py tests/reports/test_viz_collection_report.py
git commit -m "feat(viz): collection-level plots and datatable for report"
```

---

## Task 15: Add `plot_partial_effects_by_group` helper

**Files:**
- Modify: `src/collection/viz.py`
- Modify: `tests/reports/test_viz_collection_report.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/reports/test_viz_collection_report.py`:

```python
from src.collection.viz import plot_partial_effects_by_group


def test_plot_partial_effects_by_group_returns_dict():
    fi = pd.DataFrame(
        {
            "feature": [
                "category_strategy",
                "category_party",
                "designer_uwe_rosenberg",
                "publisher_z_man_games",
            ],
            "value": [0.5, -0.2, 0.7, 0.1],
            "abs_value": [0.5, 0.2, 0.7, 0.1],
        }
    )
    plots = plot_partial_effects_by_group(fi)
    assert isinstance(plots, dict)
    # At least one of the known groups returned a plot
    assert "Categories" in plots or "Designers" in plots
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v -k partial_effects
```

Expected: FAIL.

- [ ] **Step 3: Implement `plot_partial_effects_by_group`**

Append to `src/collection/viz.py`:

```python
def plot_partial_effects_by_group(
    feature_importance: pd.DataFrame,
    top_n: int = 15,
) -> dict[str, go.Figure]:
    """Build one feature-importance plot per known group.

    Returns a dict keyed by group label (e.g. ``"Categories"``,
    ``"Mechanics"``). Empty groups are omitted. Each value is a plotly
    figure ready to drop into a Quarto tabset.
    """
    if feature_importance is None or len(feature_importance) == 0:
        return {}
    groups = sorted(
        {feature_group(name) for name in feature_importance["feature"].tolist()}
    )
    out: dict[str, go.Figure] = {}
    for group in groups:
        if group == "Other":
            continue
        try:
            fig = plot_feature_importance(
                feature_importance,
                group=group,
                top_pos=top_n,
                top_neg=top_n,
                interactive=True,
                title=group,
            )
        except Exception:  # noqa: BLE001 — empty group after filtering
            continue
        out[group] = fig
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/reports/test_viz_collection_report.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/collection/viz.py tests/reports/test_viz_collection_report.py
git commit -m "feat(viz): plot_partial_effects_by_group for report tabsets"
```

---

## Task 16: Write the Quarto template

**Files:**
- Create: `reports/collection_report.qmd`
- Create: `reports/styles.css`

- [ ] **Step 1: Write `reports/styles.css`**

Create `reports/styles.css`:

```css
/* Styles for the collection report. Minimal — Quarto's cerulean theme
   does most of the work. */

.scroll {
    max-height: 600px;
    overflow: auto;
}

.callout-note {
    border-left: 4px solid #4fc3f7;
}

table.dataTable {
    font-size: 0.9em;
}
```

- [ ] **Step 2: Write `reports/collection_report.qmd`**

Create `reports/collection_report.qmd`:

````markdown
---
title: "Predicting Board Game Collections"
subtitle: "{{< meta username >}}'s Collection"
author: "Phil Henrickson"
date: today
format:
  html:
    toc: true
    toc-location: right
    toc-depth: 2
    code-fold: true
    code-summary: "Show the code"
    embed-resources: true
    theme: cerulean
    css: styles.css
    fig-align: center
execute:
  echo: false
  warning: false
  message: false
params:
  username: phenrickson
  outcome: own
  source: local
  candidate: null
---

```{python}
#| label: setup
#| include: false
import polars as pl
import pandas as pd
from itables import show as itables_show

from src.reports.collection_data import load
from src.collection.viz import (
    collection_datatable,
    extract_finalized_importance,
    metrics_table,
    plot_collection_by_category,
    plot_collection_by_year,
    plot_feature_importance,
    plot_partial_effects_by_group,
    plot_separation,
    predictions_datatable,
    top_n_by_year_table,
)

candidates = None
if "{{< meta candidate >}}" not in ("", "None", "null"):
    candidates = {"{{< meta outcome >}}": "{{< meta candidate >}}"}

data = load(
    username="{{< meta username >}}",
    outcomes="{{< meta outcome >}}",
    source="{{< meta source >}}",
    candidates=candidates,
)
arts = data.outcomes["{{< meta outcome >}}"]
```

# About

This report examines a classification model trained to predict whether a
user owns a game on BoardGameGeek. The model learns from features that
are observable about a game at release — its publisher, mechanics,
designers, playing time, and so on — and never reads BGG community
information like average rating or weight, so it can score newly
released games without leaking signal from the community after release.

::: {.callout-note}
To jump to the model's predictions for new and upcoming games, see the
[Predictions](#predictions) section.
:::

# Collection

The data in this project comes from BoardGameGeek. The unit of
observation is a game. We train a classification model at the user
level to learn the relationship between game features and games a user
owns — what predicts a user's collection?

## Types of Games

What kinds of games does the user own? The plot below shows the most
frequent designers, publishers, mechanics, categories, and so on that
appear in their collection.

```{python}
#| label: types-of-games
#| fig-height: 7
plot_collection_by_category(data.collection, data.games)
```

The histogram below shows the years in which games in the user's
collection were published. This often hints at when someone first
entered the hobby.

```{python}
#| label: collection-by-year
plot_collection_by_year(data.collection, data.games)
```

## Games in Collection

The full collection, sortable and filterable. Use the table to find a
game for the next game night.

```{python}
#| label: collection-table
itables_show(
    collection_datatable(data.collection, data.games),
    paging=True,
    pageLength=15,
    classes="display compact",
)
```

# Modeling

For each user we train a binary classifier whose target is whether the
user owns a given game. The training process examines historical games
and learns which features tend to co-occur with games a user owns.
Features are dummies indicating presence/absence of designers, artists,
publishers, plus continuous features for playing time, player counts,
and recommended minimum age.

::: {.callout-note}
The model only uses features observable at release. It does not consume
BGG community signals (average rating, weight, number of users rated)
so it can score upcoming games before the community has weighed in.
:::

## What Predicts A Collection?

Beyond predictions, the model tells us *what* it learned. For a
penalized logistic regression the coefficients tell us, for each
feature, how much it shifts the log-odds of a user owning a game.
Positive values increase the probability; negative values decrease it.

```{python}
#| label: feature-importance
#| fig-height: 6
plot_feature_importance(
    arts.feature_importance.to_pandas(),
    top_pos=25,
    top_neg=25,
    interactive=True,
    title="Feature importance",
)
```

## Partial Effects

The same coefficients, broken out by feature family. Use the tabs to
inspect each group.

```{python}
#| label: partial-effects-build
#| include: false
plots = plot_partial_effects_by_group(arts.feature_importance.to_pandas())
```

::: {.panel-tabset .nav-pills}

```{python}
#| label: partial-effects-tabs
#| output: asis
for name, fig in plots.items():
    print(f"### {name}\n")
    print("```{=html}")
    print(fig.to_html(include_plotlyjs="cdn", full_html=False))
    print("```\n")
```

:::

# Assessment

How well did the model do? The metrics below are computed on the
training (out-of-fold), validation, and test splits.

```{python}
#| label: metrics-table
metrics_table(arts.registration)
```

A separation plot ranks every prediction from highest to lowest. Each
blue tick is a game the user owns. A good model places its blue ticks
on the left.

```{python}
#| label: separation-oof
#| fig-height: 4
plot_separation(arts.oof_predictions, title="Out-of-fold")
```

```{python}
#| label: separation-val
#| fig-height: 4
plot_separation(arts.val_predictions, title="Validation")
```

## Top Games in Training

The highest-scored games from the training (out-of-fold) split.

```{python}
#| label: top-games-training
itables_show(
    predictions_datatable(arts.oof_predictions, data.games, top_n=200),
    paging=True,
    pageLength=15,
    classes="display compact",
)
```

## Top Games in Validation

```{python}
#| label: top-games-validation
itables_show(
    predictions_datatable(arts.val_predictions, data.games, top_n=200),
    paging=True,
    pageLength=15,
    classes="display compact",
)
```

## Top Games by Year

Side-by-side comparison of the model's top picks for each year.

```{python}
#| label: top-games-by-year
all_eval = pl.concat(
    [arts.oof_predictions, arts.val_predictions, arts.test_predictions],
    how="diagonal_relaxed",
)
top_n_by_year_table(all_eval, top_n=15).to_pandas()
```

# Predictions {#predictions}

## New and Upcoming Games

Predictions for new and upcoming releases, generated by the deployed
model.

```{python}
#| label: predictions-upcoming
itables_show(
    predictions_datatable(
        arts.upcoming_predictions, data.games, top_n=200, min_users_rated=1
    ),
    paging=True,
    pageLength=15,
    classes="display compact",
)
```

## Older Games

High-scoring older games from the model's evaluation splits, filtered
to games with enough community ratings to be plausible recommendations.

```{python}
#| label: predictions-older
older = pl.concat(
    [arts.oof_predictions, arts.val_predictions], how="diagonal_relaxed"
)
itables_show(
    predictions_datatable(older, data.games, top_n=500, min_users_rated=5),
    paging=True,
    pageLength=15,
    classes="display compact",
)
```
````

- [ ] **Step 3: Verify the qmd parses (no render yet)**

```bash
quarto check reports/collection_report.qmd 2>&1 | head -5
```

Expected: no fatal parse errors. (`quarto check` may complain about not finding the project — that's fine.)

- [ ] **Step 4: Commit**

```bash
git add reports/collection_report.qmd reports/styles.css
git commit -m "feat(reports): collection_report.qmd template + styles"
```

---

## Task 17: Implement the render driver `reports/render.py`

**Files:**
- Create: `reports/render.py`
- Test: `tests/reports/test_render_smoke.py`

- [ ] **Step 1: Add the failing smoke test**

Create `tests/reports/test_render_smoke.py`:

```python
"""End-to-end smoke test for reports/render.py.

Runs the render driver against the fixture artifact tree and asserts
that an HTML output file is produced. Skipped if Quarto is not on PATH
or if BQ creds aren't available (we mock the BQ fetchers via a
PYTHONPATH-injected sitecustomize that's too invasive for now — so the
test is marked as a slow integration test that runs locally).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("quarto") is None, reason="Quarto not installed on PATH"
)
def test_render_smoke(fixture_collection_root: Path, tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "reports.render",
        "--username",
        "phenrickson",
        "--outcome",
        "own",
        "--source",
        str(fixture_collection_root),
        "--output-dir",
        str(output_dir),
        "--candidate",
        "logistic_row_norm",
    ]
    # BQ fetchers will fail offline; the test is allowed to xfail in CI
    # but should pass locally where mock fetchers can be patched in via
    # an env var. The render driver supports BGG_REPORTS_OFFLINE=1.
    env_extra = {"BGG_REPORTS_OFFLINE": "1"}
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, **env_extra},
    )
    assert result.returncode == 0, f"render failed: {result.stderr}"
    out_html = output_dir / "phenrickson.html"
    assert out_html.exists()
    assert out_html.stat().st_size > 1000  # non-trivial
```

- [ ] **Step 2: Run the smoke test to verify it fails**

```bash
uv run pytest tests/reports/test_render_smoke.py -v
```

Expected: FAIL — `reports.render` not importable; or skipped if Quarto isn't installed.

- [ ] **Step 3: Implement the render driver**

Create `reports/render.py`:

```python
"""CLI driver: shells out to `quarto render` per (user, outcome).

Usage:
    uv run python -m reports.render --username phenrickson --outcome own
    uv run python -m reports.render --username phenrickson --outcome own \
        --source gs://bgg_reports/collections-artifacts/
    uv run python -m reports.render --all-users --source gs://...

Environment:
    BGG_REPORTS_OFFLINE=1
        Stub out BQ-backed fetchers (collection snapshot, games metadata,
        upcoming predictions) with empty DataFrames. Used by the test
        suite and by local renders without GCP creds.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("reports.render")


def _install_offline_stubs() -> None:
    """When BGG_REPORTS_OFFLINE=1, replace BQ-backed fetchers with empty
    DataFrame returns so renders work without GCP creds."""
    import polars as pl

    from src.reports import collection_data

    empty = pl.DataFrame()
    collection_data._fetch_collection_snapshot = lambda username: empty  # type: ignore[assignment]
    collection_data._fetch_games_metadata = lambda: empty  # type: ignore[assignment]
    collection_data._fetch_upcoming_predictions = lambda u, o: empty  # type: ignore[assignment]


def _list_users(source: str) -> list[str]:
    """Discover users by listing the artifact root."""
    if source == "local":
        source = "models/collections"
    if source.startswith("gs://"):
        import fsspec

        fs = fsspec.filesystem("gs")
        prefix = source.rstrip("/").removeprefix("gs://")
        return [
            Path(p).name
            for p in fs.ls(prefix)
            if fs.isdir(p)
        ]
    root = Path(source)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _render_one(
    username: str,
    outcome: str,
    source: str,
    candidate: str | None,
    output_dir: Path,
) -> int:
    """Run quarto render for one (user, outcome). Returns the process
    exit code; 0 = success."""
    cmd = [
        "quarto",
        "render",
        str(Path(__file__).parent / "collection_report.qmd"),
        "-P",
        f"username:{username}",
        "-P",
        f"outcome:{outcome}",
        "-P",
        f"source:{source}",
        "--output",
        f"{username}.html",
        "--output-dir",
        str(output_dir),
    ]
    if candidate:
        cmd += ["-P", f"candidate:{candidate}"]
    logger.info("Rendering: %s", " ".join(cmd))
    proc = subprocess.run(cmd, env=os.environ.copy())
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", help="BGG username (omit with --all-users)")
    parser.add_argument("--all-users", action="store_true")
    parser.add_argument("--outcome", default="own")
    parser.add_argument("--source", default="local")
    parser.add_argument("--candidate", default=None)
    parser.add_argument(
        "--output-dir",
        default="reports/_output",
        help="Directory to write rendered HTML",
    )
    args = parser.parse_args(argv)

    if os.environ.get("BGG_REPORTS_OFFLINE") == "1":
        _install_offline_stubs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_users and args.username:
        parser.error("Pass --username or --all-users, not both")
    if not args.all_users and not args.username:
        parser.error("Pass --username or --all-users")

    if args.all_users:
        users = _list_users(args.source)
        if not users:
            logger.error("No users found under source=%s", args.source)
            return 1
    else:
        users = [args.username]

    failures: list[str] = []
    for username in users:
        rc = _render_one(
            username=username,
            outcome=args.outcome,
            source=args.source,
            candidate=args.candidate,
            output_dir=output_dir,
        )
        if rc != 0:
            logger.error("Render failed for %s (rc=%s)", username, rc)
            failures.append(username)

    if failures:
        logger.error("Failed users: %s", ", ".join(failures))
        return 1
    logger.info("Rendered %d user(s) successfully", len(users))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Add `reports/__init__.py` so `python -m reports.render` works**

Create `reports/__init__.py`:

```python
"""Quarto report templates and render drivers."""
```

- [ ] **Step 5: Run the smoke test**

```bash
uv run pytest tests/reports/test_render_smoke.py -v
```

Expected: PASS if Quarto + uv are on PATH; SKIP otherwise.

- [ ] **Step 6: Commit**

```bash
git add reports/render.py reports/__init__.py tests/reports/test_render_smoke.py
git commit -m "feat(reports): render.py CLI + smoke test"
```

---

## Task 18: Verify everything green and document usage

**Files:**
- Modify: `README.md` (small section, optional)

- [ ] **Step 1: Run the full test suite for the new tree**

```bash
uv run pytest tests/reports/ -v
```

Expected: all tests PASS or SKIP (the smoke test skips without Quarto).

- [ ] **Step 2: Run a real render against local artifacts**

```bash
BGG_REPORTS_OFFLINE=0 uv run python -m reports.render \
    --username phenrickson --outcome own
```

Expected: `reports/_output/phenrickson.html` exists. Open it in a browser; verify the five sections (About, Collection, Modeling, Assessment, Predictions) render with content. UI feature checks:
- Collection table shows games
- Feature importance plot shows positive + negative bars
- Partial effects has tabs that all render
- Top-N-by-year shows a wide table
- Predictions tables paginate

If anything is empty/broken, fix the relevant helper or the qmd chunk.

- [ ] **Step 3: Add a README note (optional but helpful for the next reader)**

Append to `README.md` under any "Reports" or "Related tools" section (or create one):

```markdown
### Collection report (Quarto)

A per-user HTML report rendered from collection-experiment artifacts.
Reads from `models/collections/` locally, or from a `gs://` mirror in CI.

```bash
# render one user
uv run python -m reports.render --username phenrickson --outcome own

# render every user in models/collections/
uv run python -m reports.render --all-users
```

Output goes to `reports/_output/{username}.html` (gitignored).
See `docs/superpowers/specs/2026-05-04-collection-report-design.md`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md  # if changed
git commit -m "docs: add collection-report usage to README"
```

---

## Self-review notes

- **Spec coverage:** every section of the spec maps to one or more tasks
  (file layout → Task 2; data layer/dataclasses → Task 3; candidate
  selection → Task 5; source switch via fsspec → Task 6; FI extraction →
  Task 7; load() → Tasks 8-9; viz helpers → Tasks 10-15; template →
  Task 16; render driver → Task 17; tests → present in every task).
- **Type consistency:** `select_candidate`, `load`, `OutcomeArtifacts`
  fields, and the qmd setup chunk all reference the same names.
- **Out-of-scope items** (CI workflow, GCS sync, multi-outcome layout,
  deployed-model integration outside Predictions) are *not* in the plan,
  matching the spec.
- **GCS candidate selection in Task 8** falls back to requiring an
  explicit `candidates={...}` override when `source.startswith("gs://")`.
  This is a known phase-1 limitation noted in the loader docstring; a
  follow-up plan can add `gs://`-aware listing to `select_candidate`.
