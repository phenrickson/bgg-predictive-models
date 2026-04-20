# Collection Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `src/collection/` to support multiple user-specific outcome models (own, ever_owned, rated, rating, love) driven by a new `Outcomes` config abstraction. Scope is training-side only.

**Architecture:** Six modules with clean responsibilities: Loader, Storage (BQ raw), ArtifactStorage (GCS artifacts), Processor, Splitter, Model. New `Outcomes` module defines how each outcome's label is produced, consumed by Processor (labels), Splitter (classification vs regression strategy), Model (task dispatch). Pipeline orchestrates: process once per user, then loop over outcomes.

**Tech Stack:** Python 3.12, polars, scikit-learn, LightGBM, CatBoost, pytest, BigQuery, Google Cloud Storage.

**Design spec:** [docs/specs/2026-04-20-collection-modules-design.md](../specs/2026-04-20-collection-modules-design.md)

---

## File Structure

**Files created:**
- `src/collection/outcomes.py` — `OutcomeDefinition` dataclass + config loader + `apply_outcome()` labeling function
- `tests/test_collection_outcomes.py` — unit tests for the Outcomes module
- `tests/test_collection_splitter.py` — regression tests for classification splitter; smoke test for regression splitter

**Files modified:**
- `config.yaml` — add `collection.outcomes` section declaring the 5 outcomes
- `src/collection/collection_processor.py` — replaced with new Processor (join + generic prep, outcome-agnostic)
- `src/collection/collection_split.py` — refactor to dispatch on `outcome.task`; split `SplitConfig` into classification/regression variants
- `src/collection/collection_model.py` — refactor to dispatch on `outcome.task`; split `ModelConfig` into classification/regression variants; add regression path
- `src/collection/collection_artifact_storage.py` — add `outcome` parameter to all save/load methods; update GCS path layout
- `src/collection/collection_pipeline.py` — loop over outcomes; invoke `apply_outcome` + Splitter + Model per outcome
- `src/collection/__init__.py` — export new public API
- `Makefile` — add `train-collection` / `refresh-collection` / `collection-status` targets

**Files deleted:**
- `src/collection/collection_integration.py` — functionality absorbed into the new `collection_processor.py`

---

## Task 1: Add `collection.outcomes` section to config.yaml

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Locate the end of the existing `models:` section in `config.yaml`**

Run: `grep -n "^[a-z]" config.yaml | head -20`

Identify the top-level keys. We will add `collection:` as a new top-level section.

- [ ] **Step 2: Add the `collection` section to `config.yaml`**

Append this block to `config.yaml` (at the top level, not nested under another key):

```yaml
collection:
  outcomes:
    own:
      task: classification
      label_from: owned
    ever_owned:
      task: classification
      label_from: {any_of: [owned, prev_owned]}
    rated:
      task: classification
      label_from: {column: user_rating, predicate: "> 0"}
    rating:
      task: regression
      label_from: user_rating
      require: "user_rating > 0"
    love:
      task: classification
      label_from: {column: user_rating, predicate: ">= 8"}
      require: "user_rating > 0"
```

- [ ] **Step 3: Verify YAML parses**

Run: `uv run python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['collection']['outcomes'].keys())"`
Expected output: `dict_keys(['own', 'ever_owned', 'rated', 'rating', 'love'])`

- [ ] **Step 4: Commit**

```bash
git add config.yaml
git commit -m "feat(collection): declare outcomes in config.yaml"
```

---

## Task 2: Create Outcomes module — dataclass and config loader

**Files:**
- Create: `src/collection/outcomes.py`
- Create: `tests/test_collection_outcomes.py`

- [ ] **Step 1: Write the failing tests for `OutcomeDefinition` + `load_outcomes`**

Create `tests/test_collection_outcomes.py`:

```python
"""Tests for src.collection.outcomes."""

import pytest
from src.collection.outcomes import (
    OutcomeDefinition,
    DirectColumnRule,
    AnyOfRule,
    PredicateRule,
    load_outcomes,
)


SAMPLE_CONFIG = {
    "collection": {
        "outcomes": {
            "own": {"task": "classification", "label_from": "owned"},
            "ever_owned": {
                "task": "classification",
                "label_from": {"any_of": ["owned", "prev_owned"]},
            },
            "rated": {
                "task": "classification",
                "label_from": {"column": "user_rating", "predicate": "> 0"},
            },
            "rating": {
                "task": "regression",
                "label_from": "user_rating",
                "require": "user_rating > 0",
            },
            "love": {
                "task": "classification",
                "label_from": {"column": "user_rating", "predicate": ">= 8"},
                "require": "user_rating > 0",
            },
        }
    }
}


def test_load_outcomes_returns_five_definitions():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    assert set(outcomes.keys()) == {"own", "ever_owned", "rated", "rating", "love"}


def test_own_is_direct_column_rule():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    own = outcomes["own"]
    assert own.task == "classification"
    assert isinstance(own.label_rule, DirectColumnRule)
    assert own.label_rule.column == "owned"
    assert own.require is None


def test_ever_owned_is_any_of_rule():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    ever = outcomes["ever_owned"]
    assert isinstance(ever.label_rule, AnyOfRule)
    assert ever.label_rule.columns == ["owned", "prev_owned"]


def test_rated_is_predicate_rule():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    rated = outcomes["rated"]
    assert isinstance(rated.label_rule, PredicateRule)
    assert rated.label_rule.column == "user_rating"
    assert rated.label_rule.operator == ">"
    assert rated.label_rule.value == 0


def test_rating_is_regression_with_require():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    r = outcomes["rating"]
    assert r.task == "regression"
    assert isinstance(r.label_rule, DirectColumnRule)
    assert r.require == "user_rating > 0"


def test_love_has_ge_predicate():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    love = outcomes["love"]
    assert isinstance(love.label_rule, PredicateRule)
    assert love.label_rule.operator == ">="
    assert love.label_rule.value == 8


def test_invalid_task_raises():
    bad_config = {"collection": {"outcomes": {"x": {"task": "clustering", "label_from": "y"}}}}
    with pytest.raises(ValueError, match="task"):
        load_outcomes(bad_config)


def test_invalid_predicate_operator_raises():
    bad_config = {
        "collection": {
            "outcomes": {
                "x": {
                    "task": "classification",
                    "label_from": {"column": "y", "predicate": "~= 5"},
                }
            }
        }
    }
    with pytest.raises(ValueError, match="operator"):
        load_outcomes(bad_config)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run -m pytest tests/test_collection_outcomes.py -v`
Expected: all tests fail with `ModuleNotFoundError: No module named 'src.collection.outcomes'`

- [ ] **Step 3: Implement `src/collection/outcomes.py`**

Create `src/collection/outcomes.py`:

```python
"""Outcome definitions for user collection models.

An OutcomeDefinition describes how to construct a training label from a
user's collection dataframe. Defined declaratively in config.yaml under
collection.outcomes; parsed into dataclasses here.

Consumers:
- CollectionProcessor / apply_outcome: constructs the label column
- CollectionSplitter: dispatches to classification vs regression strategy
- CollectionModel: dispatches to classification vs regression training path
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

import polars as pl


VALID_OPERATORS = {">", ">=", "<", "<=", "==", "!="}
VALID_TASKS = {"classification", "regression"}


@dataclass(frozen=True)
class DirectColumnRule:
    """Label is the value of a single column."""
    column: str


@dataclass(frozen=True)
class AnyOfRule:
    """Label is the boolean OR of multiple columns."""
    columns: List[str]


@dataclass(frozen=True)
class PredicateRule:
    """Label is the result of applying a comparison operator to a column."""
    column: str
    operator: str
    value: float


LabelRule = Union[DirectColumnRule, AnyOfRule, PredicateRule]


@dataclass(frozen=True)
class OutcomeDefinition:
    """Declarative description of one outcome model."""
    name: str
    task: Literal["classification", "regression"]
    label_rule: LabelRule
    require: str | None = None


def _parse_predicate(predicate_str: str) -> tuple[str, float]:
    """Parse 'operator value' string (e.g. '>= 8') into (operator, value)."""
    match = re.match(r"^\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$", predicate_str)
    if not match:
        raise ValueError(
            f"Invalid predicate {predicate_str!r}. "
            f"Expected format '<operator> <number>' with operator in {sorted(VALID_OPERATORS)}"
        )
    operator, value_str = match.groups()
    if operator not in VALID_OPERATORS:
        raise ValueError(f"Invalid operator {operator!r}. Valid operators: {sorted(VALID_OPERATORS)}")
    return operator, float(value_str)


def _parse_label_rule(spec: Any) -> LabelRule:
    """Parse one of the four primitives into a LabelRule dataclass."""
    if isinstance(spec, str):
        return DirectColumnRule(column=spec)
    if isinstance(spec, dict):
        if "any_of" in spec:
            cols = spec["any_of"]
            if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
                raise ValueError(f"any_of must be a list of column names, got {cols!r}")
            return AnyOfRule(columns=list(cols))
        if "column" in spec and "predicate" in spec:
            operator, value = _parse_predicate(spec["predicate"])
            return PredicateRule(column=spec["column"], operator=operator, value=value)
    raise ValueError(f"Unrecognized label_from shape: {spec!r}")


def _parse_outcome(name: str, entry: Dict[str, Any]) -> OutcomeDefinition:
    task = entry.get("task")
    if task not in VALID_TASKS:
        raise ValueError(f"Outcome {name!r} has invalid task {task!r}. Valid: {sorted(VALID_TASKS)}")
    label_from = entry.get("label_from")
    if label_from is None:
        raise ValueError(f"Outcome {name!r} missing required 'label_from'")
    label_rule = _parse_label_rule(label_from)
    require = entry.get("require")
    if require is not None and not isinstance(require, str):
        raise ValueError(f"Outcome {name!r} 'require' must be a string, got {require!r}")
    return OutcomeDefinition(name=name, task=task, label_rule=label_rule, require=require)


def load_outcomes(config: Dict[str, Any]) -> Dict[str, OutcomeDefinition]:
    """Parse config.collection.outcomes into a registry of OutcomeDefinitions."""
    outcomes_section = config.get("collection", {}).get("outcomes", {})
    if not outcomes_section:
        raise ValueError("config.collection.outcomes is empty or missing")
    return {name: _parse_outcome(name, entry) for name, entry in outcomes_section.items()}


def _apply_require(df: pl.DataFrame, require: str) -> pl.DataFrame:
    """Filter rows by a 'col op value' expression."""
    match = re.match(r"^\s*(\w+)\s+(>=|<=|==|!=|>|<)\s+(-?\d+(?:\.\d+)?)\s*$", require)
    if not match:
        raise ValueError(f"Invalid require expression {require!r}")
    column, operator, value_str = match.groups()
    value = float(value_str)
    ops = {
        ">": lambda c, v: c > v,
        ">=": lambda c, v: c >= v,
        "<": lambda c, v: c < v,
        "<=": lambda c, v: c <= v,
        "==": lambda c, v: c == v,
        "!=": lambda c, v: c != v,
    }
    return df.filter(ops[operator](pl.col(column), value))


def _apply_label_rule(df: pl.DataFrame, rule: LabelRule) -> pl.DataFrame:
    """Add a 'label' column from the given rule."""
    if isinstance(rule, DirectColumnRule):
        return df.with_columns(pl.col(rule.column).alias("label"))
    if isinstance(rule, AnyOfRule):
        expr = pl.col(rule.columns[0]).cast(pl.Boolean)
        for c in rule.columns[1:]:
            expr = expr | pl.col(c).cast(pl.Boolean)
        return df.with_columns(expr.alias("label"))
    if isinstance(rule, PredicateRule):
        ops = {
            ">": lambda c, v: c > v,
            ">=": lambda c, v: c >= v,
            "<": lambda c, v: c < v,
            "<=": lambda c, v: c <= v,
            "==": lambda c, v: c == v,
            "!=": lambda c, v: c != v,
        }
        return df.with_columns(ops[rule.operator](pl.col(rule.column), rule.value).alias("label"))
    raise TypeError(f"Unknown rule type: {type(rule)!r}")


def apply_outcome(df: pl.DataFrame, outcome: OutcomeDefinition) -> pl.DataFrame:
    """Apply the outcome's require filter (if any), then add a label column."""
    filtered = df if outcome.require is None else _apply_require(df, outcome.require)
    return _apply_label_rule(filtered, outcome.label_rule)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run -m pytest tests/test_collection_outcomes.py -v`
Expected: all 8 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/collection/outcomes.py tests/test_collection_outcomes.py
git commit -m "feat(collection): add Outcomes module with 4-primitive config schema"
```

---

## Task 3: Add `apply_outcome` label-application tests and verify

**Files:**
- Modify: `tests/test_collection_outcomes.py`

- [ ] **Step 1: Append label-application tests**

Add to the end of `tests/test_collection_outcomes.py`:

```python
import polars as pl
from src.collection.outcomes import apply_outcome


def _fixture_df() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": [1, 2, 3, 4, 5],
        "owned": [True, False, False, True, False],
        "prev_owned": [False, True, False, False, False],
        "user_rating": [8.0, 0.0, 9.5, 7.0, 0.0],
    })


def test_apply_outcome_own_binary():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    out = apply_outcome(_fixture_df(), outcomes["own"])
    assert out["label"].to_list() == [True, False, False, True, False]
    assert len(out) == 5  # no require filter


def test_apply_outcome_ever_owned_any_of():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    out = apply_outcome(_fixture_df(), outcomes["ever_owned"])
    assert out["label"].to_list() == [True, True, False, True, False]


def test_apply_outcome_rated_predicate():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    out = apply_outcome(_fixture_df(), outcomes["rated"])
    assert out["label"].to_list() == [True, False, True, True, False]


def test_apply_outcome_rating_regression_with_require():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    out = apply_outcome(_fixture_df(), outcomes["rating"])
    # require drops user_rating <= 0 (rows 2 and 5)
    assert out["game_id"].to_list() == [1, 3, 4]
    assert out["label"].to_list() == [8.0, 9.5, 7.0]


def test_apply_outcome_love_predicate_with_require():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    out = apply_outcome(_fixture_df(), outcomes["love"])
    # require drops rows 2 and 5; love is user_rating >= 8
    assert out["game_id"].to_list() == [1, 3, 4]
    assert out["label"].to_list() == [True, True, False]
```

- [ ] **Step 2: Run tests**

Run: `uv run -m pytest tests/test_collection_outcomes.py -v`
Expected: all 13 tests pass (8 from Task 2 + 5 new)

- [ ] **Step 3: Commit**

```bash
git add tests/test_collection_outcomes.py
git commit -m "test(collection): add apply_outcome label application tests"
```

---

## Task 4: Replace `collection_processor.py` with new Processor

**Files:**
- Modify (rewrite): `src/collection/collection_processor.py`
- Delete: `src/collection/collection_integration.py`

The new Processor absorbs the join-with-game-universe work from `collection_integration.py` plus the subtype filter from the thin existing `collection_processor.py`. It is outcome-agnostic: produces one unlabeled joined dataframe per user.

- [ ] **Step 1: Read current `collection_integration.py` to capture the join logic**

Run: `cat src/collection/collection_integration.py`
Note: the existing join uses `BGGDataLoader.load_data()` + polars `.join()`. For this refactor we keep the same approach (in-memory join via `BGGDataLoader`), not SQL — simpler and `BGGDataLoader` is a thin SELECT anyway.

- [ ] **Step 2: Rewrite `src/collection/collection_processor.py`**

Replace the contents of `src/collection/collection_processor.py` with:

```python
"""Process raw user collection into a joined, game-universe-aware dataframe.

Outcome-agnostic: produces one unlabeled dataframe per user. Labeling is
applied downstream via src.collection.outcomes.apply_outcome.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import polars as pl

from src.collection.collection_storage import CollectionStorage
from src.data.loader import BGGDataLoader
from src.utils.config import BigQueryConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for collection processing."""
    games_only: bool = True
    """If True, filter out non-boardgame subtypes."""


class CollectionProcessor:
    """Join raw user collection with game universe features.

    Outcome-agnostic: returns a single unlabeled dataframe containing all
    games the user has any relationship to (owned, prev_owned, rated, etc.),
    joined with the game universe feature set. Labeling is applied by the
    pipeline downstream via apply_outcome().
    """

    def __init__(
        self,
        config: BigQueryConfig,
        environment: str = "dev",
        processor_config: Optional[ProcessorConfig] = None,
    ):
        self.bq_config = config
        self.environment = environment
        self.processor_config = processor_config or ProcessorConfig()
        self.data_loader = BGGDataLoader(config)
        self.storage = CollectionStorage(environment=environment)

    def process(self, username: str) -> Optional[pl.DataFrame]:
        """Produce the unlabeled, joined dataframe for one user.

        Returns None if the user has no stored collection.
        """
        logger.info(f"Processing collection for user '{username}'")

        collection_df = self.storage.get_latest_collection(username)
        if collection_df is None:
            logger.error(f"No collection found for user '{username}'")
            return None

        if self.processor_config.games_only and "subtype" in collection_df.columns:
            before = len(collection_df)
            collection_df = collection_df.filter(pl.col("subtype") == "boardgame")
            logger.info(f"Filtered to boardgames: {before} -> {len(collection_df)}")

        logger.info("Loading game universe features from warehouse")
        features_df = self.data_loader.load_data()

        logger.info("Joining collection with game features")
        joined = collection_df.join(features_df, on="game_id", how="left", suffix="_features")

        logger.info(
            f"Processed {len(joined)} rows × {len(joined.columns)} columns for '{username}'"
        )
        return joined
```

- [ ] **Step 3: Delete `collection_integration.py`**

```bash
git rm src/collection/collection_integration.py
```

- [ ] **Step 4: Update any stale imports**

Run: `grep -rn "collection_integration\|CollectionIntegration" --include="*.py" src/ tests/`
Any match (outside `__pycache__`) is a stale reference. If the current pipeline or CLI imports `CollectionIntegration`, update to import `CollectionProcessor` from `src.collection.collection_processor`. Matches in `src/debug/` can be left for now (debug scripts are not in the critical path).

Expected after updates: no non-debug Python file imports `collection_integration`.

- [ ] **Step 5: Verify module imports cleanly**

Run: `uv run python -c "from src.collection.collection_processor import CollectionProcessor, ProcessorConfig; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/collection/collection_processor.py
git commit -m "refactor(collection): replace Processor with outcome-agnostic joiner; delete collection_integration"
```

---

## Task 5: Split `collection_split.py` — classification vs regression with dispatcher

**Files:**
- Modify: `src/collection/collection_split.py`
- Create: `tests/test_collection_splitter.py`

- [ ] **Step 1: Write the splitter tests (classification regression + regression smoke test)**

Create `tests/test_collection_splitter.py`:

```python
"""Tests for CollectionSplitter: classification and regression strategies."""

import polars as pl
import pytest

from src.collection.outcomes import OutcomeDefinition, DirectColumnRule
from src.collection.collection_split import (
    CollectionSplitter,
    ClassificationSplitConfig,
    RegressionSplitConfig,
)


def _universe_df() -> pl.DataFrame:
    # 20 games, varying popularity and year
    return pl.DataFrame({
        "game_id": list(range(1, 21)),
        "users_rated": [100 * i for i in range(1, 21)],
        "year_published": [2000 + (i % 10) for i in range(1, 21)],
    })


def _labeled_classification_df() -> pl.DataFrame:
    # 10 owned games
    return pl.DataFrame({
        "game_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "users_rated": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "year_published": [2005] * 10,
        "label": [True] * 10,
    })


def _labeled_regression_df() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "users_rated": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "year_published": [2005] * 10,
        "label": [6.0, 7.0, 8.0, 9.0, 10.0, 6.5, 7.5, 8.5, 9.5, 7.0],
    })


def _outcome(name: str, task: str) -> OutcomeDefinition:
    return OutcomeDefinition(
        name=name, task=task, label_rule=DirectColumnRule(column="label"), require=None
    )


def test_classification_split_returns_three_splits_with_negatives():
    splitter = CollectionSplitter(
        universe_df=_universe_df(),
        classification_config=ClassificationSplitConfig(
            negative_sampling_ratio=2.0,
            min_ratings_for_negatives=0,
            random_seed=42,
        ),
    )
    train, val, test = splitter.split(_labeled_classification_df(), _outcome("own", "classification"))
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    # Classification split adds a label column with both True and False after negative sampling
    assert set(train["label"].to_list()).issubset({True, False})
    # Negatives are 2x positives in each split
    n_pos = train.filter(pl.col("label") == True).height
    n_neg = train.filter(pl.col("label") == False).height
    assert n_neg == 2 * n_pos


def test_regression_split_returns_three_splits_without_negatives():
    splitter = CollectionSplitter(
        universe_df=_universe_df(),
        regression_config=RegressionSplitConfig(
            validation_ratio=0.2, test_ratio=0.2, random_seed=42,
        ),
    )
    train, val, test = splitter.split(
        _labeled_regression_df(), _outcome("rating", "regression")
    )
    assert len(train) + len(val) + len(test) == 10
    # Labels stay numeric
    for split_df in (train, val, test):
        assert split_df["label"].dtype == pl.Float64
    # No negative sampling: label values are all from the input range (6.0-10.0)
    all_labels = (
        train["label"].to_list() + val["label"].to_list() + test["label"].to_list()
    )
    assert all(6.0 <= v <= 10.0 for v in all_labels)


def test_splitter_dispatches_on_outcome_task():
    splitter = CollectionSplitter(
        universe_df=_universe_df(),
        classification_config=ClassificationSplitConfig(random_seed=42),
        regression_config=RegressionSplitConfig(random_seed=42),
    )
    # classification path
    c_train, _, _ = splitter.split(
        _labeled_classification_df(), _outcome("own", "classification")
    )
    assert "label" in c_train.columns

    # regression path
    r_train, _, _ = splitter.split(
        _labeled_regression_df(), _outcome("rating", "regression")
    )
    assert "label" in r_train.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run -m pytest tests/test_collection_splitter.py -v`
Expected: fail — `CollectionSplitter` / `ClassificationSplitConfig` / `RegressionSplitConfig` not importable (current file exports `CollectionSplit` + `SplitConfig`).

- [ ] **Step 3: Rewrite `src/collection/collection_split.py`**

Replace contents with:

```python
"""Train/val/test splitting for user collection models.

Dispatches on OutcomeDefinition.task:
- classification: ClassificationSplitter (with negative sampling)
- regression: RegressionSplitter (no negative sampling; rated rows only)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import polars as pl

from src.collection.outcomes import OutcomeDefinition

logger = logging.getLogger(__name__)


@dataclass
class ClassificationSplitConfig:
    negative_sampling_ratio: float = 5.0
    negative_sampling_strategy: str = "popularity_weighted"  # 'random' | 'popularity_weighted' | 'uniform'
    min_ratings_for_negatives: int = 50
    min_year_for_negatives: Optional[int] = None
    max_year_for_negatives: Optional[int] = None
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


@dataclass
class RegressionSplitConfig:
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify_bins: int = 5
    random_seed: int = 42


class _ClassificationSplitter:
    """Split a labeled classification dataframe, adding negative samples."""

    def __init__(self, universe_df: pl.DataFrame, config: ClassificationSplitConfig):
        self.universe_df = universe_df
        self.config = config
        np.random.seed(config.random_seed)

    def split(self, labeled_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        positives = labeled_df.filter(pl.col("label") == True)
        positive_ids = set(positives["game_id"].to_list())

        if len(positives) == 0:
            raise ValueError("No positive rows (label=True) in classification split input")

        rng = np.random.default_rng(self.config.random_seed)
        shuffled = positives.sample(
            fraction=1.0, shuffle=True, seed=self.config.random_seed
        )
        n = len(shuffled)
        n_test = int(n * self.config.test_ratio)
        n_val = int(n * self.config.validation_ratio)
        test_pos = shuffled[:n_test]
        val_pos = shuffled[n_test : n_test + n_val]
        train_pos = shuffled[n_test + n_val :]

        def _sample_neg(n_samples: int, excluded: set) -> pl.DataFrame:
            eligible = self.universe_df.filter(~pl.col("game_id").is_in(list(excluded)))
            if self.config.min_ratings_for_negatives > 0:
                eligible = eligible.filter(
                    pl.col("users_rated") >= self.config.min_ratings_for_negatives
                )
            if self.config.min_year_for_negatives is not None:
                eligible = eligible.filter(
                    pl.col("year_published") >= self.config.min_year_for_negatives
                )
            if self.config.max_year_for_negatives is not None:
                eligible = eligible.filter(
                    pl.col("year_published") <= self.config.max_year_for_negatives
                )
            if len(eligible) == 0 or n_samples == 0:
                return eligible.head(0).with_columns(pl.lit(False).alias("label"))

            k = min(n_samples, len(eligible))
            if self.config.negative_sampling_strategy == "popularity_weighted":
                weights = eligible["users_rated"].to_numpy().astype(float)
                weights = weights / weights.sum() if weights.sum() > 0 else None
                indices = rng.choice(len(eligible), size=k, replace=False, p=weights)
            else:
                indices = rng.choice(len(eligible), size=k, replace=False)

            sampled = eligible[indices.tolist()]
            return sampled.with_columns(pl.lit(False).alias("label"))

        train_neg = _sample_neg(
            int(len(train_pos) * self.config.negative_sampling_ratio), positive_ids
        )
        val_neg = _sample_neg(
            int(len(val_pos) * self.config.negative_sampling_ratio),
            positive_ids | set(train_neg["game_id"].to_list()),
        )
        test_neg = _sample_neg(
            int(len(test_pos) * self.config.negative_sampling_ratio),
            positive_ids
            | set(train_neg["game_id"].to_list())
            | set(val_neg["game_id"].to_list()),
        )

        # Align schemas: negatives come from universe_df (no collection columns).
        def _align(pos: pl.DataFrame, neg: pl.DataFrame) -> pl.DataFrame:
            common = [c for c in pos.columns if c in neg.columns]
            return pl.concat([pos.select(common), neg.select(common)], how="vertical_relaxed")

        return _align(train_pos, train_neg), _align(val_pos, val_neg), _align(test_pos, test_neg)


class _RegressionSplitter:
    """Split a labeled regression dataframe without negative sampling."""

    def __init__(self, universe_df: pl.DataFrame, config: RegressionSplitConfig):
        self.universe_df = universe_df
        self.config = config

    def split(self, labeled_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        if len(labeled_df) == 0:
            raise ValueError("No rows in regression split input")

        shuffled = labeled_df.sample(
            fraction=1.0, shuffle=True, seed=self.config.random_seed
        )
        n = len(shuffled)
        n_test = int(n * self.config.test_ratio)
        n_val = int(n * self.config.validation_ratio)
        test_df = shuffled[:n_test]
        val_df = shuffled[n_test : n_test + n_val]
        train_df = shuffled[n_test + n_val :]
        return train_df, val_df, test_df


class CollectionSplitter:
    """Public dispatcher: picks classification or regression strategy from outcome.task."""

    def __init__(
        self,
        universe_df: pl.DataFrame,
        classification_config: Optional[ClassificationSplitConfig] = None,
        regression_config: Optional[RegressionSplitConfig] = None,
    ):
        self.universe_df = universe_df
        self._classification = _ClassificationSplitter(
            universe_df, classification_config or ClassificationSplitConfig()
        )
        self._regression = _RegressionSplitter(
            universe_df, regression_config or RegressionSplitConfig()
        )

    def split(
        self, labeled_df: pl.DataFrame, outcome: OutcomeDefinition
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        if outcome.task == "classification":
            return self._classification.split(labeled_df)
        if outcome.task == "regression":
            return self._regression.split(labeled_df)
        raise ValueError(f"Unsupported outcome task: {outcome.task}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run -m pytest tests/test_collection_splitter.py -v`
Expected: 3 tests pass

- [ ] **Step 5: Update stale imports from the old CollectionSplit / SplitConfig names**

Run: `grep -rn "from src.collection.collection_split import\|CollectionSplit\b\|SplitConfig\b" --include="*.py" src/ tests/`
Update any non-test caller (likely `collection_pipeline.py` and possibly `debug/` scripts) to import `CollectionSplitter` / `ClassificationSplitConfig` / `RegressionSplitConfig`. For now, update only non-debug code; debug imports can be left as follow-up.

- [ ] **Step 6: Commit**

```bash
git add src/collection/collection_split.py tests/test_collection_splitter.py
git commit -m "refactor(collection): splitter dispatches on outcome.task; regression path"
```

---

## Task 6: Refactor `collection_model.py` — regression path + task dispatch

**Files:**
- Modify: `src/collection/collection_model.py`

- [ ] **Step 1: Rewrite `src/collection/collection_model.py`**

Replace contents with:

```python
"""Train collection models (classification or regression) per outcome.

Dispatches on OutcomeDefinition.task. Reuses existing preprocessing and
tuning infrastructure from src.models.training.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

from src.collection.outcomes import OutcomeDefinition
from src.models.training import (
    create_preprocessing_pipeline,
    tune_model,
    select_X_y,
)
from src.models.outcomes.hurdle import find_optimal_threshold

logger = logging.getLogger(__name__)


@dataclass
class ClassificationModelConfig:
    model_type: str = "lightgbm"  # 'lightgbm' | 'catboost' | 'logistic'
    use_sample_weights: bool = False
    handle_imbalance: str = "scale_pos_weight"  # 'scale_pos_weight' | 'none'
    threshold_optimization_metric: str = "f2"  # 'f1' | 'f2' | 'precision' | 'recall'
    preprocessor_type: str = "auto"
    tuning_metric: str = "log_loss"
    patience: int = 10


@dataclass
class RegressionModelConfig:
    model_type: str = "lightgbm"  # 'lightgbm' | 'catboost'
    preprocessor_type: str = "auto"
    tuning_metric: str = "rmse"  # 'rmse' | 'mae'
    patience: int = 10


CLASSIFIER_MAPPING = {
    "logistic": lambda: LogisticRegression(max_iter=4000),
    "lightgbm": lambda: lgb.LGBMClassifier(objective="binary", verbose=-1),
    "catboost": lambda: CatBoostClassifier(verbose=0),
}

REGRESSOR_MAPPING = {
    "lightgbm": lambda: lgb.LGBMRegressor(objective="regression", verbose=-1),
    "catboost": lambda: CatBoostRegressor(verbose=0),
}

CLASSIFIER_PARAM_GRIDS = {
    "logistic": {"model__C": [0.001, 0.01, 0.1, 1.0], "model__penalty": ["l2"]},
    "lightgbm": {
        "model__n_estimators": [500],
        "model__learning_rate": [0.01, 0.05],
        "model__max_depth": [3, 5, 7],
        "model__num_leaves": [15, 31],
        "model__min_child_samples": [20],
        "model__scale_pos_weight": [1, 5, 10],
    },
    "catboost": {
        "model__iterations": [500],
        "model__learning_rate": [0.01, 0.05],
        "model__depth": [4, 6],
        "model__scale_pos_weight": [1, 5, 10],
    },
}

REGRESSOR_PARAM_GRIDS = {
    "lightgbm": {
        "model__n_estimators": [500],
        "model__learning_rate": [0.01, 0.05],
        "model__max_depth": [3, 5, 7],
        "model__num_leaves": [15, 31],
        "model__min_child_samples": [20],
    },
    "catboost": {
        "model__iterations": [500],
        "model__learning_rate": [0.01, 0.05],
        "model__depth": [4, 6],
    },
}


class CollectionModel:
    """Train one model for one outcome for one user.

    Dispatches on OutcomeDefinition.task. Callers pass pre-split dataframes
    (train/val/test) with a 'label' column.
    """

    def __init__(
        self,
        username: str,
        outcome: OutcomeDefinition,
        classification_config: Optional[ClassificationModelConfig] = None,
        regression_config: Optional[RegressionModelConfig] = None,
    ):
        self.username = username
        self.outcome = outcome
        self.classification_config = classification_config or ClassificationModelConfig()
        self.regression_config = regression_config or RegressionModelConfig()
        logger.info(
            f"CollectionModel init: user={username!r} outcome={outcome.name!r} task={outcome.task!r}"
        )

    def train(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        if self.outcome.task == "classification":
            return self._train_classification(train_df, val_df)
        if self.outcome.task == "regression":
            return self._train_regression(train_df, val_df)
        raise ValueError(f"Unsupported task: {self.outcome.task}")

    def evaluate(self, pipeline: Pipeline, df: pl.DataFrame) -> Dict[str, float]:
        if self.outcome.task == "classification":
            return self._evaluate_classification(pipeline, df)
        if self.outcome.task == "regression":
            return self._evaluate_regression(pipeline, df)
        raise ValueError(f"Unsupported task: {self.outcome.task}")

    # --- classification path ---

    def _train_classification(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        cfg = self.classification_config
        if cfg.model_type not in CLASSIFIER_MAPPING:
            raise ValueError(
                f"Unknown classification model_type: {cfg.model_type}. "
                f"Choose from {list(CLASSIFIER_MAPPING.keys())}"
            )
        model = CLASSIFIER_MAPPING[cfg.model_type]()
        param_grid = dict(CLASSIFIER_PARAM_GRIDS[cfg.model_type])

        X_train, y_train = self._prepare(train_df)
        X_val, y_val = self._prepare(val_df)

        preprocessor = create_preprocessing_pipeline(preprocessor_type=cfg.preprocessor_type)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        best_pipeline, best_params = tune_model(
            pipeline=pipeline,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            scoring=cfg.tuning_metric,
            patience=cfg.patience,
        )
        return best_pipeline, best_params

    def _evaluate_classification(self, pipeline: Pipeline, df: pl.DataFrame) -> Dict[str, float]:
        X, y = self._prepare(df)
        proba = pipeline.predict_proba(X)[:, 1]
        preds = pipeline.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "f2": fbeta_score(y, preds, beta=2, zero_division=0),
            "roc_auc": roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan"),
            "log_loss": log_loss(y, proba, labels=[0, 1]) if len(set(y)) > 1 else float("nan"),
        }

    def find_threshold(self, pipeline: Pipeline, val_df: pl.DataFrame) -> float:
        if self.outcome.task != "classification":
            raise ValueError("find_threshold is only meaningful for classification outcomes")
        X, y = self._prepare(val_df)
        proba = pipeline.predict_proba(X)[:, 1]
        return find_optimal_threshold(
            y, proba, metric=self.classification_config.threshold_optimization_metric
        )

    # --- regression path ---

    def _train_regression(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        cfg = self.regression_config
        if cfg.model_type not in REGRESSOR_MAPPING:
            raise ValueError(
                f"Unknown regression model_type: {cfg.model_type}. "
                f"Choose from {list(REGRESSOR_MAPPING.keys())}"
            )
        model = REGRESSOR_MAPPING[cfg.model_type]()
        param_grid = dict(REGRESSOR_PARAM_GRIDS[cfg.model_type])

        X_train, y_train = self._prepare(train_df)
        X_val, y_val = self._prepare(val_df)

        preprocessor = create_preprocessing_pipeline(preprocessor_type=cfg.preprocessor_type)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        best_pipeline, best_params = tune_model(
            pipeline=pipeline,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            scoring=cfg.tuning_metric,
            patience=cfg.patience,
        )
        return best_pipeline, best_params

    def _evaluate_regression(self, pipeline: Pipeline, df: pl.DataFrame) -> Dict[str, float]:
        X, y = self._prepare(df)
        preds = pipeline.predict(X)
        mse = mean_squared_error(y, preds)
        return {
            "rmse": mse ** 0.5,
            "mae": mean_absolute_error(y, preds),
            "r2": r2_score(y, preds),
        }

    # --- shared helpers ---

    def _prepare(self, df: pl.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract X, y from a labeled dataframe using existing select_X_y helper."""
        return select_X_y(df, target_col="label")
```

- [ ] **Step 2: Smoke-test the new imports**

Run: `uv run python -c "from src.collection.collection_model import CollectionModel, ClassificationModelConfig, RegressionModelConfig; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run the existing collection tests to confirm no regression in Outcomes module**

Run: `uv run -m pytest tests/test_collection_outcomes.py tests/test_collection_splitter.py -v`
Expected: all tests pass (13 outcomes + 3 splitter = 16 tests)

- [ ] **Step 4: Commit**

```bash
git add src/collection/collection_model.py
git commit -m "refactor(collection): Model dispatches on outcome.task; adds regression path"
```

---

## Task 7: Update `ArtifactStorage` to take `outcome` parameter

**Files:**
- Modify: `src/collection/collection_artifact_storage.py`

- [ ] **Step 1: Read the current save/load method signatures**

Run: `grep -n "def save_\|def load_\|def _path\|def _build\|gs://" src/collection/collection_artifact_storage.py | head -40`

Also run: `uv run python -c "from src.collection.collection_artifact_storage import CollectionArtifactStorage; import inspect; print([m for m in dir(CollectionArtifactStorage) if not m.startswith('_')])"`

Note the existing path layout. We are adding an `outcome` segment to every path: `collections/{username}/{outcome}/{version}/{filename}`. Add `latest_version(username, outcome) -> int | None` and `list_versions(username, outcome) -> list[int]` if they don't already exist — Task 8 relies on them.

- [ ] **Step 2: Add `outcome` parameter to every public save/load method**

For each public method in `CollectionArtifactStorage` (save_model, save_predictions, save_splits, save_analysis, load_model, load_predictions, load_splits, load_analysis, list_versions, etc.), add `outcome: str` as a required parameter (insert immediately after `username`). Update the internal path-building helper so paths become:

```
{env}/collections/{username}/{outcome}/{version}/{filename}
```

Concretely: wherever the current code builds `f"{prefix}/collections/{username}/{filename}"` or similar, replace with `f"{prefix}/collections/{username}/{outcome}/{version}/{filename}"` (where `version` is already passed in or derived).

If the existing code stored artifacts at `{env}/collections/{username}/model-v{N}.pkl`, the new path is `{env}/collections/{username}/{outcome}/v{N}/model.pkl`.

- [ ] **Step 3: Verify module imports cleanly**

Run: `uv run python -c "from src.collection.collection_artifact_storage import CollectionArtifactStorage; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Update callers (pipeline, debug scripts — pipeline is the critical one)**

Run: `grep -rn "CollectionArtifactStorage\|artifact_storage\." --include="*.py" src/collection/ | grep -v __pycache__`

For each call site in `collection_pipeline.py` (the critical path), add an `outcome=` keyword argument. Debug scripts outside `collection_pipeline.py` can be updated in Task 8 if they're in the outcome-aware loop, otherwise left as follow-up.

- [ ] **Step 5: Commit**

```bash
git add src/collection/collection_artifact_storage.py
git commit -m "refactor(collection): ArtifactStorage takes outcome parameter; paths include outcome/version"
```

---

## Task 8: Update `CollectionPipeline` to loop over outcomes

**Files:**
- Modify: `src/collection/collection_pipeline.py`

- [ ] **Step 1: Read the current pipeline flow**

Run: `cat src/collection/collection_pipeline.py`

Note the structure of `run_full_pipeline` and `refresh_predictions_only`.

- [ ] **Step 2: Refactor `run_full_pipeline` to loop over outcomes**

Replace the body of `run_full_pipeline` (keeping its signature compatible — it takes `username` and may take optional overrides) so the flow is:

```python
from src.collection.outcomes import load_outcomes, apply_outcome
from src.collection.collection_processor import CollectionProcessor
from src.collection.collection_split import (
    CollectionSplitter,
    ClassificationSplitConfig,
    RegressionSplitConfig,
)
from src.collection.collection_model import (
    CollectionModel,
    ClassificationModelConfig,
    RegressionModelConfig,
)
from src.utils.config import load_config


def run_full_pipeline(
    self,
    username: str,
    outcome_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run training for all outcomes (or a filtered subset) for one user."""
    # 1. Fetch + persist raw collection
    raw = self.loader.fetch(username)
    self.storage.save_collection(username, raw)

    # 2. Process once
    processor = CollectionProcessor(
        config=self.bq_config, environment=self.environment
    )
    processed = processor.process(username)
    if processed is None:
        raise RuntimeError(f"No collection data for user {username!r}")

    # 3. Load outcomes from config
    app_config = load_config().raw_config  # adjust accessor to however load_config exposes raw dict
    outcomes = load_outcomes(app_config)
    if outcome_filter:
        outcomes = {k: v for k, v in outcomes.items() if k in outcome_filter}

    # 4. Build shared splitter using the joined universe
    splitter = CollectionSplitter(
        universe_df=processed,
        classification_config=ClassificationSplitConfig(),
        regression_config=RegressionSplitConfig(),
    )

    results = {}
    for name, outcome in outcomes.items():
        logger.info(f"Training outcome={name!r} for user={username!r}")
        labeled = apply_outcome(processed, outcome)
        train_df, val_df, test_df = splitter.split(labeled, outcome)

        model = CollectionModel(username=username, outcome=outcome)
        pipeline_obj, best_params = model.train(train_df, val_df)
        metrics = model.evaluate(pipeline_obj, test_df)

        version = self._next_version(username, name)
        self.artifact_storage.save_model(
            username=username, outcome=name, version=version,
            model=pipeline_obj, metadata={"best_params": best_params, "metrics": metrics},
        )
        self.artifact_storage.save_splits(
            username=username, outcome=name, version=version,
            train=train_df, val=val_df, test=test_df,
        )
        results[name] = {"version": version, "metrics": metrics, "best_params": best_params}

    return results
```

Notes:

- To access the raw `collection.outcomes` dict from `load_config()`, check `src/utils/config.py` for the accessor. If `load_config()` returns a Pydantic/dataclass-style object without a raw-dict escape hatch, either add one or `yaml.safe_load(open('config.yaml'))` directly in the pipeline. Prefer the former if easy.
- `self._next_version(username, name)` is a helper on the pipeline that consults `artifact_storage.list_versions(username, outcome)` and returns the next integer (1 if none exist). Add it as a private method if it doesn't exist:

  ```python
  def _next_version(self, username: str, outcome: str) -> int:
      existing = self.artifact_storage.list_versions(username, outcome)
      return (max(existing) + 1) if existing else 1
  ```

- Per-outcome model/split config overrides are out of scope here; the initial version uses default `ClassificationSplitConfig` / `RegressionSplitConfig` / `ClassificationModelConfig` / `RegressionModelConfig` for all outcomes.

- [ ] **Step 3: Refactor `refresh_predictions_only` to loop**

Update it to loop over outcomes, load the latest registered model per outcome, and regenerate predictions using the processed dataframe (no retraining).

```python
def refresh_predictions_only(
    self, username: str, outcome_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    processor = CollectionProcessor(config=self.bq_config, environment=self.environment)
    processed = processor.process(username)
    if processed is None:
        raise RuntimeError(f"No collection data for user {username!r}")

    app_config = load_config().raw_config
    outcomes = load_outcomes(app_config)
    if outcome_filter:
        outcomes = {k: v for k, v in outcomes.items() if k in outcome_filter}

    results = {}
    for name, outcome in outcomes.items():
        version = self.artifact_storage.latest_version(username, name)
        if version is None:
            logger.warning(f"No trained model for user={username!r} outcome={name!r}, skipping")
            continue
        pipeline_obj = self.artifact_storage.load_model(username, name, version)
        X = processed.drop("label") if "label" in processed.columns else processed
        preds = pipeline_obj.predict_proba(X.to_pandas())[:, 1] if outcome.task == "classification" else pipeline_obj.predict(X.to_pandas())
        self.artifact_storage.save_predictions(
            username=username, outcome=name, version=version, predictions=preds, game_ids=processed["game_id"].to_list()
        )
        results[name] = {"version": version, "rows": len(preds)}
    return results
```

- [ ] **Step 4: Verify module imports cleanly**

Run: `uv run python -c "from src.collection.collection_pipeline import CollectionPipeline; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Run all collection tests**

Run: `uv run -m pytest tests/test_collection_outcomes.py tests/test_collection_splitter.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/collection/collection_pipeline.py
git commit -m "refactor(collection): pipeline loops over outcomes from config"
```

---

## Task 9: Update `cli.py` to accept `--outcome` flag

**Files:**
- Modify: `src/collection/cli.py`

- [ ] **Step 1: Read the current `cli.py`**

Run: `cat src/collection/cli.py`

Identify where `run` and `predict` subcommands are defined.

- [ ] **Step 2: Add `--outcome` (optional, repeatable) to `run` and `predict` subparsers**

In the `run` subparser:

```python
run_parser.add_argument(
    "--outcome",
    action="append",
    default=None,
    help="Restrict training to this outcome (repeatable). If omitted, trains all outcomes from config.",
)
```

In the `predict` subparser: same pattern.

In the `run` command handler, pass through to the pipeline:

```python
pipeline.run_full_pipeline(username=args.username, outcome_filter=args.outcome)
```

In the `predict` handler:

```python
pipeline.refresh_predictions_only(username=args.username, outcome_filter=args.outcome)
```

- [ ] **Step 3: Smoke-test the CLI**

Run: `uv run python -m src.collection.cli run --help`
Expected: help text shows `--outcome` flag.

- [ ] **Step 4: Commit**

```bash
git add src/collection/cli.py
git commit -m "feat(collection): cli accepts --outcome flag to filter training"
```

---

## Task 10: Update `__init__.py` and add Makefile targets

**Files:**
- Modify: `src/collection/__init__.py`
- Modify: `Makefile`

- [ ] **Step 1: Export new public API from `__init__.py`**

Replace `src/collection/__init__.py` with:

```python
"""BGG Collection module for loading, processing, storing, and modeling user collections."""

from src.collection.outcomes import (
    OutcomeDefinition,
    load_outcomes,
    apply_outcome,
)
from src.collection.collection_processor import CollectionProcessor, ProcessorConfig
from src.collection.collection_split import (
    CollectionSplitter,
    ClassificationSplitConfig,
    RegressionSplitConfig,
)
from src.collection.collection_model import (
    CollectionModel,
    ClassificationModelConfig,
    RegressionModelConfig,
)

__all__ = [
    "OutcomeDefinition",
    "load_outcomes",
    "apply_outcome",
    "CollectionProcessor",
    "ProcessorConfig",
    "CollectionSplitter",
    "ClassificationSplitConfig",
    "RegressionSplitConfig",
    "CollectionModel",
    "ClassificationModelConfig",
    "RegressionModelConfig",
]
```

- [ ] **Step 2: Verify imports resolve**

Run: `uv run python -c "from src.collection import CollectionSplitter, CollectionModel, load_outcomes; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Add Makefile targets**

Append to `Makefile`:

```makefile
### collection models
.PHONY: train-collection refresh-collection collection-status

train-collection:
	@if [ -z "$(USERNAME)" ]; then echo "USERNAME required, e.g. make train-collection USERNAME=phenrickson"; exit 1; fi
	uv run python -m src.collection.cli run --username $(USERNAME) $(if $(OUTCOME),--outcome $(OUTCOME),)

refresh-collection:
	@if [ -z "$(USERNAME)" ]; then echo "USERNAME required, e.g. make refresh-collection USERNAME=phenrickson"; exit 1; fi
	uv run python -m src.collection.cli predict --username $(USERNAME) $(if $(OUTCOME),--outcome $(OUTCOME),)

collection-status:
	@if [ -z "$(USERNAME)" ]; then echo "USERNAME required, e.g. make collection-status USERNAME=phenrickson"; exit 1; fi
	uv run python -m src.collection.cli status --username $(USERNAME)
```

- [ ] **Step 4: Smoke-test make target**

Run: `make train-collection` (no USERNAME)
Expected: prints the error "USERNAME required..." and exits non-zero. Confirms the target is wired.

- [ ] **Step 5: Commit**

```bash
git add src/collection/__init__.py Makefile
git commit -m "feat(collection): export new API and add Makefile targets"
```

---

## Task 11: Update CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add entry under `[Unreleased]` → `Changed`**

Insert this as the first bullet under the existing `[Unreleased]` → `### Changed` section:

```markdown
- **Collection Module Refactor**: Reorganized `src/collection/` around a new `Outcomes` abstraction
  - New `src/collection/outcomes.py` module defines outcomes declaratively in `config.yaml` under `collection.outcomes` (own, ever_owned, rated, rating, love)
  - `CollectionProcessor` is now outcome-agnostic (join + filter only); labeling applied downstream via `apply_outcome`
  - `CollectionSplitter` and `CollectionModel` dispatch on `outcome.task` (classification or regression)
  - `ArtifactStorage` GCS paths now include the outcome segment: `collections/{username}/{outcome}/v{N}/`
  - New Makefile targets: `train-collection`, `refresh-collection`, `collection-status`
  - Deleted `collection_integration.py` (absorbed into `collection_processor.py`)
  - Training side only; serving via `services/collections/` is a follow-up plan
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog entry for collection modules refactor"
```

---

## Task 12: Run full test suite and verify

- [ ] **Step 1: Run all tests**

Run: `uv run -m pytest tests/ -v --ignore=tests/test_geek_rating.py --ignore=tests/test_train.py 2>&1 | tail -40`

Expected: the new collection tests pass (13 outcomes + 3 splitter = 16 new passes). Pre-existing failures on `test_preprocessor.py`, `test_transformers.py`, `test_register.py` remain (they are not in scope for this plan).

- [ ] **Step 2: Run ruff**

Run: `uv run ruff check src/collection/ tests/test_collection_outcomes.py tests/test_collection_splitter.py 2>&1 | tail -20`

Expected: no errors related to the new code. Address any real errors (not pre-existing unused imports elsewhere).

- [ ] **Step 3: Smoke-test the full module graph**

Run:

```bash
uv run python -c "
from src.collection import (
    OutcomeDefinition, load_outcomes, apply_outcome,
    CollectionProcessor, CollectionSplitter, CollectionModel,
)
from src.collection.collection_pipeline import CollectionPipeline
import yaml
config = yaml.safe_load(open('config.yaml'))
print('Outcomes:', list(load_outcomes(config).keys()))
print('OK')
"
```

Expected: `Outcomes: ['own', 'ever_owned', 'rated', 'rating', 'love']` followed by `OK`.

- [ ] **Step 4: Commit any remaining fixes**

If Steps 1–3 surfaced issues, fix and commit:

```bash
git add -A
git commit -m "fix(collection): resolve remaining refactor issues"
```

---
