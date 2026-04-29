# OOF Cross-Validation Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stratified k-fold cross-validation on the training set, using each candidate's tuned hyperparameters and threshold, persisted as `predictions/oof.parquet` plus an `oof_metrics` block in `registration.json`, and surfaced through `compare_runs` / `summarize_runs`.

**Architecture:** A new `CollectionModel.oof_predict_cv` runs k-fold (Stratified for classification, plain KFold for regression) using the params already stored on the fitted pipeline. `train_candidate` invokes it after `find_threshold`, attaches results to `CandidateRunResult`, and `save_candidate_run` persists them. Comparison utilities read the new registration block and emit a third `oof` row per candidate, ordered `val` → `oof` → `test`.

**Tech Stack:** Python, polars, pandas, scikit-learn (`StratifiedKFold` / `KFold`), pytest, lightgbm.

**Spec:** [`docs/superpowers/specs/2026-04-28-oof-cv-artifact-design.md`](../specs/2026-04-28-oof-cv-artifact-design.md)

---

## File Structure

**Files modified (in order of task):**

- `src/collection/candidates.py` — add `oof_cv_folds` to `CollectionCandidate`, extend `CandidateRunResult`, wire into `train_candidate` and `save_candidate_run`.
- `src/collection/collection_model.py` — add `oof_predict_cv` method.
- `src/collection/collection_artifact_storage.py` — accept `oof_predictions` parameter on `save_candidate_run`, write `predictions/oof.parquet`.
- `src/collection/candidate_comparison.py` — emit `oof` rows from `compare_runs`; sort `val` → `oof` → `test` in `summarize_runs`.
- `src/collection/train.py` — include `oof_metrics_overall` in CLI JSON output.

**Files created:**

- `tests/test_collection_oof.py` — unit + integration tests for OOF behavior.

Each file has one responsibility; the OOF feature touches each in a tight, focused way without restructuring.

---

## Task 1: Add `oof_cv_folds` field to `CollectionCandidate`

**Files:**
- Modify: `src/collection/candidates.py:54-128`
- Test: `tests/test_collection_oof.py` (create)

This task adds the per-candidate config knob with validation. No CV behavior yet — just the field, validation, and round-trip serialization.

- [ ] **Step 1: Write failing tests**

Create `tests/test_collection_oof.py`:

```python
"""Tests for OOF cross-validation artifact generation in collection runs."""

from __future__ import annotations

import pytest

from src.collection.candidates import CollectionCandidate
from src.collection.collection_model import ClassificationModelConfig


def _classification_candidate(name: str = "cand", **overrides) -> CollectionCandidate:
    kwargs = {
        "name": name,
        "classification_config": ClassificationModelConfig(model_type="logistic"),
        "tuning": "none",
        "fixed_params": {},
    }
    kwargs.update(overrides)
    return CollectionCandidate(**kwargs)


def test_oof_cv_folds_defaults_to_5():
    cand = _classification_candidate()
    assert cand.oof_cv_folds == 5


def test_oof_cv_folds_zero_disables():
    cand = _classification_candidate(oof_cv_folds=0)
    assert cand.oof_cv_folds == 0


def test_oof_cv_folds_negative_rejected():
    with pytest.raises(ValueError, match="oof_cv_folds"):
        _classification_candidate(oof_cv_folds=-1)


def test_oof_cv_folds_one_rejected():
    with pytest.raises(ValueError, match="oof_cv_folds"):
        _classification_candidate(oof_cv_folds=1)


def test_oof_cv_folds_round_trips_through_to_dict():
    cand = _classification_candidate(oof_cv_folds=7)
    d = cand.to_dict()
    assert d["oof_cv_folds"] == 7
    restored = CollectionCandidate.from_dict(d)
    assert restored.oof_cv_folds == 7


def test_oof_cv_folds_omitted_in_dict_uses_default():
    cand = _classification_candidate(oof_cv_folds=5)
    d = cand.to_dict()
    d.pop("oof_cv_folds", None)
    restored = CollectionCandidate.from_dict(d)
    assert restored.oof_cv_folds == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_collection_oof.py -v`
Expected: All five tests FAIL — `oof_cv_folds` is not a known field on `CollectionCandidate`.

- [ ] **Step 3: Add the field and validation**

In `src/collection/candidates.py`, find the `CollectionCandidate` dataclass (around line 54). Add `oof_cv_folds` after `notes`:

Find:

```python
    downsample_negatives_ratio: Optional[float] = None
    downsample_protect_min_ratings: int = 25
    notes: str = ""
```

Replace with:

```python
    downsample_negatives_ratio: Optional[float] = None
    downsample_protect_min_ratings: int = 25
    notes: str = ""
    oof_cv_folds: int = 5
```

Then in `__post_init__` (right after the existing `downsample_protect_min_ratings` validation, before the closing of the method), add:

Find:

```python
        if self.downsample_protect_min_ratings < 0:
            raise ValueError(
                f"downsample_protect_min_ratings must be >= 0 "
                f"(got {self.downsample_protect_min_ratings})"
            )
```

Replace with:

```python
        if self.downsample_protect_min_ratings < 0:
            raise ValueError(
                f"downsample_protect_min_ratings must be >= 0 "
                f"(got {self.downsample_protect_min_ratings})"
            )
        if self.oof_cv_folds < 0:
            raise ValueError(
                f"oof_cv_folds must be >= 0 (got {self.oof_cv_folds}); "
                "use 0 to disable OOF CV"
            )
        if self.oof_cv_folds == 1:
            raise ValueError(
                "oof_cv_folds must be 0 (disabled) or >= 2; "
                "got 1 which is not a valid CV configuration"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_collection_oof.py -v`
Expected: All five tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/collection/candidates.py tests/test_collection_oof.py
git commit -m "feat(collection): add oof_cv_folds field to CollectionCandidate

Per-candidate config knob, default 5, 0 disables.
"
```

---

## Task 2: Add `oof_predict_cv` to `CollectionModel` (classification)

**Files:**
- Modify: `src/collection/collection_model.py` — append new method
- Test: `tests/test_collection_oof.py` — extend

The method takes a training frame, runs stratified k-fold using the fitted pipeline's hyperparameters and threshold, and returns OOF predictions + per-fold metrics + pooled overall metrics.

- [ ] **Step 1: Write failing tests for classification path**

Append to `tests/test_collection_oof.py`:

```python
import numpy as np
import pandas as pd
import polars as pl

from src.collection.collection_model import CollectionModel, ClassificationModelConfig
from src.collection.outcomes import DirectColumnRule, OutcomeDefinition


def _classification_outcome() -> OutcomeDefinition:
    return OutcomeDefinition(
        name="own",
        task="classification",
        label_rule=DirectColumnRule(column="label"),
        require=None,
    )


def _make_classification_frame(n: int = 200, seed: int = 0) -> pl.DataFrame:
    """Tiny synthetic classification frame with two informative features."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    logits = 1.5 * f1 - 0.5 * f2
    proba = 1.0 / (1.0 + np.exp(-logits))
    label = (rng.uniform(size=n) < proba).astype(int)
    return pl.DataFrame({
        "game_id": np.arange(n, dtype=np.int64),
        "year_published": np.full(n, 2024, dtype=np.int64),
        "users_rated": rng.integers(0, 1000, size=n).astype(np.int64),
        "f1": f1,
        "f2": f2,
        "label": label.astype(bool),
    })


def _fitted_classification_model(
    train_df: pl.DataFrame, val_df: pl.DataFrame
) -> CollectionModel:
    model = CollectionModel(
        username="tester",
        outcome=_classification_outcome(),
        classification_config=ClassificationModelConfig(model_type="logistic"),
    )
    model.train(train_df, params={})  # default params
    model.find_threshold(val_df)
    return model


def test_oof_predict_cv_classification_shape():
    train = _make_classification_frame(n=200, seed=0)
    val = _make_classification_frame(n=80, seed=1)
    model = _fitted_classification_model(train, val)

    oof_preds, per_fold, overall = model.oof_predict_cv(train, n_folds=4)

    # one OOF prediction per training row
    assert oof_preds.height == train.height
    # original columns preserved + fold/proba/pred appended
    expected_extra = {"fold", "proba", "pred"}
    assert expected_extra.issubset(set(oof_preds.columns))
    # folds partition the rows
    folds = sorted(oof_preds["fold"].unique().to_list())
    assert folds == [0, 1, 2, 3]

    # per-fold metrics: one entry per fold, each carries n_rows
    assert len(per_fold) == 4
    fold_indices = sorted(p["fold"] for p in per_fold)
    assert fold_indices == [0, 1, 2, 3]
    assert all("n_rows" in p for p in per_fold)
    # pooled overall has the same metric keys as evaluate()
    eval_keys = set(model.evaluate(train).keys())
    assert eval_keys.issubset(set(overall.keys()))


def test_oof_predict_cv_classification_deterministic():
    train = _make_classification_frame(n=200, seed=0)
    val = _make_classification_frame(n=80, seed=1)
    model = _fitted_classification_model(train, val)

    a, _, _ = model.oof_predict_cv(train, n_folds=3)
    b, _, _ = model.oof_predict_cv(train, n_folds=3)

    # same seed → same fold assignment and same predictions
    assert a["fold"].to_list() == b["fold"].to_list()
    assert a["proba"].to_list() == b["proba"].to_list()


def test_oof_predict_cv_uses_self_threshold_for_pred():
    train = _make_classification_frame(n=200, seed=0)
    val = _make_classification_frame(n=80, seed=1)
    model = _fitted_classification_model(train, val)
    threshold = model.threshold
    assert threshold is not None

    oof_preds, _, _ = model.oof_predict_cv(train, n_folds=3)
    proba = np.asarray(oof_preds["proba"].to_list())
    pred = np.asarray(oof_preds["pred"].to_list())
    expected = (proba >= threshold).astype(int)
    assert np.array_equal(pred, expected)


def test_oof_predict_cv_requires_fitted():
    model = CollectionModel(
        username="tester",
        outcome=_classification_outcome(),
        classification_config=ClassificationModelConfig(model_type="logistic"),
    )
    train = _make_classification_frame(n=60)
    with pytest.raises(RuntimeError, match="not fit"):
        model.oof_predict_cv(train, n_folds=3)


def test_oof_predict_cv_requires_threshold_for_classification():
    train = _make_classification_frame(n=200, seed=0)
    model = CollectionModel(
        username="tester",
        outcome=_classification_outcome(),
        classification_config=ClassificationModelConfig(model_type="logistic"),
    )
    model.train(train, params={})  # fitted but no threshold
    with pytest.raises(RuntimeError, match="threshold"):
        model.oof_predict_cv(train, n_folds=3)


def test_oof_predict_cv_too_few_rows():
    train = _make_classification_frame(n=5, seed=0)
    val = _make_classification_frame(n=20, seed=1)
    model = _fitted_classification_model(train, val)
    with pytest.raises(ValueError, match="too few rows"):
        model.oof_predict_cv(train, n_folds=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_collection_oof.py -v -k oof_predict_cv`
Expected: All FAIL — `CollectionModel.oof_predict_cv` does not exist.

- [ ] **Step 3: Implement the method (classification path)**

In `src/collection/collection_model.py`, add the imports near the top (the existing import block already imports `KFold`/`StratifiedKFold` indirectly via the training module — we need them directly):

Find:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
```

Replace with:

```python
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
```

Then add the method to the `CollectionModel` class. Insert it just before `_tune_classification` (around line 419 — the marker `# --- classification path ---`):

Find:

```python
    # --- classification path ---

    def _tune_classification(
```

Replace with:

```python
    def oof_predict_cv(
        self,
        train_df: pl.DataFrame,
        n_folds: int,
        random_seed: int = 42,
    ) -> Tuple[pl.DataFrame, list, Dict[str, float]]:
        """Out-of-fold cross-validation using the fitted pipeline's params.

        Re-trains the pipeline on each fold's training rows (preprocessor
        refit per fold to avoid leakage) using the hyperparameters already
        captured on ``self.fitted_pipeline``. Predicts on the held-out fold;
        OOF predictions are reassembled into one row per training row.
        Per-fold metrics are computed via the same evaluators used for
        val/test; the overall metrics are pooled across all OOF rows.

        For classification, ``self.threshold`` must be set; the same
        threshold is used for every fold's hard predictions.

        Args:
            train_df: Training frame to cross-validate over (typically
                ``train_used`` from the candidate run — already feature-
                sliced and downsampled).
            n_folds: Number of folds. Must be >= 2.
            random_seed: Seed for fold shuffling. Default 42 (matches
                ``tune_model_cv``).

        Returns:
            Tuple ``(oof_predictions, per_fold_metrics, overall_metrics)``:

            - ``oof_predictions`` is ``train_df`` with ``fold``, ``proba``
              (classification), and ``pred`` columns appended.
            - ``per_fold_metrics`` is a list of dicts, one per fold, with
              ``fold``, ``n_rows``, ``n_pos`` (classification), and the
              same metric keys as :meth:`evaluate`.
            - ``overall_metrics`` is a dict of pooled metrics across all
              OOF rows.

        Raises:
            RuntimeError: if the model is not fit, or if the model is
                classification and ``self.threshold`` is not set.
            ValueError: if ``n_folds < 2`` or ``train_df.height < n_folds * 2``.
        """
        if self.fitted_pipeline is None:
            raise RuntimeError(
                "Model is not fit. Call train(), tune(), or tune_cv() first."
            )
        if self.outcome.task == "classification" and self.threshold is None:
            raise RuntimeError(
                "Classification OOF requires a threshold. Call find_threshold() first."
            )
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if train_df.height < n_folds * 2:
            raise ValueError(
                f"OOF CV needs too few rows: train_df has {train_df.height} rows "
                f"but {n_folds} folds requires at least {n_folds * 2}"
            )

        params = self.fitted_pipeline.named_steps["model"].get_params(deep=False)

        X, y = self._prepare(train_df)
        y_arr = np.asarray(y)

        if self.outcome.task == "classification":
            splitter = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_seed
            )
        else:
            splitter = KFold(
                n_splits=n_folds, shuffle=True, random_state=random_seed
            )

        n_rows = train_df.height
        fold_assign = np.full(n_rows, -1, dtype=np.int64)
        proba_oof = np.full(n_rows, np.nan, dtype=np.float64) if self.outcome.task == "classification" else None
        pred_oof = np.full(n_rows, np.nan, dtype=np.float64)

        per_fold: list = []
        for fold_idx, (tr_idx, vl_idx) in enumerate(splitter.split(X, y_arr)):
            pipeline = self.build_pipeline()
            pipeline.named_steps["model"].set_params(**params)

            X_tr, X_vl = X.iloc[tr_idx], X.iloc[vl_idx]
            y_tr = y.iloc[tr_idx] if hasattr(y, "iloc") else y_arr[tr_idx]
            y_vl = y_arr[vl_idx]
            pipeline.fit(X_tr, y_tr)

            fold_assign[vl_idx] = fold_idx

            if self.outcome.task == "classification":
                proba = pipeline.predict_proba(X_vl)[:, 1]
                preds = (proba >= float(self.threshold)).astype(int)
                proba_oof[vl_idx] = proba
                pred_oof[vl_idx] = preds.astype(np.float64)

                fold_df = train_df[vl_idx.tolist()] if False else train_df.with_row_count("__row__").filter(pl.col("__row__").is_in(vl_idx.tolist())).drop("__row__")
                fold_metrics = self._evaluate_classification(
                    pipeline, fold_df, threshold=float(self.threshold)
                )
                fold_metrics["fold"] = int(fold_idx)
                fold_metrics["n_rows"] = int(len(vl_idx))
                fold_metrics["n_pos"] = int(np.sum(y_vl == 1))
            else:
                preds = pipeline.predict(X_vl)
                pred_oof[vl_idx] = preds.astype(np.float64)

                fold_df = train_df.with_row_count("__row__").filter(pl.col("__row__").is_in(vl_idx.tolist())).drop("__row__")
                fold_metrics = self._evaluate_regression(pipeline, fold_df)
                fold_metrics["fold"] = int(fold_idx)
                fold_metrics["n_rows"] = int(len(vl_idx))

            per_fold.append(fold_metrics)

        if np.any(fold_assign < 0):
            raise RuntimeError(
                f"OOF assignment is incomplete: {int((fold_assign < 0).sum())} rows unassigned"
            )

        # Build OOF predictions frame: train_df + fold/proba/pred
        if self.outcome.task == "classification":
            oof_predictions = train_df.with_columns([
                pl.Series("fold", fold_assign),
                pl.Series("proba", proba_oof),
                pl.Series("pred", pred_oof.astype(np.int64)),
            ])
        else:
            oof_predictions = train_df.with_columns([
                pl.Series("fold", fold_assign),
                pl.Series("pred", pred_oof),
            ])

        # Pooled overall metrics: evaluate using a probability/prediction
        # vector across all rows (no refit needed since predictions are stored).
        if self.outcome.task == "classification":
            overall = _classification_metrics_from_arrays(
                y_arr.astype(int), proba_oof, threshold=float(self.threshold)
            )
        else:
            overall = _regression_metrics_from_arrays(y_arr.astype(float), pred_oof)

        return oof_predictions, per_fold, overall

    # --- classification path ---

    def _tune_classification(
```

Now add two module-level helper functions. At the very end of `src/collection/collection_model.py` (after the class definition closes), append:

```python
def _classification_metrics_from_arrays(
    y_true: "np.ndarray", proba: "np.ndarray", threshold: float
) -> Dict[str, float]:
    """Pooled classification metrics from arrays. Mirrors the metric set of
    :meth:`CollectionModel._evaluate_classification` for consistency in
    val/test/oof reporting.
    """
    import numpy as np

    preds = (proba >= float(threshold)).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "f2": fbeta_score(y_true, preds, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, proba) if len(set(y_true)) > 1 else float("nan"),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]) if len(set(y_true)) > 1 else float("nan"),
    }


def _regression_metrics_from_arrays(
    y_true: "np.ndarray", preds: "np.ndarray"
) -> Dict[str, float]:
    """Pooled regression metrics from arrays. Mirrors the metric set of
    :meth:`CollectionModel._evaluate_regression`.
    """
    mse = mean_squared_error(y_true, preds)
    return {
        "rmse": mse ** 0.5,
        "mae": mean_absolute_error(y_true, preds),
        "r2": r2_score(y_true, preds),
    }
```

Also add the missing `numpy` import near the top of the file. Find:

```python
import pandas as pd
import polars as pl
```

Replace with:

```python
import numpy as np
import pandas as pd
import polars as pl
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_collection_oof.py -v -k oof_predict_cv`
Expected: All seven `test_oof_predict_cv*` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/collection/collection_model.py tests/test_collection_oof.py
git commit -m "feat(collection): add CollectionModel.oof_predict_cv (classification)

Stratified k-fold using the fitted pipeline's hyperparameters and the
model's tuned threshold. Returns per-row OOF predictions plus per-fold
and pooled metrics.
"
```

---

## Task 3: `oof_predict_cv` regression coverage

**Files:**
- Test: `tests/test_collection_oof.py` — append regression cases

The classification path is implemented and covered. This task adds regression-side tests; the implementation already supports regression (no `proba` column, plain `KFold`), so we only need to verify it.

- [ ] **Step 1: Write failing tests for regression path**

Append to `tests/test_collection_oof.py`:

```python
from src.collection.collection_model import RegressionModelConfig


def _regression_outcome() -> OutcomeDefinition:
    return OutcomeDefinition(
        name="rating",
        task="regression",
        label_rule=DirectColumnRule(column="label"),
        require=None,
    )


def _make_regression_frame(n: int = 200, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    target = 0.7 * f1 - 0.3 * f2 + rng.normal(scale=0.1, size=n)
    return pl.DataFrame({
        "game_id": np.arange(n, dtype=np.int64),
        "year_published": np.full(n, 2024, dtype=np.int64),
        "users_rated": rng.integers(0, 1000, size=n).astype(np.int64),
        "f1": f1,
        "f2": f2,
        "label": target.astype(np.float64),
    })


def test_oof_predict_cv_regression_shape():
    train = _make_regression_frame(n=200, seed=0)
    model = CollectionModel(
        username="tester",
        outcome=_regression_outcome(),
        regression_config=RegressionModelConfig(model_type="lightgbm"),
    )
    model.train(train, params={})

    oof_preds, per_fold, overall = model.oof_predict_cv(train, n_folds=4)

    # regression: no proba column, only fold + pred
    assert "fold" in oof_preds.columns
    assert "pred" in oof_preds.columns
    assert "proba" not in oof_preds.columns
    assert oof_preds.height == train.height

    # pooled regression metrics
    assert {"rmse", "mae", "r2"}.issubset(set(overall.keys()))

    # per-fold list well-formed
    assert len(per_fold) == 4
    for entry in per_fold:
        assert "fold" in entry and "n_rows" in entry
        assert "rmse" in entry
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_collection_oof.py::test_oof_predict_cv_regression_shape -v`
Expected: PASS — implementation already handles regression.

If the test fails, fix the regression branch in `oof_predict_cv` until it passes. Do not modify the classification branch.

- [ ] **Step 3: Commit**

```bash
git add tests/test_collection_oof.py
git commit -m "test(collection): cover oof_predict_cv regression path"
```

---

## Task 4: Wire OOF into `train_candidate` and `CandidateRunResult`

**Files:**
- Modify: `src/collection/candidates.py:195-314` (CandidateRunResult and train_candidate)
- Test: `tests/test_collection_oof.py`

`train_candidate` runs the model end-to-end. After threshold-finding, it should call `oof_predict_cv` (when enabled) and attach results to `CandidateRunResult`.

- [ ] **Step 1: Write failing test**

Append to `tests/test_collection_oof.py`:

```python
from src.collection.candidates import train_candidate


def test_train_candidate_attaches_oof_when_enabled():
    train = _make_classification_frame(n=200, seed=0)
    val = _make_classification_frame(n=80, seed=1)
    test = _make_classification_frame(n=80, seed=2)

    candidate = CollectionCandidate(
        name="cand",
        classification_config=ClassificationModelConfig(model_type="logistic"),
        tuning="none",
        fixed_params={},
        oof_cv_folds=3,
    )
    outcome = _classification_outcome()

    result = train_candidate(
        candidate, outcome,
        train_df=train, val_df=val, test_df=test,
        splits_version=1, username="tester",
    )

    assert result.oof_predictions is not None
    assert result.oof_predictions.height == train.height
    assert result.oof_metrics is not None
    assert result.oof_metrics["n_folds"] == 3
    assert result.oof_metrics["seed"] == 42
    assert result.oof_metrics["stratified_on"] == "label"
    assert result.oof_metrics["threshold"] == result.model.threshold
    assert "overall" in result.oof_metrics
    assert "per_fold" in result.oof_metrics
    assert len(result.oof_metrics["per_fold"]) == 3


def test_train_candidate_skips_oof_when_disabled():
    train = _make_classification_frame(n=200, seed=0)
    val = _make_classification_frame(n=80, seed=1)
    test = _make_classification_frame(n=80, seed=2)

    candidate = CollectionCandidate(
        name="cand",
        classification_config=ClassificationModelConfig(model_type="logistic"),
        tuning="none",
        fixed_params={},
        oof_cv_folds=0,
    )
    outcome = _classification_outcome()

    result = train_candidate(
        candidate, outcome,
        train_df=train, val_df=val, test_df=test,
        splits_version=1, username="tester",
    )

    assert result.oof_predictions is None
    assert result.oof_metrics is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_collection_oof.py::test_train_candidate_attaches_oof_when_enabled tests/test_collection_oof.py::test_train_candidate_skips_oof_when_disabled -v`
Expected: FAIL — `CandidateRunResult` has no `oof_predictions` / `oof_metrics` attributes.

- [ ] **Step 3: Extend `CandidateRunResult`**

In `src/collection/candidates.py`, find the `CandidateRunResult` dataclass:

```python
@dataclass
class CandidateRunResult:
    """In-memory artifacts from training one candidate. Pure result —
    nothing has been persisted yet. Pass to :func:`save_candidate_run` to
    write to disk.
    ...
    """

    candidate: CollectionCandidate
    outcome: OutcomeDefinition
    model: CollectionModel
    best_params: Dict[str, Any]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_used: pl.DataFrame  # the (possibly downsampled / feature-sliced) train frame
    train_n: int
    val_n: int
    test_n: int
    val_predictions: pl.DataFrame
    test_predictions: pl.DataFrame
    feature_importance: pl.DataFrame
    splits_version: Optional[int] = None
    tuning_results: Optional[pl.DataFrame] = None
```

Add two new optional fields at the end:

```python
@dataclass
class CandidateRunResult:
    """In-memory artifacts from training one candidate. Pure result —
    nothing has been persisted yet. Pass to :func:`save_candidate_run` to
    write to disk.
    ...
    """

    candidate: CollectionCandidate
    outcome: OutcomeDefinition
    model: CollectionModel
    best_params: Dict[str, Any]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_used: pl.DataFrame  # the (possibly downsampled / feature-sliced) train frame
    train_n: int
    val_n: int
    test_n: int
    val_predictions: pl.DataFrame
    test_predictions: pl.DataFrame
    feature_importance: pl.DataFrame
    splits_version: Optional[int] = None
    tuning_results: Optional[pl.DataFrame] = None
    oof_predictions: Optional[pl.DataFrame] = None
    oof_metrics: Optional[Dict[str, Any]] = None
```

- [ ] **Step 4: Wire OOF into `train_candidate`**

In `src/collection/candidates.py`, find the body of `train_candidate` near line 280-314:

```python
    best_params, tuning_results = _run_tuning(
        candidate, model, train_used, val_df
    )

    if outcome.task == "classification":
        model.find_threshold(val_df)  # stashes onto model.threshold

    val_metrics = model.evaluate(val_df)
    test_metrics = model.evaluate(test_df)

    val_predictions = model.predict_with_labels(val_df)
    test_predictions = model.predict_with_labels(test_df)
    feature_importance = pl.from_pandas(model.feature_importance())

    tuning_results_pl: Optional[pl.DataFrame] = None
    if tuning_results is not None and len(tuning_results) > 0:
        tuning_results_pl = _coerce_tuning_results(tuning_results)

    return CandidateRunResult(
        candidate=candidate,
        outcome=outcome,
        model=model,
        best_params=best_params,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_used=train_used,
        train_n=train_used.height,
        val_n=val_df.height,
        test_n=test_df.height,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        feature_importance=feature_importance,
        splits_version=splits_version,
        tuning_results=tuning_results_pl,
    )
```

Replace with:

```python
    best_params, tuning_results = _run_tuning(
        candidate, model, train_used, val_df
    )

    if outcome.task == "classification":
        model.find_threshold(val_df)  # stashes onto model.threshold

    oof_predictions: Optional[pl.DataFrame] = None
    oof_metrics: Optional[Dict[str, Any]] = None
    if candidate.oof_cv_folds and candidate.oof_cv_folds > 0:
        oof_predictions, per_fold, overall = model.oof_predict_cv(
            train_used, n_folds=candidate.oof_cv_folds
        )
        oof_metrics = {
            "n_folds": int(candidate.oof_cv_folds),
            "seed": 42,
            "stratified_on": "label" if outcome.task == "classification" else None,
            "threshold": model.threshold,
            "overall": overall,
            "per_fold": per_fold,
        }

    val_metrics = model.evaluate(val_df)
    test_metrics = model.evaluate(test_df)

    val_predictions = model.predict_with_labels(val_df)
    test_predictions = model.predict_with_labels(test_df)
    feature_importance = pl.from_pandas(model.feature_importance())

    tuning_results_pl: Optional[pl.DataFrame] = None
    if tuning_results is not None and len(tuning_results) > 0:
        tuning_results_pl = _coerce_tuning_results(tuning_results)

    return CandidateRunResult(
        candidate=candidate,
        outcome=outcome,
        model=model,
        best_params=best_params,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_used=train_used,
        train_n=train_used.height,
        val_n=val_df.height,
        test_n=test_df.height,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        feature_importance=feature_importance,
        splits_version=splits_version,
        tuning_results=tuning_results_pl,
        oof_predictions=oof_predictions,
        oof_metrics=oof_metrics,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_collection_oof.py -v`
Expected: All tests written so far PASS.

- [ ] **Step 6: Commit**

```bash
git add src/collection/candidates.py tests/test_collection_oof.py
git commit -m "feat(collection): wire OOF CV into train_candidate

CandidateRunResult gains oof_predictions and oof_metrics. When
candidate.oof_cv_folds > 0 (default 5), train_candidate runs OOF CV
after threshold-finding and attaches the artifacts to the result.
"
```

---

## Task 5: Persist OOF artifacts in `save_candidate_run`

**Files:**
- Modify: `src/collection/collection_artifact_storage.py:742-823` (`save_candidate_run`)
- Modify: `src/collection/candidates.py:340-377` (`save_candidate_run` wrapper)
- Test: `tests/test_collection_oof.py`

The storage layer gains an `oof_predictions` parameter; the wrapper folds `oof_metrics` into the registration dict and forwards predictions.

- [ ] **Step 1: Write failing test**

Append to `tests/test_collection_oof.py`:

```python
import json
from pathlib import Path

from src.collection.candidates import save_candidate_run
from src.collection.collection_artifact_storage import CollectionArtifactStorage


def test_save_candidate_run_writes_oof_artifacts(tmp_path: Path):
    train = _make_classification_frame(n=200, seed=0)
    val = _make_classification_frame(n=80, seed=1)
    test = _make_classification_frame(n=80, seed=2)

    candidate = CollectionCandidate(
        name="cand",
        classification_config=ClassificationModelConfig(model_type="logistic"),
        tuning="none",
        fixed_params={},
        oof_cv_folds=3,
    )
    outcome = _classification_outcome()
    result = train_candidate(
        candidate, outcome,
        train_df=train, val_df=val, test_df=test,
        splits_version=1, username="tester",
    )

    storage = CollectionArtifactStorage(
        username="tester", local_root=tmp_path, environment="dev"
    )
    artifact_dir = save_candidate_run(result, storage)

    artifact_path = Path(artifact_dir)
    oof_pq = artifact_path / "predictions" / "oof.parquet"
    assert oof_pq.exists(), f"oof.parquet missing at {oof_pq}"

    oof_loaded = pl.read_parquet(oof_pq)
    assert oof_loaded.height == train.height
    assert {"fold", "proba", "pred"}.issubset(set(oof_loaded.columns))

    reg = json.loads((artifact_path / "registration.json").read_text())
    assert "oof_metrics" in reg
    assert reg["oof_metrics"]["n_folds"] == 3
    assert "overall" in reg["oof_metrics"]
    assert len(reg["oof_metrics"]["per_fold"]) == 3


def test_save_candidate_run_skips_oof_when_disabled(tmp_path: Path):
    train = _make_classification_frame(n=200, seed=0)
    val = _make_classification_frame(n=80, seed=1)
    test = _make_classification_frame(n=80, seed=2)

    candidate = CollectionCandidate(
        name="cand_off",
        classification_config=ClassificationModelConfig(model_type="logistic"),
        tuning="none",
        fixed_params={},
        oof_cv_folds=0,
    )
    outcome = _classification_outcome()
    result = train_candidate(
        candidate, outcome,
        train_df=train, val_df=val, test_df=test,
        splits_version=1, username="tester",
    )

    storage = CollectionArtifactStorage(
        username="tester", local_root=tmp_path, environment="dev"
    )
    artifact_dir = save_candidate_run(result, storage)

    artifact_path = Path(artifact_dir)
    assert not (artifact_path / "predictions" / "oof.parquet").exists()
    reg = json.loads((artifact_path / "registration.json").read_text())
    assert reg.get("oof_metrics") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_collection_oof.py::test_save_candidate_run_writes_oof_artifacts -v`
Expected: FAIL — `oof.parquet` is not written and `oof_metrics` is not in `registration.json`.

- [ ] **Step 3: Add `oof_predictions` to storage `save_candidate_run`**

In `src/collection/collection_artifact_storage.py`, find the `save_candidate_run` signature near line 742:

```python
    def save_candidate_run(
        self,
        outcome: str,
        candidate: str,
        pipeline: Any,
        registration: Dict[str, Any],
        tuning_results: Optional[pl.DataFrame] = None,
        train_used: Optional[pl.DataFrame] = None,
        threshold: Optional[float] = None,
        feature_importance: Optional[pl.DataFrame] = None,
        val_predictions: Optional[pl.DataFrame] = None,
        test_predictions: Optional[pl.DataFrame] = None,
        version: Optional[int] = None,
    ) -> str:
```

Replace with:

```python
    def save_candidate_run(
        self,
        outcome: str,
        candidate: str,
        pipeline: Any,
        registration: Dict[str, Any],
        tuning_results: Optional[pl.DataFrame] = None,
        train_used: Optional[pl.DataFrame] = None,
        threshold: Optional[float] = None,
        feature_importance: Optional[pl.DataFrame] = None,
        val_predictions: Optional[pl.DataFrame] = None,
        test_predictions: Optional[pl.DataFrame] = None,
        oof_predictions: Optional[pl.DataFrame] = None,
        version: Optional[int] = None,
    ) -> str:
```

Then find this block lower in the same method:

```python
        if test_predictions is not None:
            self._upload_parquet(
                version_rel / "predictions" / "test.parquet", test_predictions
            )

        version_dir = self.base_dir / version_rel
```

Replace with:

```python
        if test_predictions is not None:
            self._upload_parquet(
                version_rel / "predictions" / "test.parquet", test_predictions
            )

        if oof_predictions is not None:
            self._upload_parquet(
                version_rel / "predictions" / "oof.parquet", oof_predictions
            )

        version_dir = self.base_dir / version_rel
```

- [ ] **Step 4: Wire `oof_metrics` into the registration in `candidates.save_candidate_run`**

In `src/collection/candidates.py`, find `save_candidate_run` near line 340:

```python
def save_candidate_run(
    result: CandidateRunResult,
    storage: CollectionArtifactStorage,
) -> str:
    """Persist a :class:`CandidateRunResult` under
    ``{outcome}/{candidate.name}/v{N}/`` via ``storage``. Returns the
    artifact directory path.

    Stamps the registration with the storage user, current git SHA, and
    timestamp at save time.
    """
    registration: Dict[str, Any] = {
        "task": result.outcome.task,
        "outcome_name": result.outcome.name,
        "candidate_spec": result.candidate.to_dict(),
        "splits_version": result.splits_version,
        "tuning_strategy": result.candidate.tuning,
        "best_params": result.best_params,
        "metrics": result.test_metrics,
        "val_metrics": result.val_metrics,
        "n_train_used": int(result.train_n),
        "n_val": int(result.val_n),
        "n_test": int(result.test_n),
        "git_sha": _git_sha(),
        "trained_at": datetime.now().isoformat(),
    }
    return storage.save_candidate_run(
        outcome=result.outcome.name,
        candidate=result.candidate.name,
        pipeline=result.model.fitted_pipeline,
        registration=registration,
        tuning_results=result.tuning_results,
        train_used=result.train_used,
        threshold=result.model.threshold,
        feature_importance=result.feature_importance,
        val_predictions=result.val_predictions,
        test_predictions=result.test_predictions,
    )
```

Replace with:

```python
def save_candidate_run(
    result: CandidateRunResult,
    storage: CollectionArtifactStorage,
) -> str:
    """Persist a :class:`CandidateRunResult` under
    ``{outcome}/{candidate.name}/v{N}/`` via ``storage``. Returns the
    artifact directory path.

    Stamps the registration with the storage user, current git SHA, and
    timestamp at save time.
    """
    registration: Dict[str, Any] = {
        "task": result.outcome.task,
        "outcome_name": result.outcome.name,
        "candidate_spec": result.candidate.to_dict(),
        "splits_version": result.splits_version,
        "tuning_strategy": result.candidate.tuning,
        "best_params": result.best_params,
        "metrics": result.test_metrics,
        "val_metrics": result.val_metrics,
        "oof_metrics": result.oof_metrics,
        "n_train_used": int(result.train_n),
        "n_val": int(result.val_n),
        "n_test": int(result.test_n),
        "git_sha": _git_sha(),
        "trained_at": datetime.now().isoformat(),
    }
    return storage.save_candidate_run(
        outcome=result.outcome.name,
        candidate=result.candidate.name,
        pipeline=result.model.fitted_pipeline,
        registration=registration,
        tuning_results=result.tuning_results,
        train_used=result.train_used,
        threshold=result.model.threshold,
        feature_importance=result.feature_importance,
        val_predictions=result.val_predictions,
        test_predictions=result.test_predictions,
        oof_predictions=result.oof_predictions,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_collection_oof.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/collection/collection_artifact_storage.py src/collection/candidates.py tests/test_collection_oof.py
git commit -m "feat(collection): persist OOF artifacts in save_candidate_run

predictions/oof.parquet for the per-row OOF predictions and
oof_metrics block in registration.json for the per-fold and overall
metrics.
"
```

---

## Task 6: Surface OOF in `compare_runs` and `summarize_runs`

**Files:**
- Modify: `src/collection/candidate_comparison.py:80-188`
- Test: `tests/test_collection_oof.py`

`compare_runs` reads `oof_metrics.overall` from the registration and emits a third `split` per candidate. `summarize_runs` orders rows `val` → `oof` → `test`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_collection_oof.py`:

```python
from src.collection.candidate_comparison import compare_runs, summarize_runs


def _registration_with(
    candidate: str,
    *,
    splits_version: int = 1,
    val: Dict[str, float],
    test: Dict[str, float],
    oof_overall: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    reg = {
        "candidate": candidate,
        "version": 1,
        "splits_version": splits_version,
        "task": "classification",
        "threshold": 0.5,
        "best_params": {},
        "n_train_used": 100,
        "n_val": 30,
        "n_test": 30,
        "git_sha": "deadbeef",
        "trained_at": "2026-04-28T00:00:00",
        "val_metrics": val,
        "metrics": test,
    }
    if oof_overall is not None:
        reg["oof_metrics"] = {
            "n_folds": 3,
            "seed": 42,
            "stratified_on": "label",
            "threshold": 0.5,
            "overall": oof_overall,
            "per_fold": [{"fold": i, "n_rows": 30, **oof_overall} for i in range(3)],
        }
    return reg


def test_compare_runs_emits_oof_rows():
    runs = [_registration_with(
        "cand_a",
        val={"f1": 0.5, "log_loss": 0.6},
        test={"f1": 0.4, "log_loss": 0.7},
        oof_overall={"f1": 0.45, "log_loss": 0.65},
    )]
    df = compare_runs(runs)
    splits = sorted(df["split"].unique().to_list())
    assert splits == ["oof", "test", "val"]
    oof_rows = df.filter(pl.col("split") == "oof")
    metrics = sorted(oof_rows["metric"].to_list())
    assert metrics == ["f1", "log_loss"]


def test_compare_runs_omits_oof_when_missing():
    runs = [_registration_with(
        "cand_b",
        val={"f1": 0.5},
        test={"f1": 0.4},
        oof_overall=None,
    )]
    df = compare_runs(runs)
    assert "oof" not in df["split"].unique().to_list()


def test_summarize_runs_orders_val_oof_test():
    runs = [_registration_with(
        "cand_a",
        val={"f1": 0.5},
        test={"f1": 0.4},
        oof_overall={"f1": 0.45},
    )]
    summary = summarize_runs(runs)
    splits = summary["split"].to_list()
    assert splits == ["val", "oof", "test"]


def test_summarize_runs_drops_oof_for_candidate_without_it():
    runs = [
        _registration_with(
            "cand_a",
            val={"f1": 0.5},
            test={"f1": 0.4},
            oof_overall={"f1": 0.45},
        ),
        _registration_with(
            "cand_b",
            val={"f1": 0.6},
            test={"f1": 0.5},
            oof_overall=None,
        ),
    ]
    summary = summarize_runs(runs)
    a_rows = summary.filter(pl.col("candidate") == "cand_a")
    b_rows = summary.filter(pl.col("candidate") == "cand_b")
    assert a_rows["split"].to_list() == ["val", "oof", "test"]
    assert b_rows["split"].to_list() == ["val", "test"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_collection_oof.py -v -k "compare_runs or summarize_runs"`
Expected: FAIL — `compare_runs` does not emit `oof` rows and `summarize_runs` orders alphabetically.

- [ ] **Step 3: Extend `compare_runs`**

In `src/collection/candidate_comparison.py`, find the loop near line 103-132:

```python
    for r in runs:
        splits_versions_seen.add(r.get("splits_version"))
        common = {
            "candidate": r.get("candidate"),
            "version": r.get("version"),
            "splits_version": r.get("splits_version"),
            "task": r.get("task"),
            "threshold": r.get("threshold"),
            "best_params": str(r.get("best_params")),
            "n_train_used": r.get("n_train_used"),
            "n_val": r.get("n_val"),
            "n_test": r.get("n_test"),
            "git_sha": r.get("git_sha"),
            "trained_at": r.get("trained_at"),
        }
        for split_label, metric_dict_key in (("val", "val_metrics"), ("test", "metrics")):
            metrics = r.get(metric_dict_key) or {}
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                if metric_name in EXCLUDED_METRICS:
                    continue
                rows.append(
                    {
                        **common,
                        "split": split_label,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )
```

Replace with:

```python
    for r in runs:
        splits_versions_seen.add(r.get("splits_version"))
        common = {
            "candidate": r.get("candidate"),
            "version": r.get("version"),
            "splits_version": r.get("splits_version"),
            "task": r.get("task"),
            "threshold": r.get("threshold"),
            "best_params": str(r.get("best_params")),
            "n_train_used": r.get("n_train_used"),
            "n_val": r.get("n_val"),
            "n_test": r.get("n_test"),
            "git_sha": r.get("git_sha"),
            "trained_at": r.get("trained_at"),
        }
        for split_label, metric_dict_key in (("val", "val_metrics"), ("test", "metrics")):
            metrics = r.get(metric_dict_key) or {}
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                if metric_name in EXCLUDED_METRICS:
                    continue
                rows.append(
                    {
                        **common,
                        "split": split_label,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )

        oof_overall = (r.get("oof_metrics") or {}).get("overall") or {}
        for metric_name, value in oof_overall.items():
            if not isinstance(value, (int, float)):
                continue
            if metric_name in EXCLUDED_METRICS:
                continue
            rows.append(
                {
                    **common,
                    "split": "oof",
                    "metric": metric_name,
                    "value": float(value),
                }
            )
```

- [ ] **Step 4: Update `summarize_runs` ordering**

In `src/collection/candidate_comparison.py`, find `summarize_runs` near line 163:

```python
def summarize_runs(runs: List[Dict[str, Any]]) -> pl.DataFrame:
    """Wide, metric-focused view of the selected run per candidate.

    Two rows per candidate (val + test), one column per metric, no metadata
    clutter. Use this for at-a-glance candidate comparison; use
    :func:`compare_runs` when you need the long-form view (filtering by
    metric, joining metadata, etc.).

    Args:
        runs: Output of :func:`load_candidate_runs`.

    Returns:
        Polars DataFrame with one row per ``(candidate, split)``, sorted by
        candidate then split (val before test). Empty frame with just
        ``candidate`` and ``split`` columns if ``runs`` is empty.
    """
    tall = compare_runs(runs)
    if tall.height == 0:
        return pl.DataFrame(schema={"candidate": pl.Utf8, "split": pl.Utf8})

    wide = tall.pivot(values="value", index=["candidate", "split"], on="metric")
    return wide.sort(
        ["candidate", "split"],
        descending=[False, True],  # val before test alphabetically
    )
```

Replace with:

```python
_SPLIT_ORDER: Dict[str, int] = {"val": 0, "oof": 1, "test": 2}


def summarize_runs(runs: List[Dict[str, Any]]) -> pl.DataFrame:
    """Wide, metric-focused view of the selected run per candidate.

    Up to three rows per candidate (val, oof, test), one column per metric,
    no metadata clutter. The ``oof`` row is only present for candidates
    whose registration carries an ``oof_metrics`` block. Use this for
    at-a-glance candidate comparison; use :func:`compare_runs` when you
    need the long-form view (filtering by metric, joining metadata, etc.).

    Args:
        runs: Output of :func:`load_candidate_runs`.

    Returns:
        Polars DataFrame with one row per ``(candidate, split)``, sorted by
        candidate then split (val → oof → test). Empty frame with just
        ``candidate`` and ``split`` columns if ``runs`` is empty.
    """
    tall = compare_runs(runs)
    if tall.height == 0:
        return pl.DataFrame(schema={"candidate": pl.Utf8, "split": pl.Utf8})

    wide = tall.pivot(values="value", index=["candidate", "split"], on="metric")
    wide = wide.with_columns(
        pl.col("split")
        .replace_strict(_SPLIT_ORDER, default=99)
        .alias("__split_order")
    )
    return (
        wide.sort(["candidate", "__split_order"])
        .drop("__split_order")
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_collection_oof.py -v`
Expected: All tests PASS.

If `replace_strict` is not available in the installed polars version, fall back to a join-based ordering: build a small `pl.DataFrame({"split": [...], "__split_order": [...]})` and `wide.join(order_df, on="split", how="left")`. Run the tests again to confirm.

- [ ] **Step 6: Commit**

```bash
git add src/collection/candidate_comparison.py tests/test_collection_oof.py
git commit -m "feat(collection): emit OOF rows in compare_runs/summarize_runs

Third row per candidate, ordered val -> oof -> test.
"
```

---

## Task 7: Include `oof_metrics_overall` in CLI JSON output

**Files:**
- Modify: `src/collection/train.py:118-134`
- Test: `tests/test_collection_oof.py`

The training CLI prints a single-line JSON summary. Add `oof_metrics_overall` so callers see CV performance without loading parquet.

- [ ] **Step 1: Write failing test**

Append to `tests/test_collection_oof.py`:

```python
def test_train_cli_includes_oof_metrics_overall(tmp_path: Path, capsys, monkeypatch):
    """End-to-end: train CLI run produces oof_metrics_overall in stdout JSON."""
    from src.collection import train as train_cli
    from unittest.mock import patch

    train_df = _make_classification_frame(n=200, seed=0)
    val_df = _make_classification_frame(n=80, seed=1)
    test_df = _make_classification_frame(n=80, seed=2)

    raw_config = {
        "collections": {
            "outcomes": {
                "own": {"task": "classification", "label_from": "label"},
            },
            "candidates": [
                {
                    "name": "logistic_default",
                    "classification_config": {"model_type": "logistic"},
                    "tuning": "none",
                    "fixed_params": {},
                    "oof_cv_folds": 3,
                }
            ],
        }
    }

    fake_cfg = MagicMock()
    fake_cfg.raw_config = raw_config
    fake_cfg.get_environment_prefix.return_value = "dev"

    storage = CollectionArtifactStorage(
        username="tester", local_root=tmp_path, environment="dev"
    )
    storage.save_canonical_splits("own", train_df, val_df, test_df, version=1)

    with patch("src.collection.train.load_config", return_value=fake_cfg), patch(
        "src.collection.train.CollectionArtifactStorage", return_value=storage
    ):
        rc = train_cli.main([
            "--username", "tester",
            "--environment", "dev",
            "--outcome", "own",
            "--candidate", "logistic_default",
            "--splits-version", "1",
            "--local-root", str(tmp_path),
        ])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(out)
    assert "oof_metrics_overall" in payload
    assert isinstance(payload["oof_metrics_overall"], dict)
    assert "f1" in payload["oof_metrics_overall"]
```

You'll also need this import at the top of the test file (add it if not already there):

```python
from unittest.mock import MagicMock
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_collection_oof.py::test_train_cli_includes_oof_metrics_overall -v`
Expected: FAIL — `oof_metrics_overall` is not in the CLI output.

- [ ] **Step 3: Update CLI output**

In `src/collection/train.py`, find the JSON output near line 118:

```python
    print(
        json.dumps(
            {
                "candidate": candidate.name,
                "outcome": args.outcome,
                "version": version,
                "splits_version": result.splits_version,
                "threshold": result.model.threshold,
                "val_metrics": result.val_metrics,
                "test_metrics": result.test_metrics,
                "n_train": result.train_n,
                "n_val": result.val_n,
                "n_test": result.test_n,
                "artifact_dir": artifact_dir,
            }
        )
    )
```

Replace with:

```python
    oof_overall = (result.oof_metrics or {}).get("overall")
    print(
        json.dumps(
            {
                "candidate": candidate.name,
                "outcome": args.outcome,
                "version": version,
                "splits_version": result.splits_version,
                "threshold": result.model.threshold,
                "val_metrics": result.val_metrics,
                "test_metrics": result.test_metrics,
                "oof_metrics_overall": oof_overall,
                "n_train": result.train_n,
                "n_val": result.val_n,
                "n_test": result.test_n,
                "artifact_dir": artifact_dir,
            }
        )
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_collection_oof.py::test_train_cli_includes_oof_metrics_overall -v`
Expected: PASS.

- [ ] **Step 5: Run the full test file**

Run: `uv run pytest tests/test_collection_oof.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/collection/train.py tests/test_collection_oof.py
git commit -m "feat(collection): include oof_metrics_overall in train CLI output"
```

---

## Task 8: End-to-end verification on a real run

**Files:** none modified — manual sanity check.

This is verification-before-completion: run the full flow on real data with the existing config to confirm OOF artifacts appear and `summarize_runs` produces the expected table.

- [ ] **Step 1: Run the full collection-modules test suite**

Run: `uv run pytest tests/ -v -x`
Expected: All tests pass. Investigate and fix any failures before continuing — there should be no regressions in unrelated tests.

- [ ] **Step 2: Run a real candidate end-to-end**

Use the existing justfile / Makefile target for training one candidate. Pick the smallest configured candidate to keep this fast:

Run: `uv run python -m src.collection.train --username phenrickson --outcome own --candidate logistic_default --environment dev`

Expected stdout: a single JSON line that now includes a non-null `"oof_metrics_overall"` block.

If the candidate name `logistic_default` is not in the local config, list configured candidates with `uv run python -c "from src.utils.config import load_config; from src.collection.candidates import load_candidates; print(list(load_candidates(load_config().raw_config)))"` and pick one.

- [ ] **Step 3: Inspect on-disk artifacts**

Run: `ls -la models/collections/phenrickson/own/logistic_default/v*/predictions/`
Expected: `val.parquet`, `test.parquet`, **and `oof.parquet`** all present.

Run: `uv run python -c "import polars as pl; df = pl.read_parquet('models/collections/phenrickson/own/logistic_default/$(ls -t models/collections/phenrickson/own/logistic_default | head -1)/predictions/oof.parquet'); print(df.columns); print(df.head(3))"`

Expected: columns include `game_id`, `label`, `fold`, `proba`, `pred`, plus the rest of the training frame columns. `fold` values are 0..n_folds-1.

- [ ] **Step 4: Inspect `summarize_runs` output**

Use the existing `summarize` entry point (check the Makefile / justfile for the exact target — likely `just summarize` or `uv run python -m src.collection.compare`). Confirm each candidate's row block reads `val` then `oof` then `test`, matching the screenshot in the brainstorming session.

- [ ] **Step 5: Final commit (only if any docstring/comment fixes from the verification pass)**

If everything works without changes, no commit. If a small adjustment was needed, commit:

```bash
git add <files>
git commit -m "fix(collection): <specific tweak from end-to-end verification>"
```

---

## Self-review

**Spec coverage:**

- New `oof_cv_folds` field with validation → Task 1.
- `CollectionModel.oof_predict_cv` (classification + regression) → Task 2 (classification), Task 3 (regression).
- `CandidateRunResult` extension → Task 4.
- `train_candidate` wiring → Task 4.
- `predictions/oof.parquet` persistence → Task 5.
- `oof_metrics` block in `registration.json` → Task 5.
- `compare_runs` emits `oof` split rows → Task 6.
- `summarize_runs` orders `val` → `oof` → `test` → Task 6.
- CLI `oof_metrics_overall` field → Task 7.
- Test coverage (unit classification, unit regression, integration save, comparison, CLI) → Tasks 1–7.
- End-to-end verification → Task 8.

**Placeholder scan:** No "TBD", "implement later", "similar to Task N", or vague "add error handling" steps. Every code change has the exact code; every test has the exact assertions.

**Type consistency:**

- `oof_predict_cv` returns `Tuple[pl.DataFrame, list, Dict[str, float]]` — used identically in Task 4 (`oof_predictions, per_fold, overall = ...`).
- `CandidateRunResult.oof_predictions` (`Optional[pl.DataFrame]`) and `oof_metrics` (`Optional[Dict[str, Any]]`) match the wiring in Task 4 and the storage parameters in Task 5.
- Storage method signature gains `oof_predictions: Optional[pl.DataFrame] = None`; the wrapper passes the same field through.
- `oof_metrics` registration block keys — `n_folds`, `seed`, `stratified_on`, `threshold`, `overall`, `per_fold` — are produced in Task 4 and read in Task 6 (`(r.get("oof_metrics") or {}).get("overall") or {}`) and Task 7 (`(result.oof_metrics or {}).get("overall")`). Consistent.
