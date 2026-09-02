# Embedding Input Scaling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the blanket `StandardScaler` in the game-embedding preprocessor with Gelman-style scaling (continuous ÷ 2·SD, binary dummies left at 0/1) so rare uncorrelated features stop capturing whole SVD/PCA dimensions. Switch the embedding algorithm `svd → pca` (`whiten=False`) since the dummies are no longer centred. Add a minimum-frequency floor on dummy features. Ship a component-diagnostic script.

**Spec:** `docs/superpowers/specs/2026-08-31-embedding-input-scaling-design.md`

**Architecture:** Two small, reusable transformers in `src/features/transformers.py` (`TwoSDScaler`, `MinCountSelector`), following the existing `RowNormalizer` / `LogTransformer` pattern (column-subset transformer, pass-through others, `set_output` + `get_feature_names_out`). Wired into `create_embedding_preprocessor` (`src/models/embeddings/transformer.py`). The algorithm switch is `config.yaml`-only — `PCAEmbedding` already exists and already emits component loadings. Then a new embedding **experiment version** is trained and evaluated on the bgg-viewer bench before any BigQuery table regenerates.

**Tech stack:** Python 3.12, scikit-learn, pandas, pytest, uv. Conventions: `uv run python -m …` for modules, `uv run --extra test python -m pytest …` for tests, `tmp_path` for hermetic tests.

**Branch:** `feat/embedding-input-scaling` → PR to `main`. Production embedding generation continues on `embeddings-v2026` until the new version clears evaluation. User reviews and merges the PR.

**Blast radius:** the game SVD embedding feeds only `analytics.game_similarity_search` → `game_neighbors` → the bgg-viewer similar-games page, plus the 2D coordinates job. The predictive models consume the *text* embeddings (`bgg_description_embeddings`) and are untouched.

---

## File structure

**Modified:**
- `src/features/transformers.py` — add `TwoSDScaler`, `MinCountSelector`.
- `src/models/embeddings/transformer.py` — swap scaler, add min-count step in `create_embedding_preprocessor`.
- `config.yaml` — `embeddings.algorithm: svd → pca`; `embeddings.algorithms.pca.whiten: false`, `svd_solver: randomized`.

**New:**
- `tests/test_two_sd_scaler.py`
- `tests/test_min_count_selector.py`
- `src/models/embeddings/diagnose_components.py` — component-loadings × prevalence diagnostic CLI.
- `tests/test_diagnose_components.py`

**Out of scope (per spec):**
- Adopting `TwoSDScaler` in `create_bgg_preprocessor` / the deployed predictive models.
- Value clipping (documented fallback only).
- Regenerating `game_similarity_search` / coordinates / `game_neighbors` — happens after evaluation, tracked separately.

---

## Stage 1: `TwoSDScaler`

A column-subset scaler: continuous columns → `(x − mean) / (factor · SD)`; binary (0/1) columns and any unmatched column pass through unchanged. Auto-detects binary columns at `fit` (values ⊆ {0, 1}); an explicit `continuous_columns` list overrides.

### Task 1: Branch + failing tests for `TwoSDScaler`

**Files:**
- Create: `tests/test_two_sd_scaler.py`

- [ ] **Step 1: Branch**

```bash
git checkout -b feat/embedding-input-scaling
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_two_sd_scaler.py`:

```python
"""Tests for TwoSDScaler — Gelman-style scaling (continuous ÷ 2·SD, dummies at 0/1)."""

import numpy as np
import pandas as pd
import pytest

from src.features.transformers import TwoSDScaler


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "playtime": rng.normal(60, 30, 500),          # continuous
            "complexity": rng.uniform(1, 5, 500),          # continuous
            "mechanic_dice": rng.integers(0, 2, 500),      # binary dummy
            "category_war": rng.integers(0, 2, 500),       # binary dummy
        }
    )


def test_continuous_columns_get_mean_zero_sd_half(frame):
    out = TwoSDScaler().fit_transform(frame)
    for col in ["playtime", "complexity"]:
        assert out[col].mean() == pytest.approx(0.0, abs=1e-9)
        # divided by 2 SD → resulting SD is ~0.5
        assert out[col].std(ddof=0) == pytest.approx(0.5, rel=1e-6)


def test_binary_columns_pass_through_untouched(frame):
    out = TwoSDScaler().fit_transform(frame)
    for col in ["mechanic_dice", "category_war"]:
        pd.testing.assert_series_equal(out[col], frame[col], check_dtype=False)


def test_explicit_continuous_columns_override(frame):
    # force 'complexity' to be treated as pass-through
    out = TwoSDScaler(continuous_columns=["playtime"]).fit_transform(frame)
    pd.testing.assert_series_equal(out["complexity"], frame["complexity"], check_dtype=False)
    assert out["playtime"].std(ddof=0) == pytest.approx(0.5, rel=1e-6)


def test_transform_uses_fit_statistics(frame):
    scaler = TwoSDScaler().fit(frame)
    shifted = frame.copy()
    shifted["playtime"] = shifted["playtime"] + 100  # different distribution
    out = scaler.transform(shifted)
    # mean is NOT re-centred on the new frame; it's offset by 100 / (2·sd_fit)
    expected_offset = 100 / (2 * frame["playtime"].std(ddof=0))
    assert out["playtime"].mean() == pytest.approx(
        (frame["playtime"].mean() - frame["playtime"].mean()) / 1 + expected_offset, abs=0.05
    )


def test_unknown_columns_pass_through(frame):
    extra = frame.assign(id=range(len(frame)))
    out = TwoSDScaler(continuous_columns=["playtime"]).fit_transform(extra)
    pd.testing.assert_series_equal(out["id"], extra["id"], check_dtype=False)


def test_get_feature_names_out_is_identity(frame):
    scaler = TwoSDScaler().fit(frame)
    assert list(scaler.get_feature_names_out()) == list(frame.columns)


def test_zero_sd_column_is_safe():
    df = pd.DataFrame({"const": [3.0] * 10, "x": np.arange(10.0)})
    out = TwoSDScaler(continuous_columns=["const", "x"]).fit_transform(df)
    assert np.isfinite(out["const"]).all()  # no divide-by-zero blow-up
```

- [ ] **Step 3: Run — expect failure**

```bash
uv run --extra test python -m pytest tests/test_two_sd_scaler.py -q
```
Expected: `ImportError` / all tests fail (`TwoSDScaler` doesn't exist).

### Task 2: Implement `TwoSDScaler`

**Files:**
- Modify: `src/features/transformers.py`

- [ ] **Step 1: Add the class**

Place it directly after `RowNormalizer` (same section). Mirror `RowNormalizer`'s structure — `__init__` stores params only, `fit` learns `*_` attributes, `set_output` + `get_feature_names_out` for sklearn-pandas compatibility.

```python
class TwoSDScaler(BaseEstimator, TransformerMixin):
    """Gelman-style input scaling: continuous columns → ``(x - mean) / (factor * SD)``,
    binary (0/1) columns pass through unchanged.

    Gelman (2008), *Scaling regression inputs by dividing by two standard
    deviations*: with ``factor=2`` a continuous input ends up with SD ~0.5,
    matching a balanced 0/1 dummy, while a rare dummy keeps its natural
    variance ``p(1-p)`` instead of being forced to 1 by ``StandardScaler``.
    In a PCA/SVD pipeline that stops rare uncorrelated features from capturing
    their own component.

    Parameters
    ----------
    continuous_columns : list of str, optional
        Columns to scale. If ``None``, auto-detected at ``fit`` as every column
        whose observed values are not a subset of ``{0, 1}``.
    factor : float, default 2.0
        SD multiplier in the denominator.
    """

    def __init__(self, continuous_columns=None, factor: float = 2.0):
        self.continuous_columns = continuous_columns
        self.factor = factor
        self._output_config = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        if self.continuous_columns is None:
            cont = []
            for c in X.columns:
                vals = pd.unique(X[c].dropna())
                if not set(np.asarray(vals).ravel()).issubset({0, 1}):
                    cont.append(c)
            self.continuous_columns_ = cont
        else:
            self.continuous_columns_ = [c for c in self.continuous_columns if c in X.columns]
        self.means_ = X[self.continuous_columns_].mean()
        sds = X[self.continuous_columns_].std(ddof=0)
        self.scales_ = (self.factor * sds).replace(0.0, 1.0)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X = X.copy()
        cols = self.continuous_columns_
        X[cols] = (X[cols] - self.means_) / self.scales_
        return X

    def set_output(self, *, transform=None):
        if transform is not None and transform not in ["default", "pandas"]:
            raise ValueError(
                f"Invalid transform parameter: {transform}. Must be 'default' or 'pandas'."
            )
        self._output_config = transform
        return self

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(list(input_features))
        return np.asarray(self.feature_names_in_)
```

- [ ] **Step 2: Run — expect pass**

```bash
uv run --extra test python -m pytest tests/test_two_sd_scaler.py -q
```
Fix `test_transform_uses_fit_statistics` if its arithmetic assertion is clumsy — the intent is "transform applies fit-time mean/scale, not re-fit"; simplify to comparing two `transform` calls on frames that differ by a known constant.

- [ ] **Step 3: Full transformer suite still green**

```bash
uv run --extra test python -m pytest tests/test_transformers.py -q
```

- [ ] **Step 4: Commit**

```bash
git add src/features/transformers.py tests/test_two_sd_scaler.py
git commit -m "feat(transformers): add TwoSDScaler (Gelman 2-SD input scaling)"
```

---

## Stage 2: `MinCountSelector`

Drop binary columns whose column sum (number of games with the feature) is below `min_count`. Continuous and unmatched columns pass through. Reliability floor — not the core fix — so it's a separate, independently-revertable step.

### Task 3: Failing tests for `MinCountSelector`

**Files:**
- Create: `tests/test_min_count_selector.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for MinCountSelector — drop rare binary indicator columns."""

import numpy as np
import pandas as pd

from src.features.transformers import MinCountSelector


def _frame():
    df = pd.DataFrame({
        "cont": np.arange(100.0),
        "common_dummy": ([1] * 40) + ([0] * 60),
        "rare_dummy": ([1] * 3) + ([0] * 97),
    })
    return df


def test_drops_rare_binary_columns():
    out = MinCountSelector(min_count=10).fit_transform(_frame())
    assert "rare_dummy" not in out.columns
    assert "common_dummy" in out.columns


def test_keeps_continuous_regardless_of_sum():
    # a continuous column with small values must never be dropped
    df = pd.DataFrame({"x": [0.01] * 100})
    out = MinCountSelector(min_count=10).fit_transform(df)
    assert "x" in out.columns


def test_transform_applies_fit_column_set():
    sel = MinCountSelector(min_count=10).fit(_frame())
    out = sel.transform(_frame().iloc[:5])          # rare_dummy all-zero in this slice
    assert "rare_dummy" not in out.columns
    assert list(out.columns) == list(sel.get_feature_names_out())


def test_get_feature_names_out_reflects_kept_columns():
    sel = MinCountSelector(min_count=10).fit(_frame())
    assert set(sel.get_feature_names_out()) == {"cont", "common_dummy"}
```

- [ ] **Step 2: Run — expect failure**

```bash
uv run --extra test python -m pytest tests/test_min_count_selector.py -q
```

### Task 4: Implement `MinCountSelector`

**Files:**
- Modify: `src/features/transformers.py`

- [ ] **Step 1: Add the class** (after `TwoSDScaler`)

```python
class MinCountSelector(BaseEstimator, TransformerMixin):
    """Drop binary (0/1) columns whose column sum is below ``min_count``.

    Removes indicator features carried by too few rows to estimate reliably.
    Continuous columns (values not a subset of {0, 1}) and unmatched columns
    are always kept.
    """

    def __init__(self, min_count: int = 10):
        self.min_count = min_count
        self._output_config = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        drop = []
        for c in X.columns:
            col = X[c].dropna()
            if set(np.asarray(pd.unique(col)).ravel()).issubset({0, 1}):
                if float(col.sum()) < self.min_count:
                    drop.append(c)
        self.columns_to_drop_ = drop
        self.columns_kept_ = [c for c in X.columns if c not in drop]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.drop(columns=self.columns_to_drop_, errors="ignore")

    def set_output(self, *, transform=None):
        if transform is not None and transform not in ["default", "pandas"]:
            raise ValueError(
                f"Invalid transform parameter: {transform}. Must be 'default' or 'pandas'."
            )
        self._output_config = transform
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_kept_)
```

- [ ] **Step 2: Run — expect pass**

```bash
uv run --extra test python -m pytest tests/test_min_count_selector.py -q
```

- [ ] **Step 3: Commit**

```bash
git add src/features/transformers.py tests/test_min_count_selector.py
git commit -m "feat(transformers): add MinCountSelector (drop rare indicator columns)"
```

---

## Stage 3: Wire into the embedding preprocessor + algorithm swap

### Task 5: Failing test for the updated preprocessor

**Files:**
- Create: `tests/test_embedding_preprocessor_scaling.py`

- [ ] **Step 1: Write failing test**

```python
"""The embedding preprocessor uses Gelman scaling, not blanket StandardScaler."""

from sklearn.preprocessing import StandardScaler

from src.models.embeddings.transformer import create_embedding_preprocessor
from src.features.transformers import TwoSDScaler, MinCountSelector


def test_linear_pipeline_uses_two_sd_scaler_not_standardscaler():
    pipe = create_embedding_preprocessor(model_type="linear")
    step_types = {name: type(est) for name, est in pipe.steps}
    assert TwoSDScaler in step_types.values()
    assert StandardScaler not in step_types.values()


def test_linear_pipeline_has_min_count_step():
    pipe = create_embedding_preprocessor(model_type="linear")
    assert any(isinstance(est, MinCountSelector) for _, est in pipe.steps)


def test_min_count_is_configurable():
    pipe = create_embedding_preprocessor(model_type="linear", min_feature_count=25)
    sel = next(est for _, est in pipe.steps if isinstance(est, MinCountSelector))
    assert sel.min_count == 25
```

- [ ] **Step 2: Run — expect failure**

```bash
uv run --extra test python -m pytest tests/test_embedding_preprocessor_scaling.py -q
```

### Task 6: Update `create_embedding_preprocessor`

**Files:**
- Modify: `src/models/embeddings/transformer.py`

- [ ] **Step 1: Edit the `linear` branch**

Add a `min_feature_count: int = 10` parameter to `create_embedding_preprocessor`. In the `model_type == "linear"` step list:
- keep `("variance_selector", VarianceThreshold(threshold=0))`
- add `("min_count", MinCountSelector(min_count=min_feature_count))` after it
- replace `("scaler", StandardScaler())` with `("scaler", TwoSDScaler())`

Import `TwoSDScaler, MinCountSelector` from `src.features.transformers`. Leave the `tree` branch alone.

- [ ] **Step 2: Run — expect pass**

```bash
uv run --extra test python -m pytest tests/test_embedding_preprocessor_scaling.py -q
```

- [ ] **Step 3: Preprocessor still fits end-to-end on a sample**

```bash
uv run --extra test python -m pytest tests/ -q -k "embedding" 
```
Expected: green (or pre-existing unrelated failures only — note them, don't fix here).

- [ ] **Step 4: Commit**

```bash
git add src/models/embeddings/transformer.py tests/test_embedding_preprocessor_scaling.py
git commit -m "feat(embeddings): Gelman scaling + min-count floor in the preprocessor"
```

### Task 7: Algorithm swap in config

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Edit `embeddings:` block**

```yaml
embeddings:
  algorithm: pca          # was: svd
  embedding_dim: 64
  ...
  algorithms:
    pca:
      whiten: false        # was: true
      svd_solver: randomized
    svd:
      n_iter: 5
```

- [ ] **Step 2: Verify `PCAEmbedding` forwards `svd_solver`**

Check `src/models/embeddings/algorithms.py` `PCAEmbedding.__init__` passes `**kwargs` to `sklearn.decomposition.PCA` (it does). If `get_algorithm_params` in `train.py` filters kwargs, ensure `svd_solver` survives. Add a test if the path is non-obvious:

```bash
uv run python -c "
from src.models.embeddings.algorithms import create_embedding_algorithm
a = create_embedding_algorithm('pca', 64, whiten=False, svd_solver='randomized')
print(a.model)
"
```
Expected: `PCA(n_components=64, random_state=..., svd_solver='randomized')` (no `whiten=True`).

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "config(embeddings): svd -> pca, whiten off, randomized solver"
```

---

## Stage 4: Component diagnostic

### Task 8: Failing test for the diagnostic

**Files:**
- Create: `tests/test_diagnose_components.py`

- [ ] **Step 1: Write failing test**

```python
"""diagnose_components summarises each PCA component by loading × feature prevalence."""

import numpy as np

from src.models.embeddings.diagnose_components import summarize_components


def test_summarize_flags_rare_feature_dominated_component():
    # component 0 loads almost entirely on feature index 2
    components = np.array([
        [0.02, 0.03, 0.99, 0.01],
        [0.5, 0.5, 0.0, 0.5],
    ])
    feature_names = ["common_a", "common_b", "rare_x", "common_c"]
    prevalence = np.array([0.40, 0.35, 0.0006, 0.30])

    rows = summarize_components(components, feature_names, prevalence, top_k=3)

    comp0 = rows[0]
    assert comp0["top_features"][0]["feature"] == "rare_x"
    assert comp0["top_features"][0]["prevalence"] == 0.0006
    # concentration = max(loading)^2 / sum(loading^2), ~1.0 here
    assert comp0["concentration"] > 0.9
    assert comp0["min_prevalence_in_top"] == 0.0006
```

- [ ] **Step 2: Run — expect failure**

```bash
uv run --extra test python -m pytest tests/test_diagnose_components.py -q
```

### Task 9: Implement `diagnose_components.py`

**Files:**
- Create: `src/models/embeddings/diagnose_components.py`

- [ ] **Step 1: Pure summariser + CLI**

`summarize_components(components, feature_names, prevalence, top_k=10) -> list[dict]` — for each component: `top_features` (feature, loading, abs_loading, prevalence), `concentration = max(loading**2)/sum(loading**2)`, `min_prevalence_in_top`, `explained_variance_ratio` (optional arg).

CLI (`python -m src.models.embeddings.diagnose_components --experiment game-embeddings [--version N]`):
- Load the fitted pipeline via `EmbeddingGenerator` / `ExperimentTracker`.
- Pull `components_` + `feature_names_out` from the fitted PCA + preprocessor.
- Compute per-feature prevalence from the training feature matrix (fraction of rows == 1 for binary cols; `nan` for continuous).
- Print a table sorted by `concentration` desc, and a flag list: components whose `min_prevalence_in_top < 0.005`.

- [ ] **Step 2: Run — expect pass**

```bash
uv run --extra test python -m pytest tests/test_diagnose_components.py -q
```

- [ ] **Step 3: Smoke-run against the current (svd) model** for a baseline

```bash
uv run python -m src.models.embeddings.diagnose_components --experiment game-embeddings
```
Expected: prints components; the rare-feature-dominated ones (fan-expansion, Resource Queue) show high concentration + tiny `min_prevalence_in_top`. Save the output to the PR description as the "before".

- [ ] **Step 4: Commit**

```bash
git add src/models/embeddings/diagnose_components.py tests/test_diagnose_components.py
git commit -m "feat(embeddings): component-loadings x prevalence diagnostic"
```

---

## Stage 5: Train + evaluate the new embedding version (runbook)

Not TDD — a training run plus human judgement. Do not promote or regenerate any BigQuery table in this plan. Full evaluation methodology (Stages A–E, promote rule) lives in `docs/superpowers/specs/2026-08-31-embedding-similarity-eval-design.md`; this plan runs the subset that needs no new tooling: the component diagnostic (Stage A) and the bench A/B (Stage D), plus a manual neighbour-churn check (Stage E).

### Task 10: Train a candidate embedding

- [ ] **Step 1: Train under a new experiment version**

```bash
uv run python -m src.models.embeddings.train --algorithm pca --embedding-dim 64 --experiment game-embeddings
```
(Config now defaults to `pca`; the flag is explicit for the record.) This writes a new version under `models/experiments/embeddings/game-embeddings/`.

- [ ] **Step 2: Run the diagnostic on the new version**

```bash
uv run python -m src.models.embeddings.diagnose_components --experiment game-embeddings --version <new>
```
Pass criterion: no component with `min_prevalence_in_top < 0.005`. If some remain, raise `min_feature_count` (Task 6 param / config) and retrain; record where it settles.

- [ ] **Step 3: Compare explained variance old vs new** — expect a modest total drop; note it.

- [ ] **Step 4: Export embeddings for the bench**

Generate embeddings for the full catalog from the new version and load them into the bgg-viewer `/dev/similar` bench + analysis table (replace the dev artifact's `embedding` column, or point the bench dataset query at the new version's output table if it's been written to a scratch location).

- [ ] **Step 5: Bench evaluation** — record in the PR:
  - System Gateway isolation score: before vs after (expect: drops into normal range).
  - White Castle / Sankoré / Project L / Inferno: no longer a detached mutual cluster.
  - Panel of unambiguous games (Catan, Gloomhaven, Wingspan, Ark Nova, Codenames): neighbour lists recognisably same-or-better.

- [ ] **Step 6: Open the PR** — `feat/embedding-input-scaling` → `main`, with before/after diagnostic output and bench findings in the body. Stop. User reviews and merges.

---

## Follow-ups (not this plan)

- Promote the new embedding version, regenerate `raw.game_embeddings` → `analytics.game_similarity_search` → the bgg-data-warehouse `game_neighbors` profile → the 2D coordinates job. Spot-check neighbour lists.
- Evaluate `TwoSDScaler` in `create_bgg_preprocessor` for the linear rating models (own refit + CV comparison).
- Value clipping / winsorising, only if the diagnostic shows a specific continuous column distorting a component.
