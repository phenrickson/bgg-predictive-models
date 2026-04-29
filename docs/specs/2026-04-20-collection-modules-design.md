# Collection Modules Design

**Date:** 2026-04-20
**Status:** Design approved, ready for implementation plan

## Goal

Refactor `src/collection/` into a coherent module structure that supports training multiple user-specific outcome models (own, ever_owned, rated, rating, love) from a single configuration. The existing code has scaffolding for a single binary `owned` outcome; this design generalizes to any outcome defined declaratively in config.

Scope stops at the training side. Serving (`services/collections/`) is a follow-up plan.

## Non-goals

- Reproducing the Quarto-style user report (follow-up)
- Standing up `services/collections/` FastAPI endpoints (follow-up)
- Per-outcome feature engineering (YAGNI — all outcomes use the same feature set initially)
- Expanding test coverage beyond what's needed to validate the new abstractions

## Module map

Six modules in `src/collection/`:

| Module | File | Responsibility |
|---|---|---|
| **CollectionLoader** | `collection_loader.py` | Fetch raw collection from BGG XML API (unchanged) |
| **CollectionStorage** | `collection_storage.py` | BQ read/write for raw collections (unchanged) |
| **ArtifactStorage** | `collection_artifact_storage.py` | GCS storage for models/predictions/splits/analysis, keyed by (username, outcome, version) |
| **CollectionProcessor** | `collection_processor.py` | Join raw collection with game universe in BigQuery; produce unlabeled dataframe. Absorbs current `collection_integration.py` and the subtype-filter work from the existing thin `collection_processor.py`; `collection_integration.py` is deleted. |
| **CollectionSplitter** | `collection_split.py` | Train/val/test splits; dispatches to classification or regression strategy based on outcome |
| **CollectionModel** | `collection_model.py` | Train one model for one outcome; dispatches to classification or regression path |
| **Outcomes** (new) | `outcomes.py` | `OutcomeDefinition` dataclass + config loader + label application |

Orchestration stays in `collection_pipeline.py`, which loops over outcomes from config and invokes the per-outcome modules.

## Outcomes module

### Config schema

Outcomes are declared under `collection.outcomes` in the top-level `config.yaml`:

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

Four primitives cover all 5 outcomes:

- `label_from: <column>` — copy column directly as the label
- `label_from: {any_of: [col_a, col_b, ...]}` — boolean OR across columns
- `label_from: {column: X, predicate: "<op> <value>"}` — threshold/comparison producing a boolean label; supported operators: `>`, `>=`, `<`, `<=`, `==`, `!=`
- `require: "<col> <op> <value>"` — optional row filter; rows failing the predicate are dropped before labeling

### Python surface

```python
@dataclass
class OutcomeDefinition:
    name: str
    task: Literal["classification", "regression"]
    label_rule: LabelRule      # parsed from one of the 4 primitives
    require: str | None = None

def load_outcomes(config: dict) -> dict[str, OutcomeDefinition]:
    """Parse config.collection.outcomes into a registry of OutcomeDefinitions."""

def apply_outcome(df: pl.DataFrame, outcome: OutcomeDefinition) -> pl.DataFrame:
    """Apply `require` row filter, then add a `label` column from `label_rule`."""
```

Predicate and require strings are parsed with a small hand-written parser supporting only the fixed operator set — no `eval`.

## Module details

### CollectionLoader

No change. Fetches BGG XML API, parses collection status fields (owned, prev_owned, wishlist, user_rating, subtype, etc.), handles retries. Requires `BGG_API_TOKEN`.

### CollectionStorage (BQ raw)

No change. Stores/retrieves versioned raw collection snapshots per user. Raw storage is not outcome-partitioned — one raw snapshot serves all outcomes.

### ArtifactStorage (GCS)

Add `outcome` parameter to all artifact save/load methods. Path layout is per-user sub-path:

```
gs://{bucket}/{env}/collections/{username}/{outcome}/{version}/
    model.pkl
    predictions.parquet
    splits.parquet
    analysis.json
```

Versions are tied to experiment runs. Retraining `love` creates a new `love/v{N}/` without affecting `own/v{M}/`.

### CollectionProcessor

Outcome-agnostic. Joins raw collection with game universe features using a BigQuery query (not polars in-memory). Returns one unlabeled, game-universe-joined dataframe per user. Applies generic preprocessing that is common across outcomes (subtype filtering, type normalization, any warehouse-side feature hydration needed).

Replaces the current `collection_integration.py` and absorbs the subtype-filter work from the current `collection_processor.py`.

```python
processor.process(username) -> pl.DataFrame
```

The Pipeline invokes Processor once per user, then calls `apply_outcome(df, outcome)` per outcome.

### CollectionSplitter

Two internal strategies, one public dispatcher:

- **ClassificationSplitter** — existing `collection_split.py` logic: negative sampling (random / popularity_weighted / uniform), min-ratings thresholds, time-based or stratified random splits.
- **RegressionSplitter** — splits only rated games (rows where `require: user_rating > 0` already applied); no negative sampling; stratify by rating value (binned).

```python
CollectionSplitter.split(df, outcome) -> (train, val, test)
```

Dispatches on `outcome.task`.

`SplitConfig` forks: classification-only fields (`negative_sampling_ratio`, `min_ratings_for_negatives`, `negative_sampling_strategy`, `handle_imbalance`) live in a classification sub-config; regression gets its own minimal config (validation_ratio, test_ratio, stratification bin count, random_seed).

### CollectionModel

Two internal paths, one public train method:

- **Classification path** — existing LightGBM/CatBoost/LogReg classifier behavior: threshold optimization, class-weighted training, classification metrics (AUC, F1, F2, precision, recall, log loss).
- **Regression path** — LightGBM/CatBoost regressors; regression metrics (RMSE, MAE, R²); no threshold optimization.

```python
CollectionModel.train(splits, outcome, model_config) -> trained_model
```

Dispatches on `outcome.task`.

`ModelConfig` forks by task. Classification-only fields (`handle_imbalance`, `threshold_optimization_metric`) move to a classification sub-config; regression gets its own metric knobs.

Reuses `src.models.training.create_preprocessing_pipeline` and `tune_model` for both paths (they already support regressors).

### CollectionPipeline

Top-level orchestrator. Structure:

```python
def run_full_pipeline(username: str):
    raw = loader.fetch(username)
    storage.save_raw(username, raw)

    processed = processor.process(username)   # one join per user

    for outcome in load_outcomes(config):
        labeled = apply_outcome(processed, outcome)
        splits = splitter.split(labeled, outcome)
        model = CollectionModel.train(splits, outcome, model_config)
        preds = predict(model, processed)
        artifact_storage.save(username, outcome, model=model, predictions=preds, ...)
```

`refresh_predictions_only(username)` follows the same loop but skips training: reloads each outcome's registered model and regenerates predictions.

## Orchestration: Makefile

No new CLI. Top-level interface is Makefile targets that invoke `collection_pipeline` as a module:

```makefile
train-collection:
	uv run -m src.collection.collection_pipeline run --username $(USERNAME) $(if $(OUTCOME),--outcome $(OUTCOME),)

refresh-collection:
	uv run -m src.collection.collection_pipeline refresh --username $(USERNAME)

collection-status:
	uv run -m src.collection.collection_pipeline status --username $(USERNAME)
```

Usage:

```
make train-collection USERNAME=phenrickson                    # all outcomes
make train-collection USERNAME=phenrickson OUTCOME=love       # single outcome
make refresh-collection USERNAME=phenrickson
```

The existing `cli.py` stays as the implementation detail the Makefile invokes.

## Versioning

Versions are per (username, outcome), tied to experiment runs. Each `CollectionModel.train` call produces a new experiment version; ArtifactStorage writes under `{outcome}/v{N}/`. Retraining one outcome (e.g. after a config change to the `love` threshold) bumps only that outcome's version.

Experiment tracking uses the existing `src.models.experiments.ExperimentTracker` infrastructure.

## Data flow summary

```
BGG API
   │
   ▼
CollectionLoader ──▶ CollectionStorage (BQ raw)
                          │
                          ▼
                   CollectionProcessor ─── joins raw collection + game universe in BQ
                          │
                          ▼  (unlabeled dataframe, one per user)
              ┌──────────┴──────────┬──────────┬──────────┬──────────┐
              ▼                     ▼          ▼          ▼          ▼
        apply_outcome(own)    (ever_owned)   (rated)   (rating)   (love)
              │                     │          │          │          │
              ▼                     ▼          ▼          ▼          ▼
        CollectionSplitter  ◀── dispatches on outcome.task
              │
              ▼
        CollectionModel     ◀── dispatches on outcome.task
              │
              ▼
        ArtifactStorage (GCS) — {username}/{outcome}/v{N}/
```

## Open considerations (deferred)

- **Serving surface** (`services/collections/`): out of scope for this plan; will be a follow-up after the training-side modules stabilize.
- **Per-outcome feature engineering:** deferred pending evidence that shared features hurt any outcome's performance.
- **Quarto-style user report:** deferred; the current design produces the artifacts such a report would consume (models, predictions, analysis JSON), but the report generation itself is a separate concern.
- **Batch training across users:** the design supports it trivially (loop over usernames in the Makefile or orchestrator) but is not built into the pipeline.
- **Orchestration interface:** Makefile targets are the chosen starting point, but may be revisited as the workflow grows (e.g. scheduled multi-user training, richer parameter surfaces). Not a blocker for this plan.
