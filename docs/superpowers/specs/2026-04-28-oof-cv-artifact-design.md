# OOF Cross-Validation Artifact Design

**Date:** 2026-04-28
**Status:** Draft (pending user review)
**Owner:** bgg-predictive-models — collection models

## Goal

Add stratified k-fold cross-validation on the training set, using the
already-tuned model's hyperparameters and threshold, as a new artifact
within each candidate run. Persist the per-row out-of-fold (OOF)
predictions and per-fold + overall metrics so OOF performance can be
compared alongside `val` and `test` in `compare_runs` /
`summarize_runs`.

## Motivation

Today, a candidate run reports val and test metrics — single-shot
estimates against splits the model never saw. We have no read on
*training-set generalization via resampling*. After choosing
hyperparameters on val and reporting on test, the natural follow-up
question is: *given those chosen hyperparameters and that threshold,
how does this model perform under stratified k-fold on the training
set?* That is a different question from "what params should we pick"
(tuning) and from "how does it do on held-out data" (val/test). It
gives us a third, resampling-based read alongside the existing two.

## Scope

**In scope**

- New `oof_cv_folds` field on `CollectionCandidate` (default `5`,
  `0` disables).
- New `oof_predict_cv` method on `CollectionModel` that runs k-fold
  on a training frame using the fitted pipeline's hyperparameters
  and the model's `self.threshold`.
- New OOF artifacts written under each candidate run:
  - `predictions/oof.parquet` — same shape as `predictions/val.parquet`
    (full input frame + `proba`/`pred`) plus a `fold` column.
  - `oof_metrics` block in `registration.json` carrying per-fold +
    overall metrics, fold count, seed, and the threshold used.
- `train_candidate` wires OOF CV in after threshold-finding,
  bundles results into `CandidateRunResult`.
- `save_candidate_run` (both `candidates.save_candidate_run` and
  `CollectionArtifactStorage.save_candidate_run`) accepts and persists
  the new artifacts.
- `compare_runs` / `summarize_runs` emit a third `split` value (`oof`)
  per candidate, sorted `val` → `oof` → `test`.
- CLI (`src.collection.train`) prints overall OOF metrics in its
  single-line JSON output.

**Out of scope**

- Per-fold threshold optimization (we use the fitted model's single
  threshold for OOF metrics — see "Threshold handling" below).
- Repeated CV / nested CV.
- OOF for the production-winner path (`{outcome}/v{N}/`); only
  candidate runs.
- Backfill of OOF artifacts onto existing runs.
- Changes to `tune_model_cv` (OOF is a separate, post-tuning pass).

## Design

### Per-candidate config

`CollectionCandidate` gains:

```python
oof_cv_folds: int = 5
```

Validated in `__post_init__`:

- `oof_cv_folds < 0` → `ValueError`.
- `oof_cv_folds == 1` → `ValueError` (CV needs ≥ 2 folds).
- `oof_cv_folds == 0` disables OOF for this candidate.

The field flows through `to_dict` / `from_dict` like other candidate
fields. `config.collections.candidates` entries that omit the field
get the default 5.

### `CollectionModel.oof_predict_cv`

New method on `CollectionModel`:

```python
def oof_predict_cv(
    self,
    train_df: pl.DataFrame,
    n_folds: int,
    *,
    random_seed: int = 42,
) -> tuple[pl.DataFrame, list[dict], dict]:
```

Pure compute — returns frames/dicts; persists nothing. Requires
`self.fitted_pipeline` set, and (for classification) `self.threshold`
set; raises `RuntimeError` otherwise. Also raises `ValueError` if
`train_df.height < n_folds * 2`, or if any fold would have zero
positives for classification (let `StratifiedKFold` propagate its
error).

Behavior:

1. Build the splitter: `StratifiedKFold` for classification (split on
   `label`), plain `KFold` for regression — `shuffle=True`,
   `random_state=random_seed`. Matches `tune_model_cv`'s convention.
2. Read the hyperparameters off `self.fitted_pipeline.named_steps["model"]`
   via `get_params(deep=False)`. These are the params that were tuned;
   OOF must use exactly those.
3. For each fold:
   - Clone the pipeline structure with `build_pipeline()`; apply the
     captured params; refit on the fold's training rows. Preprocessor
     refit per fold to avoid leakage (mirrors `tune_model_cv`).
   - Predict on the holdout rows. Classification: `proba`, then
     `pred = (proba >= self.threshold).astype(int)`. Regression:
     `pred = pipeline.predict(...)`.
   - Compute fold metrics with the existing `_evaluate_classification` /
     `_evaluate_regression` helpers (reusing them keeps metric vocab
     consistent across `val`/`test`/`oof`). Stash the fold's holdout
     row indices, predictions, and metrics.
4. Reassemble OOF predictions: take `train_df`, attach a `fold` column
   (the fold each row was held out in), `proba` (classification only),
   and `pred`. Output is the same shape as `predictions/val.parquet`
   plus the `fold` column. One OOF prediction per training row.
5. Compute the **pooled** overall metrics by running the metric
   functions on the concatenated OOF predictions (not by averaging
   fold metrics — pooled is the standard OOF score and matches what
   the val/test rows in `summarize_runs` carry).

Returns `(oof_predictions, per_fold_metrics, overall_metrics)`:

- `oof_predictions`: `pl.DataFrame` — full `train_df` with `fold`,
  `proba` (classification), `pred` columns appended.
- `per_fold_metrics`: `list[dict]` — one entry per fold with keys
  `fold`, `n_rows`, `n_pos` (classification only), and the same
  metric keys as `evaluate()`.
- `overall_metrics`: `dict` — pooled metrics across all OOF rows,
  same keys as `evaluate()`.

### `CandidateRunResult` extension

Two new optional fields:

```python
oof_predictions: Optional[pl.DataFrame] = None
oof_metrics: Optional[Dict[str, Any]] = None
```

`oof_metrics` shape:

```python
{
    "n_folds": 5,
    "seed": 42,
    "stratified_on": "label" | None,        # None for regression
    "threshold": float | None,                # None for regression
    "overall": {<metric>: <float>, ...},      # pooled
    "per_fold": [{"fold": 0, "n_rows": ..., ...metrics}, ...],
}
```

Both fields are `None` when `candidate.oof_cv_folds == 0`.

### `train_candidate` wiring

Insert immediately after `model.find_threshold(val_df)` (classification)
or immediately after the tuning step (regression), before val/test
evaluation:

```python
oof_predictions = None
oof_metrics = None
if candidate.oof_cv_folds and candidate.oof_cv_folds > 0:
    oof_predictions, per_fold, overall = model.oof_predict_cv(
        train_used, n_folds=candidate.oof_cv_folds,
    )
    oof_metrics = {
        "n_folds": candidate.oof_cv_folds,
        "seed": 42,
        "stratified_on": "label" if outcome.task == "classification" else None,
        "threshold": model.threshold,
        "overall": overall,
        "per_fold": per_fold,
    }
```

Both flow into the returned `CandidateRunResult`.

### Persistence

`candidates.save_candidate_run` extends the registration with the new
block:

```python
registration["oof_metrics"] = result.oof_metrics    # may be None
```

It then forwards two new arguments to
`CollectionArtifactStorage.save_candidate_run`:

```python
storage.save_candidate_run(
    ...,
    oof_predictions=result.oof_predictions,
    oof_metrics=result.oof_metrics,
)
```

`CollectionArtifactStorage.save_candidate_run` adds the corresponding
parameters and writes:

- `predictions/oof.parquet` ← `oof_predictions` (skipped if `None`).
- The `oof_metrics` dict is *not* written to a separate file; it is
  included in `registration.json` by `candidates.save_candidate_run`
  before delegating to the storage layer. The storage layer therefore
  gains only one new parameter (`oof_predictions`); `oof_metrics`
  flows through `registration` as it does today for `val_metrics` /
  `metrics`.

Resulting on-disk layout (additions marked NEW):

```
{outcome}/{candidate}/v{N}/
    model.pkl
    threshold.json
    registration.json                   # gains "oof_metrics" block (NEW)
    tuning_results.parquet
    train_used.parquet
    feature_importance.parquet
    predictions/
        val.parquet
        test.parquet
        oof.parquet                     # NEW — full train_used + fold/proba/pred
```

### Comparison surface

`candidate_comparison.compare_runs` extends the per-run loop to read
`registration["oof_metrics"]["overall"]` (when present) and emit rows
with `split="oof"`. The `threshold`, `best_params`, `n_train_used`,
and other metadata in `common` already apply unchanged (the OOF
predictions were generated using the same fitted model and the same
threshold).

`summarize_runs` ordering changes from "val before test alphabetically"
to an explicit `val` → `oof` → `test` order. Implementation: add a
`split_order` key column (0/1/2) on the wide frame before sorting,
sort by `(candidate, split_order)`, drop the helper column. This
yields the narrative `val` (tuned) → `oof` (CV on train) → `test`
(generalization).

### CLI output

`src.collection.train` prints a single-line JSON. Add an
`oof_metrics_overall` field carrying just the pooled metrics dict
(or `null` when OOF is disabled), so callers can read CV performance
without loading parquet:

```json
{
  "candidate": "lgbm_default",
  "outcome": "own",
  "version": 7,
  "splits_version": 3,
  "threshold": 0.04,
  "val_metrics": {...},
  "test_metrics": {...},
  "oof_metrics_overall": {...},
  "n_train": 32411, "n_val": 1456, "n_test": 1457,
  "artifact_dir": "..."
}
```

## Data flow

```
load canonical splits
  └─ train_candidate(...)
       ├─ slice features / downsample           → train_used
       ├─ tune (or fit fixed params)            → fitted_pipeline, best_params
       ├─ find_threshold(val_df)                → self.threshold (classification)
       ├─ oof_predict_cv(train_used, n_folds)   → oof_predictions, oof_metrics  (NEW)
       ├─ evaluate(val) / evaluate(test)        → val_metrics, test_metrics
       ├─ predict_with_labels(val/test)         → val_predictions, test_predictions
       └─ feature_importance()
  └─ save_candidate_run(result)
       ├─ model.pkl, threshold.json, registration.json (with oof_metrics block)
       ├─ tuning_results.parquet, train_used.parquet, feature_importance.parquet
       └─ predictions/{val,test,oof}.parquet                              (oof NEW)
```

## Error handling

- `oof_predict_cv` requires `self.fitted_pipeline` and (for
  classification) `self.threshold`. Missing → `RuntimeError` with the
  same message style as `_require_fitted`.
- `oof_cv_folds` validation in `CollectionCandidate.__post_init__`
  catches negative values and the degenerate `n_folds == 1`.
- `train_used.height < n_folds * 2` raises `ValueError` with a clear
  message before the CV loop starts.
- Stratification edge cases (a class with too few rows for the
  requested number of folds) propagate from `StratifiedKFold` —
  matches existing tuner behavior.
- `train_candidate` does not catch OOF errors. If OOF fails, the run
  fails. Continue-on-error semantics belong at the CLI level (already
  the case for tuning).

## Testing

New tests under `tests/collection/`:

1. **Unit — classification.** Build a small classification frame, fit
   a `CollectionModel` with `tune="none"` and `find_threshold`, call
   `oof_predict_cv(n_folds=3)`. Assert:
   - `oof_predictions.height == train_df.height`.
   - Every row has a non-null `proba`, `pred`, `fold`.
   - `fold` values are exactly `{0, 1, 2}` and partition the rows.
   - `overall_metrics` keys match `evaluate()` keys.
   - Re-running with the same seed produces identical predictions
     (determinism).
2. **Unit — regression.** Same shape, but with a regression outcome.
   Assert no `proba` column, plain `KFold` is used (no stratification
   required), pooled `rmse`/`mae`/`r2` reported.
3. **Integration — train_candidate + save_candidate_run.** Run a tiny
   end-to-end candidate with `oof_cv_folds=3`, save, then read back
   `predictions/oof.parquet` and `registration.json`. Assert OOF
   parquet has the expected columns and that `registration.json["oof_metrics"]`
   carries the per-fold + overall blocks.
4. **Integration — disabled.** Same flow with `oof_cv_folds=0`.
   Assert `predictions/oof.parquet` is not written and the
   registration's `oof_metrics` is `None`.
5. **Comparison.** Two saved candidate registrations (one with
   `oof_metrics`, one without) → `compare_runs` returns `oof` rows
   only for the candidate that has them; `summarize_runs` orders
   `val` → `oof` → `test` for that candidate.

Existing tests must continue to pass unchanged.

## Risks and trade-offs

- **Compute cost.** Adds `n_folds` extra fits per candidate (default
  5). With LightGBM on the current frame this is acceptable; for
  expensive candidates the user can set `oof_cv_folds=0` to disable.
- **Threshold consistency.** Using `self.threshold` (tuned on val) for
  OOF threshold-based metrics means OOF and val/test all use the same
  cutoff. This is the right choice for the "given this tuned model,
  how does it CV?" question. A user who wants per-fold optimal
  thresholds can recompute from `predictions/oof.parquet` (which
  carries the probabilities).
- **Pooled vs averaged overall metric.** We pool predictions across
  folds before computing the overall metric (rather than averaging
  per-fold scores). Pooled is the canonical OOF score, matches the
  shape of val/test metrics, and is what shows up in
  `summarize_runs` next to them. Per-fold variance is still inspectable
  via `oof_metrics.per_fold` in the registration.
- **Sort order change.** `summarize_runs` already had `val` before
  `test`. The new order is `val` → `oof` → `test`. Anyone scripting
  off the existing two-row-per-candidate output will need to handle
  the third row. Acceptable: the function is internal-only.
