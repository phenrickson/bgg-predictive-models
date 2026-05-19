# BGG Model Snapshots and Splits

**Status:** Design
**Date:** 2026-05-08
**Scope:** `src/models/` (bgg-rating-models)

## Problem

Today, every run of `src/models/outcomes/train.py` regenerates its train/tune/test
splits inline from `config.yaml` years. Two consequences:

- Two experiments are only "comparable" if both happened to read the same year
  config and the same feature snapshot. Nothing records what data they actually
  saw.
- Year-over-year evaluation, headline-split training, and any future split
  variation each compete for the same flat `models/experiments/{model_type}/{name}/`
  namespace, so cross-cutting analysis ("how does this rating recipe behave
  across years?") requires manual reassembly.

The collection-models project solved a related problem with canonical splits
(`{outcome}/_splits/v{N}/`) that multiple candidates share. We want the same
discipline for bgg-rating-models, but extended to handle the cascaded model
chain (complexity → rating + users_rated → geek_rating) and the multi-split
nature of YoY evaluation.

## Goals

- **Stable data for clean comparisons, today and in the future.** Fix the
  feature universe at a point in time so two experiments — whether run a day
  apart or a year apart — see identical bytes. The upstream
  `bgg-data-warehouse` `games_active` table is continuously refreshed; a game
  published in 2018 has different feature values today than it did a month
  ago (ratings accumulate, designers and categories get added, complexity
  weights tick up). Without a fixed snapshot, "comparable" experiments are
  quietly using different data.
- A model artifact's location on disk encodes what it's comparable to.
- Cascading dependencies (rating reads predicted_complexity) resolve within a
  fixed evaluation surface — never across surfaces.
- **Honest training-time features for cascaded models.** The
  `predicted_complexity` column that rating trains on must come from a
  complexity model that did not see those rows during training. Today's
  in-sample scoring (finalized complexity model predicts every row, including
  ones it was trained on) gives downstream models an over-optimistic feature
  at training time. Out-of-fold predictions on the train fold remove that
  leak.
- Adding a new split (e.g. a new YoY year) is additive; existing experiments
  stay put.
- A candidate is a recipe with one home; per-split results live under it.

## Non-Goals

- Migrating the existing `models/experiments/{model_type}/{name}/` tree. New
  layout is greenfield. Existing artifacts remain readable in place but are
  not maintained.
- Out-of-fold cascaded scoring. Today's "finalized model scores everything"
  approach is preserved as the default. OOF is a future extension.
- Streamlit / dashboard updates. The streamlit pages and Quarto reports that
  read the existing tree continue to do so. Adapting them to the new layout
  is follow-on work.
- Cloud/GCS sync of the new tree. Local-only initially; sync utilities follow
  the same pattern as today and are not part of this spec.

## Concepts

Three nested units, in order of decreasing scope:

1. **Snapshot** — the feature universe at a point in time. The actual data.
   `models/experiments/_snapshots/v{N}/universe.parquet` is the canonical bytes.
   A new snapshot version gets cut whenever the underlying data changes
   meaningfully (BQ refresh, new embeddings, new feature columns). The user
   decides when to bump it. Once built, a snapshot is immutable and
   self-contained — the only step that talks to BigQuery is the snapshot
   build itself; all training and evaluation downstream reads from the
   snapshot's parquet files.

2. **Split** — a slicing rule applied to a snapshot, producing
   `(train, tune, test)` folds. Lives under
   `_snapshots/v{N}/splits/{split_name}/`. Multiple splits can live under one
   snapshot. Splits are pure derivations of the snapshot — given the same
   snapshot bytes and the same slicing rule, the split bytes are identical.

3. **Experiment** — a candidate recipe (algorithm, preprocessor config, etc.)
   trained against one or more splits within a snapshot. Lives under
   `_snapshots/v{N}/experiments/{model_type}/{candidate_name}/v{M}/`. Per-split
   training results live in `results/{split_name}/`.

The unit of comparability is `(snapshot_version, split_name)`. Two experiments
trained against the same snapshot+split are honestly comparable: same data,
same fold definitions, same upstream cascade options.

### Terminology note: "scoring"

Throughout this spec, "scoring" refers to running a trained model against the
snapshot's universe to produce `predicted_*` columns that downstream models
need as features (e.g. complexity scores feed rating training). This is an
*evaluation-time* artifact and lives inside the snapshot tree.

The operational/production scoring service that runs against future games and
writes to the BigQuery landing table is a separate concern and is *not*
affected by this spec.

## Layout

```text
models/experiments/_snapshots/
  v1/
    universe.parquet                            # full feature+outcome+id frame
    metadata.json                               # source info, created_at, n_rows, columns

    splits/
      standard/
        train.parquet
        tune.parquet
        test.parquet
        metadata.json                           # train_through, tune_*, test_*, time_col
      yoy_2018/
        train.parquet, tune.parquet, test.parquet, metadata.json
      yoy_2019/
        ...
      yoy_2024/
        ...

    experiments/
      complexity/
        ard-complexity/
          v1/
            config.json                         # frozen recipe (algorithm, preprocessor_kwargs, ...)
            registration.json                   # snapshot_version, candidate, created_at
            finalized.pkl                       # one finalized pipeline per candidate (refit on full snapshot)
            results/
              standard/
                pipeline.pkl                    # trained on standard's train fold
                metrics.json                    # train, tune, test metrics
                parameters.json                 # tuned hyperparameters
                feature_importance.csv
                predictions/
                  tune.parquet
                  test.parquet
                  score.parquet                 # finalized model on full snapshot — feeds rating/users_rated training
              yoy_2018/
                pipeline.pkl, metrics.json, parameters.json, feature_importance.csv
                predictions/{tune,test,score}.parquet
              yoy_2019/...
            summary.json                        # cross-split rolled-up metrics

      rating/
        ard-ridge-rating/
          v1/
            config.json
            registration.json                   # records upstream_experiments: {complexity: ard-complexity/v1}
            finalized.pkl
            results/
              standard/
                pipeline.pkl, metrics.json, ...
                predictions/{tune,test,score}.parquet
              yoy_2018/...
            summary.json

      users_rated/...
      geek_rating/...
      hurdle/...
```

### Key conventions

- Snapshot versions are integers (`v1`, `v2`), bumped manually when data
  changes meaningfully.
- Split names are lowercase, underscored (`standard`, `yoy_2018`). The
  `standard` name is the default headline split.
- Candidate names mirror `config.yaml` (e.g. `ard-ridge-rating`).
- A candidate's `v{M}` version is independent across model types — a rating
  candidate `v3` and a complexity candidate `v3` are unrelated.
- `score.parquet` contains the finalized model's predictions on every row of
  the snapshot's `universe.parquet` (option A — in-sample for rows the model
  was trained on; this is today's behavior, preserved).
- `finalized.pkl` lives at the candidate level, not under any one split. It is
  the deployable artifact — the candidate's pipeline refit on the full
  snapshot universe through some `final_end_year`.

## Out-of-fold scoring

`score.parquet` for an upstream candidate (e.g. complexity) under a given
split contains `predicted_*` columns for every row of the snapshot universe.
The columns are produced by *different* fitted models depending on the row,
so that downstream training never consumes an over-optimistic prediction.

For a candidate trained on split `S` with train fold rows `T`, tune fold
`V`, test fold `E`:

| Rows | Source of predictions |
| --- | --- |
| Train rows (`T`) | K-fold OOF: split `T` into K folds, train K models, each predicts its held-out fold. Stitched together. |
| Tune rows (`V`) | The model trained on the full `T`. (Already held-out from `T`'s perspective.) |
| Test rows (`E`) | The model trained on the full `T`. |
| Rows outside `T ∪ V ∪ E` | The finalized model (refit on `T ∪ V ∪ E`, or through whatever `final_end_year` was used). |

K defaults to 5 and is configurable per candidate (in `config.yaml`). The OOF
fold assignment uses a deterministic seed so re-running produces identical
predictions.

The cost is K extra trainings of the candidate per split. For a 5-fold
default and a typical workflow (1 standard split + 7 YoY splits), that's
~40 K-fold trainings per upstream candidate. Worth it for honest downstream
features; worth deferring to runtime config if a candidate is too expensive
to fold (in which case the candidate can declare `oof_folds: 1` to fall back
to in-sample scoring for the train rows, with the understanding that
downstream models trained on its `score.parquet` will be biased).

The OOF machinery applies to `complexity`, `rating`, and `users_rated` —
the upstream models in the cascade. Hurdle and geek_rating are downstream
of others (or have no downstream consumers in training), so they only need
held-out predictions for evaluation, not OOF for the train rows. They
still produce a `score.parquet` for completeness, but the train rows there
are in-sample (and unused by anything downstream).

## Cascading dependencies

Within a single `(snapshot_version, split_name)` surface, downstream training
reads upstream `score.parquet` from a sibling experiment dir. Concretely:

When training `rating/ard-ridge-rating/v1` on `_snapshots/v1/splits/standard/`:

1. Trainer loads `_snapshots/v1/splits/standard/{train,tune,test}.parquet`
2. Trainer reads `_snapshots/v1/experiments/complexity/ard-complexity/v1/results/standard/predictions/score.parquet`
3. Trainer joins `predicted_complexity` onto train/tune/test by `game_id`
4. Train rating
5. Write rating's results to `_snapshots/v1/experiments/rating/ard-ridge-rating/v1/results/standard/`

Rating's `registration.json` records the upstream choice:

```json
{
  "snapshot_version": 1,
  "candidate": "ard-ridge-rating",
  "version": 1,
  "created_at": "2026-05-08T...",
  "upstream_experiments": {
    "complexity": "ard-complexity/v1"
  }
}
```

This makes the dependency chain explicit and auditable. Within one snapshot,
multiple complexity candidates can coexist, and a rating run picks one and
records the choice.

Cross-snapshot contamination is structurally impossible: a rating run under
`_snapshots/v1` cannot reach into `_snapshots/v2`'s complexity output.

The same cascade rule applies to every dependency:

- `rating` and `users_rated` depend on `complexity`
- `geek_rating` depends on `complexity`, `rating`, and `users_rated`
- `hurdle` has no upstream dependencies

For a candidate to be runnable on a given split, all its upstream candidates
must already have a `score.parquet` for that split. The trainer checks this
upfront and errors if it's missing.

## Candidate definitions in config.yaml

Mirroring `collections.candidates`. Each model type gets a `candidates` list:

```yaml
models:
  complexity:
    candidates:
      - name: ard-complexity
        algorithm: ard
        use_embeddings: true
        use_sample_weights: false
        preprocessor_kwargs: {...}
      - name: catboost-complexity
        algorithm: catboost
        ...

  rating:
    candidates:
      - name: ard-ridge-rating
        algorithm: ard
        min_ratings: 5
        upstream:
          complexity: ard-complexity        # default upstream choice
        ...
```

When a candidate is run, the trainer copies its YAML block into the
experiment's `config.json`. Future edits to the central config don't change
existing experiment dirs — every experiment is a frozen record of the recipe
that ran.

The `upstream` block is a default that the CLI can override (`--upstream
complexity=catboost-complexity`).

## Storage layer

A new module — call it `src/models/snapshot_storage.py` — owns paths and IO,
mirroring `CollectionArtifactStorage` in shape. Public surface:

- `SnapshotStorage(snapshot_version: int, base_dir="models/experiments/_snapshots")`
- `save_universe(df) -> path`
- `load_universe() -> df`
- `save_metadata(meta) / load_metadata() -> dict`
- `save_split(split_name, train, tune, test, meta) -> dict[paths]`
- `load_split(split_name) -> dict[fold -> df]`
- `list_splits() -> list[str]`
- `save_experiment_run(model_type, candidate, version, ...)` — wraps the
  current `ExperimentTracker.create_experiment` + `Experiment.log_*` flow
- `load_experiment_run(model_type, candidate, version=None)`
- `list_experiments(model_type=None) -> list[dict]`
- `read_upstream_score(model_type, candidate, split_name) -> df`

`ExperimentTracker` and `Experiment` are reused for the per-result
artifacts; `SnapshotStorage` resolves paths and delegates write/read to
them. This keeps existing experiment-logging code (metrics, predictions,
threshold analysis, calibration curves, diagnostic plots) intact.

## CLI surface

Three new entry points, plus a modified train command:

### `src/models/build_snapshot.py`

```bash
uv run python -m src.models.build_snapshot \
  [--snapshot-version N]   # default: next available
  [--use-embeddings]
  [--local-data PATH]
```

Loads features (BQ or local), joins embeddings if requested, writes
`_snapshots/v{N}/universe.parquet` and `metadata.json`. No filtering by
year — the snapshot is the full universe; year filters happen at split time.

### `src/models/build_split.py`

```bash
uv run python -m src.models.build_split \
  --snapshot-version N \
  --split-name standard \
  [--train-through 2022 --tune-start 2023 --tune-through 2023 \
   --test-start 2024 --test-through 2024]   # default: from config.yaml years.training
```

Reads `_snapshots/v{N}/universe.parquet`, slices by year, writes
`_snapshots/v{N}/splits/{split_name}/{train,tune,test}.parquet` and
`metadata.json`.

A second mode generates the YoY family in one call:

```bash
uv run python -m src.models.build_split \
  --snapshot-version N \
  --yoy --yoy-start 2018 --yoy-end 2024
```

Equivalent to running the single-split form once per year with the
naming convention `yoy_{year}`.

### Modified `src/models/outcomes/train.py`

The training entry point gains snapshot/split awareness. Replace the existing
year arguments with:

```bash
uv run python -m src.models.outcomes.train \
  --model rating \
  --candidate ard-ridge-rating \
  --snapshot-version N \
  --splits standard,yoy_2018,yoy_2019,...   # default: standard
  [--upstream complexity=ard-complexity]
```

For each requested split:

1. Load the split's `(train, tune, test)` from snapshot storage
2. Resolve and join upstream `score.parquet` files
3. Train, tune, evaluate (existing pipeline code in `src/models/training.py`)
4. Write `results/{split_name}/` artifacts via `SnapshotStorage`

After all splits succeed, write `summary.json` rolling up cross-split
metrics, and (if `--finalize` given) refit on the full snapshot universe and
write `finalized.pkl` at the candidate level.

### `src/models/score_universe.py`

```bash
uv run python -m src.models.score_universe \
  --snapshot-version N \
  --splits standard,yoy_2018,...
  --model complexity \
  --candidate ard-complexity
```

Runs the per-split `score.parquet` step for the named candidate. Required
between training upstream and downstream models, because rating training
under a split needs `score.parquet` from complexity under the same split.

(In practice this can be folded into the train command — train rating implies
"score complexity if missing" — but the standalone command is useful for
re-scoring with a new finalize cutoff.)

## Reference flow: end-to-end training of the full chain

```bash
# 1. Build the snapshot once
uv run python -m src.models.build_snapshot --use-embeddings

# 2. Build splits off it
uv run python -m src.models.build_split --snapshot-version 1 --split-name standard
uv run python -m src.models.build_split --snapshot-version 1 --yoy --yoy-start 2018 --yoy-end 2024

# 3. Train upstream models (no dependencies)
uv run python -m src.models.outcomes.train --model hurdle --candidate logistic-hurdle \
  --snapshot-version 1 --splits standard,yoy_2018,...,yoy_2024

uv run python -m src.models.outcomes.train --model complexity --candidate ard-complexity \
  --snapshot-version 1 --splits standard,yoy_2018,...,yoy_2024

# 4. Train mid-tier models (depend on complexity)
uv run python -m src.models.outcomes.train --model rating --candidate ard-ridge-rating \
  --snapshot-version 1 --splits standard,yoy_2018,...,yoy_2024 \
  --upstream complexity=ard-complexity

uv run python -m src.models.outcomes.train --model users_rated --candidate ard-ridge-users_rated \
  --snapshot-version 1 --splits standard,yoy_2018,...,yoy_2024 \
  --upstream complexity=ard-complexity

# 5. Train geek_rating (depends on all three)
uv run python -m src.models.outcomes.train --model geek_rating --candidate ard-geek_rating \
  --snapshot-version 1 --splits standard,yoy_2018,...,yoy_2024 \
  --upstream complexity=ard-complexity,rating=ard-ridge-rating,users_rated=ard-ridge-users_rated
```

After this, `_snapshots/v1/experiments/` contains the full chain across the
headline split and seven YoY splits — all honestly comparable to each other,
because they share the same snapshot and the same fold definitions.

Adding a new candidate (e.g. `catboost-rating`) is the same workflow — it
slots into `experiments/rating/catboost-rating/v1/` and produces a parallel
set of results comparable to `ard-ridge-rating/v1`.

## Comparability semantics

The framework guarantees the following:

- **Within `(snapshot_version, split_name)`:** any two experiments are
  comparable on `metrics.json` for that split. They saw identical bytes for
  train/tune/test.
- **Across splits within a snapshot:** comparing a candidate's `summary.json`
  vs another candidate's `summary.json` is meaningful — both ran on the same
  snapshot data, sliced the same ways.
- **Across snapshots:** not comparable. Different `universe.parquet` =
  different data. The framework deliberately makes cross-snapshot comparison
  awkward to discourage it.

## Out of scope / future work

- **Snapshot diffing tools.** "What changed between v1 and v2?"
- **Snapshot/split sync to GCS.** Same pattern as existing experiments sync;
  out of scope here.
- **Streamlit and Quarto report adaptation.** Existing dashboards continue
  reading from the legacy tree; adapting them to read from `_snapshots/`
  is follow-on work.
- **Migration of legacy experiments.** Existing
  `models/experiments/{model_type}/{name}/v{N}/` artifacts stay where they
  are; new work goes into the snapshot tree.
- **Score/deploy artifacts (post-finalize, post-2026 games going to BQ).**
  Operational scoring continues through the existing scoring service; the
  snapshot tree is for evaluation, not for production prediction outputs.

## Open questions

None blocking. Items deferred to implementation:

- Exact JSON schemas for `metadata.json` at each level (snapshot, split,
  experiment registration) — finalize when writing the storage module.
- How `--upstream` defaults interact with `config.yaml`'s per-candidate
  `upstream` block — finalize when writing the train CLI.
