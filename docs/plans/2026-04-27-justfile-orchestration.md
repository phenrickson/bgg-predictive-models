# Justfile Orchestration for Collection Candidate Workflow

**Goal:** A `just`-based command-line entrypoint over the existing
collection candidate workflow so experiments can be run from
scripts/cron/CI without invoking Python directly. Scope is wrapping what
already exists, plus a small additive change to `CollectionCandidate`
(new `load_candidates(config)` loader) and a public rename on
`CollectionArtifactStorage` (`_next_version` → `next_version`).

**Coexistence with the existing Makefile:** The collection-related Make
targets (`train-collection`, `refresh-collection`, `collection-status`)
wrap the *production-winner* pipeline in `src/collection/cli.py` and stay
untouched. This work adds a *parallel* `just` workflow for
*candidate experimentation*. The two have different consumers and
different output paths (production v{N}/ vs. per-candidate
{candidate}/v{N}/).

**Architecture:** Four CLI modules under `src/collection/` (verb-named to
match the existing `src/pipeline/` layout) plus a `justfile` at the repo
root. Each CLI is a thin argparse wrapper that calls one function and
writes one set of artifacts. Candidate specs live in
`config.yaml` under `collections.candidates` — same file, same
`config.collections` namespace as `outcomes`. Adding a candidate is a
git diff against `config.yaml`.

**Tech Stack:** Python 3.12, `just`, `pyyaml` (already a dep), `polars`,
existing collection module code.

**Pre-existing code this work builds on:**
- `src/collection/candidates.py` — `CollectionCandidate` dataclass with
  `to_dict` / `from_dict`
- `src/collection/candidate_runner.py` — `train_candidate(candidate,
  outcome, storage, splits_version=None)` returning `CandidateRunResult`
- `src/collection/candidate_comparison.py` — `load_candidate_runs` and
  `compare_runs`
- `src/collection/collection_artifact_storage.py` —
  `save_canonical_splits`, `save_candidate_run`, `list_candidates`, etc.
- `src/collection/outcomes.py` — `OutcomeDefinition`, config loader

---

## File Layout

**Files created:**
- `src/collection/split.py` — persist canonical splits for one outcome
- `src/collection/train.py` — load a candidate from config and run
  `train_candidate`
- `src/collection/compare.py` — load runs, print or write CSV/parquet
- `src/collection/promote.py` — copy a candidate run into the
  production-winner path `{outcome}/v{N}/`
- `justfile` — recipes (`split`, `train`, `train-all`, `compare`,
  `promote`, `sweep`)

**Files modified:**
- `src/collection/candidates.py` — add `load_candidates(config)` (mirrors
  `load_outcomes`)
- `src/collection/collection_artifact_storage.py` — rename
  `_next_version` → `next_version` (public). Update internal callers
  in this file and in `src/collection/collection_pipeline.py`
- `config.yaml` — add `collections.candidates` block with example specs
- `tests/test_collection_pipeline.py` — update mock to track the rename

**Files NOT modified (intentionally):**
- `src/collection/candidate_runner.py` — already correct
- `src/collection/candidate_comparison.py` — already correct
- `src/collection/collection_pipeline.py` — production-winner pipeline is
  out of scope for this plan, except for the mechanical
  `_next_version` → `next_version` callsite update
- `Makefile` — existing `train-collection`, `refresh-collection`,
  `collection-status` targets stay; this plan adds parallel `just`
  recipes for candidate experimentation

---

## Design Decisions

1. **Candidate specs live in `config.yaml`.** A new
   `collections.candidates:` list (parallel to `collections.outcomes`),
   loaded via `load_candidates(config)` returning
   `dict[str, CollectionCandidate]`. Adding/editing candidates is a
   `config.yaml` diff on a branch — git is the experiment-tracking
   surface. Long-term, if `config.yaml` becomes unwieldy, splitting the
   `collections.*` section into its own file is a cheap follow-up.
2. **`load_candidates(config)`** parses the list and re-hydrates each
   entry via the existing `CollectionCandidate.from_dict`. Validation
   happens in `__post_init__` as before. Duplicate names raise.
3. **Splits version selection:** CLIs default to *latest* canonical
   splits. Each accepts `--splits-version N` to pin. The
   compare CLI raises if loaded runs reference different splits
   versions (this enforcement already exists in `compare_runs`).
4. **Promotion semantics:** copy artifacts (model.pkl, threshold.json,
   registration.json) into `{outcome}/v{N}/`. Not symlinks. Adds a
   `promoted_from: {candidate, version}` field to the production
   registration.
5. **Failure behavior in batch recipes:** `train-all` runs every
   candidate, collects failures, and exits non-zero at the end if any
   failed (continue-on-error). `sweep` always runs `compare` if `split`
   succeeded — even if `train-all` had partial failures — but propagates
   a non-zero exit if any candidate failed. Rationale: a comparison
   sweep is most useful when it shows all results, and cron/CI still
   notices failure via the exit code.
6. **Concurrency:** none. Two `train` invocations against the same
   candidate may race and both think they're writing v3. Out of scope;
   documented as a known limitation in the script docstring.
7. **CLI invocation:** `uv run python -m src.collection.<verb>` matches
   the existing `src/pipeline/` and `src/collection/cli.py` patterns.
   Verb-named modules: `split`, `train`, `compare`, `promote`.
8. **Username and environment:** required CLI args on every CLI (no
   implicit "current user"). Flag names match the existing
   `src/collection/cli.py` — `--username` and `--environment` (long
   form, no short aliases). This makes scripts safely scriptable across
   users/envs without ambient state.
9. **Storage API change:** `CollectionArtifactStorage._next_version` is
   renamed to `next_version` (made public). It's already called by
   `CollectionPipeline` and is needed by `promote.py`; making it public
   is cleaner than sprinkling `# noqa: SLF001` across external callers.

---

## What's in `config.yaml`

A new block under the existing `collections:` section, parallel to
`outcomes:`:

```yaml
collections:
  outcomes:
    own:
      task: classification
      label_from: owned
    # ... etc
  candidates:
    - name: logistic_default
      tuning: cv
      cv_folds: 5
      downsample_negatives_ratio: 50
      downsample_protect_min_ratings: 25
      classification_config:
        model_type: logistic
        tuning_metric: log_loss
        threshold_optimization_metric: f2
    - name: lgbm_default
      tuning: cv
      cv_folds: 5
      downsample_negatives_ratio: 50
      downsample_protect_min_ratings: 25
      classification_config:
        model_type: lightgbm
        tuning_metric: log_loss
        threshold_optimization_metric: f2
    - name: lgbm_no_year
      tuning: cv
      cv_folds: 5
      downsample_negatives_ratio: 50
      downsample_protect_min_ratings: 25
      classification_config:
        model_type: lightgbm
        tuning_metric: log_loss
        threshold_optimization_metric: f2
      feature_columns:
        - users_rated
        - average_rating
        - bayesaverage
        - num_weights
        - average_weight
        - min_players
        - max_players
        - min_age
        - playing_time
```

---

## CLI surface

All four take `--username`, `--environment` (default `dev`),
`--outcome`, `--local-root` (default `models/collections`).

| CLI | Extra args | Behavior |
|---|---|---|
| `src.collection.split` | — | Loads processed collection + universe, runs splitter, writes `{outcome}/_splits/v{N}/`. Stdout: `splits_version: N`. |
| `src.collection.train` | `--candidate <name>`, `--splits-version <N>` | Looks up candidate in `config.collections.candidates`, runs `train_candidate`, writes `{outcome}/{candidate}/v{N}/`. Stdout: one JSON object with `version`, `splits_version`, `val_metrics`, `test_metrics`. |
| `src.collection.compare` | `--candidates a,b,c`, `--versions latest\|all`, `--out path.csv\|.parquet` | Loads runs via `load_candidate_runs`, calls `compare_runs`, prints/writes the result. Stderr: `compared N candidates, M rows`. |
| `src.collection.promote` | `--candidate <name>`, `--version <N>\|latest` | Copies the candidate's pipeline + threshold + registration into `{outcome}/v{N}/` (production path). Adds `promoted_from`. Stdout: `promoted_to: production v{N}`. |

---

## Justfile recipes

Defaults: `username := "phenrickson"`, `environment := "dev"`,
`local_root := "models/collections"`. Override per-invocation with
`just username=alice train ...`.

| Recipe | Default args | What it runs |
|---|---|---|
| `default` | — | `just --list` |
| `split` | `outcome="own"` | `src.collection.split` |
| `train` | `outcome="own" candidate="lgbm_default" splits_version=""` | `src.collection.train` (`--splits-version` omitted when blank) |
| `train-all` | `outcome="own"` | Lists candidate names from config (inline `python -c`), runs `src.collection.train` for each, continue-on-error, exits non-zero if any failed. |
| `compare` | `outcome="own" out="" candidates=""` | `src.collection.compare` |
| `promote` | `outcome="own" candidate="lgbm_default" version="latest"` | `src.collection.promote` |
| `sweep` | `outcome="own"` | `split` → `train-all` → `compare`. Always runs `compare` if `split` succeeded. Exits non-zero if any candidate failed. |

Optional flags (`--splits-version`, `--out`, `--candidates`) use
shell-side conditionals (`$([ -n "{{var}}" ] && echo "--flag {{var}}")`)
rather than `just`'s template if/else, which is finicky and easy to
break.

---

## Out of scope

- Concurrent-run locking (documented as known limitation).
- Remote orchestration (Airflow/Prefect/Dagster).
- A GUI for selecting candidates.
- Auto-detection of the best candidate to promote.
- Fetching a fresh collection from BGG inside `just sweep` — that
  belongs upstream of this workflow and uses different scripts/recipes.
- Any modifications to `CollectionPipeline` (production-winner path is
  separate and untouched), except for the mechanical `_next_version`
  rename.
- `compare_runs` enhancements — the existing implementation is correct
  and already enforces the same-splits-version invariant.
- Standalone smoke tests for the new CLIs — the underlying functions
  (`train_candidate`, `compare_runs`, etc.) already have unit tests; the
  CLI wrappers are thin enough that argparse + a real `just sweep` run
  is sufficient verification.
