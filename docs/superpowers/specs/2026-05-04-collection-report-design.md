# Collection Report Design

**Date:** 2026-05-04
**Status:** Draft

## Goal

Render a per-user Quarto HTML report summarizing a collection model's
behavior and predictions. The template is parameterized by username and
outcome, reads experiment artifacts (locally for dev, from GCS for
production deployment), and is rendered by a GitHub Actions workflow that
uploads the resulting HTML to a public bucket.

The reference for content and structure is the existing report at
`https://storage.googleapis.com/bgg_reports/bgg_collections/docs/phenrickson.html`
(R/Quarto source captured in `references/analysis.qmd`). The new template
mirrors its section structure but is implemented in Python to align with
this project.

## Non-goals

- Per-user customized prose. Prose lives in the template and applies to
  any user.
- Committing rendered HTML to git. Reports are build artifacts.
- Replacing the Streamlit Collections page. Streamlit stays for
  exploration; the report is a frozen, shareable view.
- Designing the CI workflow YAML. The workflow is a separate plan; this
  spec defines the entry point it calls (`reports/render.py`).

## File layout

```
reports/
  collection_report.qmd        # template
  styles.css                   # report styles (port from references/)
  render.py                    # CLI render driver
  _output/                     # rendered HTML; gitignored
src/reports/
  __init__.py                  # empty
  collection_data.py           # data loader (local + GCS)
src/collection/viz.py          # add new plot/table helpers; reused by Streamlit
tests/reports/
  test_collection_data.py
  test_viz_collection_report.py
  test_render_smoke.py
```

Rationale: helpers live under `src/` so they're importable by both the
report and Streamlit. The qmd, its styles, and its render driver live
together in `reports/`.

## Data layer — `src/reports/collection_data.py`

Single entry point returning a dataclass that separates user-level data
(outcome-agnostic) from per-outcome artifacts. The template currently
renders one outcome but the structure supports multi-outcome reports
without a loader refactor.

```python
@dataclass
class OutcomeArtifacts:
    outcome: str
    selected_candidate: str          # e.g. "logistic_row_norm"
    selected_version: int

    pipeline: Pipeline               # finalized model
    registration: dict
    threshold: float | None
    feature_importance: pl.DataFrame  # extracted from the fitted pipeline,
                                      # not from the candidate's parquet

    oof_predictions: pl.DataFrame
    val_predictions: pl.DataFrame
    test_predictions: pl.DataFrame

    upcoming_predictions: pl.DataFrame  # deployed-model scores from BQ landing

@dataclass
class CollectionReportData:
    username: str
    collection: pl.DataFrame             # raw BGG snapshot, BQ — outcome-agnostic
    games: pl.DataFrame                  # game metadata, BQ — outcome-agnostic
    outcomes: dict[str, OutcomeArtifacts]  # keyed by outcome name

def load(
    username: str,
    outcomes: str | list[str] = "own",
    source: str = "local",
    candidates: dict[str, str] | None = None,  # per-outcome override
) -> CollectionReportData: ...
```

`outcomes` accepts a single string (load one) or a list (load several).
Phase 1 templates pass `"own"` and access `data.outcomes["own"]` for
per-outcome sections; outcome-agnostic sections (Collection) read from
top-level fields. Multi-outcome views later iterate `data.outcomes`.

The on-disk layout already separates these levels
(`{username}/collection/...` vs `{username}/{outcome}/...`), so this
mirrors what's there.

### Feature importance is extracted from the pipeline

The report shows feature importance for the *finalized* model, the same
way the Streamlit Finalized Model tab does
(`_extract_finalized_importance` in `src/streamlit/pages/7 Collections.py`):

1. Pull `feature_importances_` (tree models) or `coef_` (linear models)
   from `pipeline.named_steps["model"]`.
2. Recover post-preprocessing feature names by transforming a small
   slice of canonical training data
   (`{outcome}/_splits/v{N}/train.parquet`) through
   `pipeline.named_steps["preprocessor"]`. Sklearn's
   `get_feature_names_out` is unreliable on this stack.
3. Return a frame with `feature, value, abs_value`.

The candidate-level `feature_importance.parquet` on disk is *not* used
by the report — that file reflects the candidate's training run, while
the report wants the finalized pipeline's view. The loader extracts
fresh.

This adds one read to `load()`: the canonical splits parquet for the
selected outcome's `_splits/v{N}/`. The splits version comes from the
finalized run's `registration.json` (`splits_version` field).

The extraction logic is shared with Streamlit; lift
`_extract_finalized_importance` to `src/collection/viz.py` and call it
from both surfaces.

### Source switch

- `source="local"` — read artifacts from `models/collections/{username}/{outcome}/...`
- `source="gs://bucket/prefix/"` — same tree, GCS

`fsspec`/`gcsfs` for direct `gs://` reads. Polars reads parquet from
`gs://` natively; pickle/json use `fsspec.open(...)`. No local sync step.

BQ-backed fields (`collection`, `games`, `upcoming_predictions`) always
come from BQ regardless of `source`. `source` only switches the artifact
tree.

### Candidate selection

If `candidate` is None:
1. Prefer `logistic_row_norm` if it has a finalized registration (default).
2. Otherwise pick any candidate marked finalized; if multiple, break the
   tie via the BQ collection registry (most recent active entry wins).
3. If no candidate is finalized, raise.

Selected version = the candidate's most recent finalized version.

## Render helpers — additions to `src/collection/viz.py`

Existing: `plot_feature_importance`, `feature_group`, `tidy_feature_name`.

New (used by both report and Streamlit):

| Helper | Purpose |
|---|---|
| `plot_collection_by_category(collection, games)` | Top mechanics/designers/publishers/etc. in collection (bar chart, faceted by category) |
| `plot_collection_by_year(collection)` | Histogram of `year_published` for owned games |
| `collection_datatable(collection, games)` | Sortable HTML datatable of a user's collection (itables) |
| `plot_separation(predictions)` | Predicted-proba area chart with true-positive overlays (extracted from Streamlit Overview tab) |
| `top_n_by_year_table(predictions, top_n=15)` | Pivot: rank × year, true positives highlighted (extracted from Streamlit Top-N tab) |
| `predictions_datatable(predictions, games, ...)` | Sortable datatable of predictions with BGG image/link columns |
| `metrics_table(registration)` | Wide metrics table (val/oof/test × metric) from registration.json |
| `plot_partial_effects_by_group(feature_importance)` | One plot per feature group (mechanics, designers, ...) for a tabset |
| `extract_finalized_importance(pipeline, train_sample)` | Pull `coef_`/`feature_importances_` from a fitted Pipeline and recover post-preprocessing feature names; lifted from `src/streamlit/pages/7 Collections.py` |

All helpers are pure: take frames in, return Plotly Figure or rendered
HTML/itables object out. No I/O.

## Template — `reports/collection_report.qmd`

Frontmatter:

```yaml
---
title: "Predicting Board Game Collections"
subtitle: "{{< meta username >}}'s Collection"
format:
  html:
    toc: true
    code-fold: true
    embed-resources: true
    theme: cerulean
    css: styles.css
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
```

First chunk loads everything once via `collection_data.load(...)`.
Outcome-agnostic sections (Collection) read from top-level fields.
Per-outcome sections read from `data.outcomes[params.outcome]` (phase 1)
or iterate `data.outcomes` (future multi-outcome views).

Section structure (mirrors `references/analysis.qmd`):

1. **About** — prose only.
2. **Collection** *(uses top-level `data.collection`, `data.games`)*
   - Types of Games — `plot_collection_by_category`, `plot_collection_by_year`
   - Games in Collection — `collection_datatable`
3. **Modeling** *(uses `data.outcomes[outcome]`)*
   - What Predicts a Collection — `plot_feature_importance`
   - Partial Effects — `plot_partial_effects_by_group` in a Quarto tabset
4. **Assessment** *(uses `data.outcomes[outcome]`)*
   - Metrics table — `metrics_table`
   - Separation plot — `plot_separation` (oof + val)
   - Top Games in Training — `predictions_datatable(oof_predictions, ...)`
   - Top Games in Validation — `predictions_datatable(val_predictions, ...)`
   - Top Games by Year — `top_n_by_year_table` over oof+val+test
5. **Predictions** *(uses `data.outcomes[outcome]`)*
   - New and Upcoming Games — `predictions_datatable(upcoming_predictions, ...)` filtered to upcoming + min users_rated
   - Older Games — high-scoring older games from oof+val concat, filtered by min users_rated

## Render driver — `reports/render.py`

```bash
# local dev
uv run python -m reports.render --username phenrickson --outcome own

# CI
uv run python -m reports.render --username phenrickson --outcome own \
    --source gs://bgg_reports/collections-artifacts/

# all users
uv run python -m reports.render --all-users --source gs://...
```

Responsibilities:

1. Discover users — from CLI flag, or by listing the artifact root.
2. Per user: shell out to `quarto render reports/collection_report.qmd \
   -P username:X -P outcome:Y -P source:Z --output-dir reports/_output/`.
3. Output filename: `{username}.html`.
4. Per-user errors are caught and logged; the batch continues. Exit
   non-zero at the end if any user failed.

## Tests

- `tests/reports/test_collection_data.py` — fixture artifact tree under
  `tests/fixtures/collections/{user}/{outcome}/...` with small
  parquet/json files. Asserts `load(source=fixture_path)` populates the
  dataclass. BQ-backed fields are mocked.
- `tests/reports/test_viz_collection_report.py` — feed each new `viz.py`
  helper a small fixture frame; assert the returned figure/table has the
  expected shape and doesn't raise.
- `tests/reports/test_render_smoke.py` — end-to-end: run
  `render.py --username <fixture-user> --source <fixture-path>`, assert
  `reports/_output/{user}.html` exists and is non-trivial. Skipped if
  Quarto is not on PATH.

## Out of scope (future)

- The GitHub Actions workflow that calls `render.py` and uploads to GCS.
  Designed in a follow-up plan.
- Syncing local artifacts up to GCS (`gsutil rsync` or equivalent). The
  report reads from wherever artifacts already are; the sync is a
  separate concern.
- Mixing in deployed-model predictions for non-Predictions sections.
  Currently only the Predictions section reads from BQ landing; if
  later we want production scores reflected elsewhere, revisit.
- Multi-outcome reports (e.g. own + rating + complexity in one
  document). The data layer supports this — `load(outcomes=[...])`
  returns a dict keyed by outcome — but the template only renders one
  outcome. A future spec adds the multi-outcome layout (tabset, side-by-side
  sections, etc.).
