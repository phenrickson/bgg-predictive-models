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

Single entry point returning a dataclass holding everything the template
needs:

```python
@dataclass
class CollectionReportData:
    username: str
    outcome: str
    selected_candidate: str          # e.g. "logistic_row_norm"
    selected_version: int

    collection: pl.DataFrame         # raw BGG snapshot, from BQ
    games: pl.DataFrame              # game metadata (BQ)

    pipeline: Pipeline               # finalized model
    registration: dict
    threshold: float | None
    feature_importance: pl.DataFrame

    oof_predictions: pl.DataFrame
    val_predictions: pl.DataFrame
    test_predictions: pl.DataFrame

    upcoming_predictions: pl.DataFrame  # latest deployed scores from BQ landing

def load(
    username: str,
    outcome: str,
    source: str = "local",
    candidate: str | None = None,
) -> CollectionReportData: ...
```

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
Subsequent section chunks call into the dataclass and `viz.py` helpers.

Section structure (mirrors `references/analysis.qmd`):

1. **About** — prose only.
2. **Collection**
   - Types of Games — `plot_collection_by_category`, `plot_collection_by_year`
   - Games in Collection — `collection_datatable`
3. **Modeling**
   - What Predicts a Collection — `plot_feature_importance`
   - Partial Effects — `plot_partial_effects_by_group` in a Quarto tabset
4. **Assessment**
   - Metrics table — `metrics_table`
   - Separation plot — `plot_separation` (oof + val)
   - Top Games in Training — `predictions_datatable(oof_predictions, ...)`
   - Top Games in Validation — `predictions_datatable(val_predictions, ...)`
   - Top Games by Year — `top_n_by_year_table` over oof+val+test
5. **Predictions**
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
