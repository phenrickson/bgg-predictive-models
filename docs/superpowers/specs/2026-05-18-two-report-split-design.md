# Two-Report Split + Room for General Reports Design

**Date:** 2026-05-18
**Status:** Draft

## Goal

Split the single per-user collection report into two independent reports
with separate audiences, cadences, and triggers:

- A **predictions report** — shareable, low-jargon, re-rendered **daily**.
- A **model report** — technical findings/evaluation, re-rendered **only
  when a model is promoted**.

At the same time, establish a site layout and index convention that
treats the published site as a typed *BGG reports platform*, so future
**site-wide (non-per-user) reports** slot in additively without
restructuring. No global-report machinery is built now — only the
namespace and index structure that makes it additive later.

## Background

Today `reports/collection_report.qmd` (704 lines) renders one HTML per
user containing every section. The CI pipeline on the
`feature/dockerized-report-builds` branch is a 3-job matrix
(`discover` → `render-user` → `deploy`) that pulls a prebuilt `reports`
Docker image, renders one user per matrix runner, seeds the deploy from
the previous gs:// bundle so partial renders don't wipe absent users,
and publishes to GitHub Pages.

The current single report's sections:

- **About** — prose only
- **Collection** — Types of Games, the user's collection table
- **Modeling** — Feature Importance, Partial Effects (tabset)
- **Assessment** — Top Games in Training, Top Games in Testing, by Year
- **Predictions** — New and Upcoming Games, Older Games
- **Model Details** — appendix

This design splits that document along the audience/cadence boundary and
generalizes the publishing layout.

## Non-goals

- The future training/finalization job itself. This design only defines
  the `gh workflow run` seam that job (and the local `promote` recipe)
  will call.
- Any actual site-wide/global report — its template, workflow, and
  content are a separate spec written when requirements exist. This
  design only reserves the `/reports/` namespace and the index's
  "Reports" section so that future work is additive.
- The `workflow_run` auto-rebuild-on-image-change TODO from commit
  `99f9770`. Orthogonal; can be added to either/both workflows later.
- Multi-outcome reports. The data layer (`collection_data.load`)
  already supports `outcomes=[...]`; templates stay single-outcome.
- Migrating or retiring consumers of `collection_report.qmd` beyond the
  two CI workflows.

## Section split

| Report | Sections |
|---|---|
| **Predictions** | About → Collection (Types of Games, collection table) → Predictions (New & Upcoming, Older high-scoring) |
| **Model** | About → Collection → Modeling (Feature Importance, Partial Effects) → Assessment (Top in Training/Testing, by Year) → Model Details appendix |

Rationale: the boundary is audience *and* cadence *and* trigger — three
independent axes of divergence. The Collection section appears in both
because it's cheap context that orients either audience; it is rendered
from shared code, not duplicated.

## Templates — `reports/`

`collection_report.qmd` splits into:

- `predictions_report.qmd` — thin composition: About prose + Collection
  context + Predictions sections.
- `model_report.qmd` — thin composition: About + Collection context +
  Modeling + Assessment + Model Details appendix.

Shared logic is factored out so neither template duplicates section
logic. This extends the existing `src/reports/` seam
(`tables.py`, `format.py`, `fixtures.py`) and `src/collection/viz.py`:

- The first-chunk bootstrap (project-root import shim, `params`
  handling, offline/fixture stubs) and the `collection_data.load(...)`
  call are shared — either as a shared `_setup.qmd` include or a
  `src/reports/` helper invoked identically from both templates.
- The Collection-context section (Types of Games plot, collection
  datatable) is a shared include / helper used by both.
- Each template is a composition of shared pieces plus its own
  prose. No section's rendering logic lives in two places.

### Render driver — `reports/render.py`

`render.py` gains a `--report {predictions,model}` argument:

- Selects which `.qmd` to render.
- Determines the output path namespace (see Site layout): predictions →
  `{user}.html` at the output root; model → `model/{user}.html`.
- All existing behavior (per-user discovery, offline/fixture modes,
  pre-flight `select_candidate` check, gs:// source, continue-on-error
  batch, `QUARTO_PYTHON` venv wiring) is unchanged and applies to both
  report types.

`--report` is required (no default) so the caller is always explicit
about which document it is producing.

## Site layout

The published site is organized as **typed report namespaces**, not a
flat user list:

- **Per-user collection reports:**
  - `/{user}.html` → predictions report. Unchanged top-level URL — this
    is the friendly "main" report and existing shared links stay valid.
  - `/model/{user}.html` → model report.
- **Reserved for future site-wide reports:** `/reports/<name>.html`. No
  code, no workflow, no template now. The namespace is documented and
  the index knows how to render a "Reports" section; it is simply empty
  until a global report exists.

Predictions stays at `/{user}.html` (rather than moving to, e.g.,
`/collection/predictions/{user}.html`) specifically to preserve existing
shared URLs. Per-user is the established namespace and is kept as-is,
with `/model/` and `/reports/` added beside it.

The gs:// bundle-mirror seeding already operates per-path, so the three
namespaces never collide: a daily predictions render of `/{user}.html`
never touches `/model/...`, and an on-promote model render never touches
`/{user}.html`.

## Index — `reports/build_index.py`

`build_index.py` emits the page in **typed sections** rather than one
flat list:

- **Collections** section: one row per user, two links —
  "Predictions" → `/{user}.html`, "Model" → `/model/{user}.html`.
- **Reports** section: site-wide reports, rendered from whatever exists
  under `/reports/`. Empty/omitted today; populated additively when a
  global report appears.

Index generation reads what is actually present per namespace (on disk /
in the gs:// bundle), so adding a global report later requires zero
index-code change — only data. A user with a predictions report but no
model report yet (or vice versa) renders with only the link that
exists.

## Workflows — `.github/workflows/`

### `build-collection-reports.yml` → predictions pipeline

The existing 3-job matrix workflow on this branch becomes the
predictions pipeline:

- Trigger: **daily cron** (replacing the Sunday `0 9 * * 0` cron) +
  `workflow_dispatch` (existing `users` input: blank = discover all).
- `discover` → `render-user` matrix → `deploy`, structure unchanged.
- `render-user` passes `--report predictions`; output lands at
  `/{user}.html`.
- `deploy` seeds from the gs:// bundle, overlays this run's
  predictions HTMLs, rebuilds the typed index, mirrors back, deploys
  to Pages — all existing logic, now scoped to the predictions
  namespace.

Only changes vs. the current branch state: cron frequency (Sunday →
daily) and the `--report predictions` argument.

### New `build-model-reports.yml` → model pipeline

A second workflow, structurally symmetric to the predictions one:

- Trigger: `workflow_dispatch` with inputs:
  - `users` — comma-separated; blank = discover all (same contract and
    `discover` job logic as the predictions workflow).
  - `outcome` — which outcome to render the model report for.
- Same `discover` → `render-user` → `deploy` shape. The `discover` job
  is the same logic as predictions (symmetric, reusable).
- `render-user` passes `--report model`; output lands at
  `/model/{user}.html`.
- `deploy` seeds from the gs:// bundle and overlays only the
  `/model/` namespace, so a single-user model render leaves every
  other user's model report and all predictions reports intact.

### Trigger model

**The promoter triggers the model-report render.** Whoever/whatever
promotes a finalized model is responsible for kicking
`build-model-reports.yml` afterward:

- **Today (local):** the `promote` justfile recipe / CLI ends by
  calling `gh workflow run build-model-reports.yml -f users=<user>
  -f outcome=<outcome>` — a surgical single-user render.
- **Future (CI job):** the planned training/finalization job's final
  step is the same `gh workflow run` call. The job inherits the seam;
  no new detection mechanism is designed.
- **Manual:** a blank-`users` dispatch re-renders all users' model
  reports (for a template change or a batch retrain).

This is the same event-driven pattern as the existing Docker→render
TODO. No GCS polling, no "what did I last render" state, no detection
workflow. One seam, identical local and in-CI.

### Shared Docker image

Both workflows pull the same prebuilt `reports` image from
`docker-reports-build.yml` (unchanged). Both `predictions_report.qmd`
and `model_report.qmd` ship in that image; the image rebuild on
`reports/**` change is orthogonal to which report a workflow renders.

## Error handling

- Per-user render failures remain isolated by the existing matrix
  `fail-fast: false`; one user's broken artifacts fail only that
  matrix cell. The workflow exits non-zero if any cell failed.
- The pre-flight `select_candidate` check in `render.py` still gates
  each user before Quarto spins up; a user with no finalized artifacts
  for the requested outcome is logged and skipped, the batch
  continues.
- `deploy` seeding from the gs:// bundle is the partial-render safety
  net: a failed or skipped user keeps its last-published HTML in the
  affected namespace rather than disappearing from the live site.

## Testing

Extends the existing `tests/reports/` suite:

- `test_render_smoke.py` — add cases rendering each report type
  (`--report predictions`, `--report model`) against the fixture user;
  assert the expected output path (`{user}.html` vs.
  `model/{user}.html`) exists and is non-trivial. Skipped if Quarto is
  not on PATH (existing convention).
- `test_collection_data.py` — unchanged; the shared loader is not
  modified by this split.
- Shared-helper extraction (Collection-context section, setup chunk):
  if lifted into `src/reports/`, add focused unit tests mirroring the
  existing `test_viz_collection_report.py` pattern (feed a small
  fixture frame, assert shape, no raise).
- `build_index.py` — a unit test that, given a temp directory seeded
  with `{user}.html` and `model/{user}.html` files, asserts the
  generated index contains a Collections section with both links and
  that a missing model file degrades to a single link. The empty
  Reports section is asserted to be absent/empty when `/reports/` has
  no files.

## Open follow-ups (separate specs)

- The training/finalization job that will call the model-report
  dispatch seam.
- The first real site-wide/global report (template + single-render
  workflow targeting `/reports/<name>.html`).
- The `workflow_run` auto-rebuild-on-image-change trigger.
