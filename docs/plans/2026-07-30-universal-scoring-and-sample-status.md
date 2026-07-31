# Universal Scoring and Sample Status — Implementation Plan

**Date:** 2026-07-30
**Spec:** [2026-07-30-universal-scoring-and-sample-status-design.md](../specs/2026-07-30-universal-scoring-and-sample-status-design.md)
**Status:** Phases 1 and 2 shipped. **Next step: merge bgg-data-warehouse#96, then full-refresh
`bgg_predictions` immediately.** Until that happens no consumer sees these fields.

## Progress

| | | |
|---|---|---|
| Phase 1 — scorer emits both fields | **done** | [#62](https://github.com/phenrickson/bgg-predictive-models/pull/62) merged, deployed |
| Scoring image build repair | **done** | [#63](https://github.com/phenrickson/bgg-predictive-models/pull/63) — image had not built since 2026-04-29 |
| Deploy verification | **done** | revision `bgg-model-scoring-00020-8dn`; predictions bit-identical to the April image (0 differences across 100 games × 5 columns) |
| Workflow inputs for the backfill | **done** | [#64](https://github.com/phenrickson/bgg-predictive-models/pull/64) merged |
| Phase 2 — backfill rated games | **done** | 31,092 games scored, `sample_status=in_sample`, `training_cutoff_year=2024` |
| Phase 3 — `bgg_predictions.sqlx` | **open** | [bgg-data-warehouse#96](https://github.com/phenrickson/bgg-data-warehouse/pull/96), dry-run passes, awaiting merge |
| Phase 3 — full refresh | **not started** | must follow the merge immediately |

**Landing table now** — latest row per game, 46,675 games:

| | games | years |
|---|---|---|
| `in_sample`, cutoff 2024 | 31,092 | 1900–2024 |
| NULL (scored before the change) | 15,583 | 2024–2028 |

The NULLs self-heal via change detection: 3,446 are already stale and rescore on the next
run, 5,905 had a feature-hash change within 7 days, 4,335 within 30. No action needed.

**Measured, superseding earlier estimates:** ~183ms/game in production (2,500-game batches at
~6.8 min each), not the ~77ms measured against smaller ad-hoc batches. The backfill took 77
minutes across 12 batches.

## Goal & success criteria

The ~30k rated games get scored, and every prediction row says whether it is a forecast or a
fitted value.

Done when:

1. `predictions.bgg_predictions` has `sample_status` (STRING) and `training_cutoff_year`
   (INT64) populated for every row.
2. Both derive from the registration of the model the scorer actually loaded, never from
   `config.yaml`.
3. Rated games have predictions; `bgg_predictions` goes from ~17k to ~47k rows.
4. `sample_status` splits at exactly `year_published <= 2024`.

## Settled

| | |
|---|---|
| Scoring target | upcoming (already done) + rated (`users_rated >= 25`, ~30k) |
| `in_sample` means | `year_published <= test_through` = **2024** |
| `sample_status` | binary `in_sample` / `out_of_sample`, STRING-typed |
| Run mode | existing loop, existing batch size — ~40 min, no parallelism needed |

## Established by measurement

- **The cutoff is in the registration already.** All five deployed models report
  `train_through: 2022, tune_through: 2023, test_through: 2024` under
  `original_experiment.metadata` — a dict the scorer already loads. Nothing to plumb.
  `config.yaml`'s `finalize_through: 2025` is read by no game-model code.
- **`/simulate_games` is the prod path**, not `/predict_games`. They are not two speeds of the
  same thing: simulate writes the posterior **median** plus 8 interval columns; predict writes
  `pipeline.predict()`, rounds `users_rated` to the nearest 50, and clips `geek_rating` to
  [1,10]. Every existing row came from simulate, so the backfill must use simulate too.
  `predict_games` is referenced by no workflow.
- **Cost: ~77ms/game, ~4s fixed per request.** 250 games in 23s, 2,000 in 157s, 6,000 in 673s.
  Per-game cost rises slightly with batch size, so there is no gain from larger batches.
- **The 12,018 games with no embedding** all lack `year_published` and are deliberately
  excluded. Not a backlog.

## Affected files

**bgg-predictive-models**
- `services/scoring/main.py` — resolve cutoff from registration, compute status, emit it
- `src/data/bigquery_uploader.py` — pass the two columns through
- `src/data/loader.py` — ratings predicate for the backfill selection
- `terraform/bigquery.tf` — two columns on `ml_predictions_landing`
- `tests/`

**bgg-data-warehouse**
- `definitions/bgg_predictions.sqlx` — two columns in `source_data`

No change: `definitions/sources.js` (declarations carry no schema); `game_profile.sqlx` picks up
new columns automatically via its whole-row struct.

---

## Phase 1 — Emit `sample_status` and `training_cutoff_year`

*Branch:* `feat/sample-status-scoring` → PR to `main`.

**1.1** Helper returning `original_experiment.metadata["test_through"]` from the loaded
registrations. Raise if absent — no `config.yaml` fallback. If the five disagree, take the
minimum and log a warning.
*Verify:* unit tests for agreement, disagreement, missing key.

**1.2** `in_sample` when `year_published <= cutoff`, else `out_of_sample`. Strictly binary —
null-year games are never scored.
*Verify:* unit test on the boundary — 2024 → `in_sample`, 2025 → `out_of_sample`.

**1.3** Emit from `simulate_games` (`flat_rows`, [`main.py:1184`](../../services/scoring/main.py#L1184))
and from `predict_games` (`dw_predictions`, [`main.py:728`](../../services/scoring/main.py#L728))
so the two paths cannot diverge, even though only simulate runs in prod.
*Verify:* a `game_ids` request against a local run returns both fields.

**1.4** Terraform: `sample_status` (STRING, NULLABLE) and `training_cutoff_year` (INTEGER,
NULLABLE) on `ml_predictions_landing`.
*Verify:* `terraform plan` shows only the two field additions.

Deploy via the Cloud Build workflow after merge.

---

## Phase 2 — Score the rated games

Runs **before** the Dataform change, so one full refresh absorbs both the new columns and the
new rows.

**2.1** The change-detection query
([`loader.py:410-444`](../../src/data/loader.py#L410-L444)) filters on year and staleness only
— it has no ratings predicate. Add one so the backfill targets rated games rather than
everything with an embedding.
*Verify:* the selection query returns ~30k before any scoring runs.

**2.2** Run the existing workflow with a wide `start_year`. ~40 minutes at current batch size.
*Verify:* `COUNT(DISTINCT game_id)` in the landing table → ~47k; `sample_status` splits at
2024; no NULL `training_cutoff_year`.

---

## Phase 3 — Warehouse plumbing

*Branch in bgg-data-warehouse:* `feat/prediction-sample-status` → PR to `main`.

**3.1** Add both columns to the `source_data` select in `bgg_predictions.sqlx`.
*Verify:* Dataform compile **plus a `CREATE TABLE` dry-run** — a bare `SELECT` dry-run misses
`ref()` and duplicate-field errors.

**3.2** Merge, then immediately full-refresh `bgg_predictions` via the Dataform API. Do not let
an ordinary scheduled run hit the merged model first — Dataform will not `ALTER` the existing
incremental table.
*Verify before:* `COUNT(*)`, `COUNT(DISTINCT game_id)`, `MIN/MAX(first_prediction_ts)`.
*After:* ~47k rows, both columns non-NULL, `first_prediction_ts` unchanged for pre-existing
games — it derives from `MIN(score_ts)` over the append-only landing table, so a refresh
cannot reset it.

---

## Open

**The `is_new` stampede.** ~30k games get the same `first_prediction_ts`. Consumers found so
far are in this repo's reports — [`predictions_report.qmd:82`](../../reports/predictions_report.qmd#L82)
(already scoped to `upcoming`, so likely unaffected) and
[`collection_data.py:431`](../../src/reports/collection_data.py#L431) (computes its own flag
from `first_score_ts`). bgg-viewer and bgg-dash-viewer not yet checked. Verify the consumers
before deciding whether any mitigation is needed.

## Risks

| Risk | Handling |
|---|---|
| **Full refresh of `bgg_predictions`** | Rebuildable from the append-only landing table; take before/after counts |
| **In-sample predictions flatter aggregate metrics** | Any aggregate must filter `sample_status` |

## Out of scope

- Feature-hash-triggered incremental rescoring.
- Retraining, split changes, model selection.
- bgg-viewer UI.
- Prediction history — `bgg_predictions` stays one latest row per game.
