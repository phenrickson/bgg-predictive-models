# Universal Scoring and Sample Status — Implementation Plan

**Date:** 2026-07-30
**Spec:** [2026-07-30-universal-scoring-and-sample-status-design.md](../specs/2026-07-30-universal-scoring-and-sample-status-design.md)
**Status:** Awaiting approval

## Goal & success criteria

Every scoreable game (~128k, gated by description embeddings) carries a current prediction,
and every prediction row states whether it is a forecast or a fitted value.

Done when:

1. `predictions.bgg_predictions` has `sample_status` (STRING) and `training_cutoff_year`
   (INT64) populated for every row.
2. Both are derived from the registration of the model the scorer actually loaded, never from
   `config.yaml`.
3. Distinct `game_id` count in `bgg_predictions` is ~128k, not ~17k.
4. `sample_status` splits at exactly `year_published <= 2024`.

## Decisions (closed)

| | |
|---|---|
| `in_sample` means | `year_published <= test_through` — the year the shipped model was refit through |
| The value today | **2024** (not 2025 — see below) |
| `sample_status` values | binary `in_sample` / `out_of_sample`, STRING-typed for later extension |
| Full-population run | batched, existing endpoint, wider window passed explicitly; request defaults unchanged |

## What reading the code and the artifact changed

1. **The cutoff is already in the registration.** Verified against the deployed blob
   `gs://bgg-predictive-models/prod/models/registered/hurdle/hurdle-v2026/v3/registration.json`
   — `original_experiment.metadata` carries `train_through: 2022`, `tune_through: 2023`,
   `test_through: 2024`. Models are refit through the test year, so **`test_through` is the
   training cutoff**. The scorer already loads this dict (it reads `original_experiment.name`
   from it for every row), so no plumbing is needed — this is a read, not a migration.

2. **The number is 2024, not 2025.** `config.yaml`'s `finalize_through: 2025` is read by no
   game-model code; the only `.py` hits are in `services/collections/`, a separate system.
   `finalize_model.load_data` independently clamps to `current_year - recent_year_threshold`
   = 2024, which agrees with the artifact. Anything that hardcoded 2025 is off by a year.

3. **The production write path is `/simulate_games`, not `/predict_games`.**
   [`run-scoring-service.yml:108`](../../.github/workflows/run-scoring-service.yml#L108) posts
   to `/simulate_games` with `use_change_detection: true`. Both endpoints call
   `upload_predictions`, so **both** must emit the new fields — but only simulate feeds the
   scheduled runs.

4. **Batching already exists and already drains.**
   `load_changed_games_with_embeddings` ([`loader.py:410-444`](../../src/data/loader.py#L410-L444))
   selects games that are unscored OR feature-hash-stale OR version-mismatched, `LIMIT
   max_games`. Scored games leave the set, so looping until `games_simulated == 0` walks the
   whole population. The workflow already loops; it caps at `MAX_BATCHES=10`. Decision 3 is a
   `start_year` and a batch-cap change — no new endpoint, no new code.

## Affected files/systems

**bgg-predictive-models**
- `services/scoring/main.py` — resolve cutoff from registration, compute status, emit in both endpoints
- `src/data/bigquery_uploader.py` — pass the two columns through
- `terraform/bigquery.tf` — two columns on `ml_predictions_landing`
- `.github/workflows/run-scoring-service.yml` — batch cap + wide-window inputs
- `tests/` — new unit tests

**bgg-data-warehouse**
- `definitions/bgg_predictions.sqlx` — two columns in `source_data`
- `definitions/game_first_prediction.sqlx` — only if the `is_new` decision needs it

No change needed: `definitions/sources.js` (declarations carry no schema) and
`definitions/game_profile.sqlx` (whole-row struct `IF(p.game_id IS NULL, NULL, p)` at
[line 99](../../../bgg-data-warehouse/definitions/game_profile.sqlx#L99) picks up new columns
automatically — the viewer's game profile gets the flag for free).

---

## Phase 0 — Confirm before writing code

**0.1 — ~~Confirm all five registrations agree.~~ Done.** All five deployed models report
`train_through: 2022, tune_through: 2023, test_through: 2024` — `hurdle` v3, `complexity` v3,
`rating` v3, `users_rated` v3, `geek_rating` v2. The single-flag assumption holds, and the
cutoff is unambiguously **2024**.

**0.2 — Confirm the embeddings ceiling.** Count `games_features` rows with no row in
`predictions.bgg_description_embeddings`, split by `year_published`, to confirm ~128k is the
real ceiling and the remainder are genuinely unembedded rather than not-yet-embedded.
*Verify:* dry-run first, then the counts.

**0.3 — Time one batch.** Run the existing workflow once with `start_year=2020`,
`max_games=5000`; record wall-clock and Cloud Run memory.
*Verify:* extrapolate to 128k before committing to Phase 2. If a 5k batch approaches the
`--max-time 1800` ceiling, lower the batch size rather than raise the timeout.

---

## Phase 1 — Emit `sample_status` and `training_cutoff_year`

*Branch:* `feat/sample-status-scoring` → PR to `main`.

**1.1 — Resolve the cutoff.** A helper takes the five loaded registrations and returns
`original_experiment.metadata["test_through"]`. Raise if the key is absent — no `config.yaml`
fallback, by design. If the five disagree, take the **minimum** and log a warning (min is the
conservative "seen by every model" boundary).
*Verify:* unit tests for agreement, disagreement, and missing key.

**1.2 — Compute the status.** `in_sample` when `year_published <= cutoff`, `out_of_sample`
above it, `unknown` when `year_published` is NULL. (`load_game_data` does not filter NULL years
the way the change-detection query does, so NULL is reachable and must not silently become
`out_of_sample`.)
*Verify:* unit test on the boundary — 2024 → `in_sample`, 2025 → `out_of_sample`, NULL →
`unknown`.

**1.3 — Emit from both endpoints.** Add both columns to the `dw_predictions` selection in
`predict_games` ([`main.py:728-739`](../../services/scoring/main.py#L728-L739)) and to
`flat_rows` in `simulate_games` ([`main.py:1184-1201`](../../services/scoring/main.py#L1184-L1201)).
*Verify:* a `game_ids` request against a local run returns both fields.

**1.4 — Terraform.** Add `sample_status` (STRING, NULLABLE) and `training_cutoff_year`
(INTEGER, NULLABLE) to `ml_predictions_landing`. The uploader already sets
`ALLOW_FIELD_ADDITION`, so a load could add them implicitly — declare them in terraform anyway
so state does not drift, matching the 2026-02-05 interval-columns precedent.
*Verify:* `terraform plan` shows only the two field additions.

Deploy via the Cloud Build workflow after merge.

---

## Phase 2 — Full-population backfill

**Runs before the Dataform change, deliberately** — so a single full refresh absorbs both the
new columns and the 110k new games instead of two.

**2.1** Raise `MAX_BATCHES` (128k ÷ batch size, plus headroom) and let `start_year` accept a low
floor. Keep the `2025` workflow default and the `2024` request-model default untouched.
*Verify:* the diff touches only the input plumbing and the cap.

**2.2** Run with `start_year=1900` and the batch size from 0.3, monitoring per-batch timing.
*Verify:* `SELECT COUNT(DISTINCT game_id) FROM raw.ml_predictions_landing` → ~128k;
`sample_status` splits at 2024; no NULL `training_cutoff_year`.

---

## Phase 3 — Warehouse plumbing

*Branch in bgg-data-warehouse:* `feat/prediction-sample-status` → PR to `main`.

**3.1** Add both columns to the `source_data` select in `bgg_predictions.sqlx`.
*Verify:* Dataform compile **plus a `CREATE TABLE` dry-run** — a bare `SELECT` dry-run misses
`ref()` and duplicate-field errors.

**3.2** Merge, then immediately full-refresh `bgg_predictions` via the Dataform API (targeted
refresh needs the `database` field in the target). Do not let an ordinary scheduled run hit the
merged model first — Dataform will not `ALTER` the existing incremental table.
*Verify before:* record `COUNT(*)`, `COUNT(DISTINCT game_id)`, `MIN/MAX(first_prediction_ts)`.
*After:* ~128k rows, both columns non-NULL, and `first_prediction_ts` for pre-existing games
**unchanged** — it derives from `MIN(score_ts)` over the append-only landing table
([`game_first_prediction.sqlx:9`](../../../bgg-data-warehouse/definitions/game_first_prediction.sqlx#L9)),
so a refresh cannot reset it.

---

## Phase 4 — The `is_new` stampede

110k games get their first `score_ts` on backfill day, so `is_new_1d` / `is_new_7d` go true for
all of them at once. Existing games are unaffected (their `MIN(score_ts)` is historical).

**Recommendation:** exclude the backfill date explicitly in `bgg_predictions.sqlx` —
`AND DATE(fp.first_prediction_ts) != DATE('<backfill-date>')` on the `is_new_*` expressions,
with a comment naming the backfill. Blunt but auditable and removable, and it keeps "new on
BGG" meaningful in the week after. The alternative is to accept one noisy week.

**This is the one item still open.**

---

## Risks / unknowns / rollback

| Risk | Handling |
|---|---|
| **Full refresh of `bgg_predictions`** — the only irreversible-ish step | Rebuildable from the append-only landing table; take before/after counts |
| **In-sample predictions flatter aggregate quality metrics** | Any aggregate over `bgg_predictions` must filter `sample_status`; note it wherever such figures are computed |
| **Memory on a large batch** — the loader materialises embeddings into pandas | Batch size set from the 0.3 measurement, not guessed |
| **Cloud Run 1800s per-batch ceiling** | Lower batch size rather than raise the timeout |
| **Unknown:** whether all five models share `test_through: 2024` | Phase 0.1, before any code |

Cost is not a factor: ~44 MB at 128k rows, ~$0.0009/month storage.

## Out of scope

- Feature-hash-triggered incremental rescoring (deferred by the spec).
- Retraining, split changes, model selection.
- Any bgg-viewer UI, including the Predictions view.
- Prediction history — `bgg_predictions` stays one latest row per game.
- The hurdle threshold discrepancy noted while reading the registration — separate issue.
