# Universal Scoring and Sample Status Design

**Date:** 2026-07-30
**Status:** Draft — decisions marked below need Phil's call before an implementation plan

## Goal

Every game in the data warehouse carries a current prediction from the active deployed
models, and every prediction says whether it is a **forecast** or a **fitted value** — a
score for a game the model was trained on.

Two consumers depend on this:

- **bgg-viewer game profile** — shows the prediction for any game, labelled with its status.
  Phil wants in-sample predictions visible precisely *because* they show model behaviour;
  they must be labelled, not hidden.
- **bgg-viewer catalog artifact** — already carries the five predicted columns for every game
  in its working set. It needs the status flag to travel with them.

## Non-goals

- **Feature-hash-triggered rescoring.** Agreed to defer: backfill first, incremental
  rescoring as its own piece. This spec must not make that harder, but does not build it.
- Retraining, changing the splits, or touching model selection.
- Any bgg-viewer UI. The Predictions view is a separate design.
- Backfilling prediction *history*. `bgg_predictions` stays one latest row per game;
  `ml_predictions_landing` keeps the append-only record.

## Current state

Measured 2026-07-30, not assumed:

| Fact | Value |
|---|---|
| `raw.ml_predictions_landing` | 17,067 distinct games, `year_published` 2024–2028, 252 jobs |
| `predictions.bgg_predictions` | 17,067 rows, 5,864,858 bytes (**344 bytes/row**) |
| Games with an embedding | 127,949 (the other 12,018 have no `year_published` and are deliberately not scored) |
| bgg-viewer catalog working set | 35,263 games; 7,424 currently have a prediction |

**The scoring target is not every embedded game.** Two sets get scored:

- **all upcoming games**, no ratings filter — *already scored*; these are the 17,067
- **all rated games**, ~30k

Everything else — old games with too few ratings to matter — is deliberately left unscored.
So the coverage gap is **~30k rated games, not ~110k**. At a measured ~77ms/game this is
roughly 40 minutes in the existing batch loop, which makes run mode a non-question.

**The year gate is not in the warehouse.** [`definitions/bgg_predictions.sqlx`](../../../bgg-data-warehouse/definitions/bgg_predictions.sqlx)
has no year filter — it takes the latest row per `game_id` from the landing table. The
17,067-game limit comes entirely from the scoring service:

- `start_year: Optional[int] = 2024` — request-model defaults, [`services/scoring/main.py:91`](../../services/scoring/main.py#L91) and [`:124`](../../services/scoring/main.py#L124)
- the filter is only appended when set, [`services/scoring/main.py:292`](../../services/scoring/main.py#L292):
  ```python
  if start_year is not None:
      where_parts.append(f"f.year_published >= {start_year}")
  ```
- `game_ids` takes precedence over year filtering entirely ([`main.py:287`](../../services/scoring/main.py#L287))
- `load_games_for_main_scoring(..., max_games: int = 50000)` ([`main.py:305`](../../services/scoring/main.py#L305))

So scoring the full population is a **payload and cap change, not a new capability**.

**Sample status is a property of the model, not the game.** The shipped models are refit
through their **test year**, so `test_through` is the training cutoff.

Verified against the deployed artifact — `gs://bgg-predictive-models/prod/models/registered/
hurdle/hurdle-v2026/v3/registration.json`, under `original_experiment.metadata`:

```json
"train_through": 2022,
"tune_through":  2023,
"test_through":  2024,
```

**The cutoff is 2024, not 2025.** `config.yaml`'s `finalize_through: 2025` is read by no
game-model code — grep it: the only `.py` hits are in `services/collections/`, a separate
system reading its own value. Independently, `finalize_model.load_data` clamps the end year to
`current_year - recent_year_threshold` = `2026 - 2` = 2024, which agrees.

The scoring service already loads this dict — it reads `original_experiment.name` from it for
every prediction row. So the cutoff is available at scoring time with no new plumbing, and it
moves correctly when the model is refit, which is why it must come from here and not from
config.

## The gap

1. **Coverage.** Upcoming games are scored; the ~30k rated games are not.
2. **No status field.** `bgg_predictions` columns are `job_id, game_id, name, year_published,
   predicted_*, {target}_model_{name,version,experiment}, score_ts, source_environment,
   first_prediction_ts, is_new_1d, is_new_7d`. Nothing distinguishes a forecast from a fitted
   value.

Consumers today can only infer status by hardcoding "2025", which silently goes wrong at the
next refit. That is the reason this belongs in the pipeline.

## Design

### 1. Scoring population

Upcoming + rated. Upcoming is already covered, so the work is the **~30k rated games**
(`users_rated >= 25`, matching the `hurdle` definition in `games_features`).

One change: drive the run with a wide `start_year` so the change-detection loop reaches rated
games from earlier years. The request model's default stays 2024 so scheduled runs are
unaffected; the backfill passes the wider window explicitly.

`max_games` at 50,000 does not truncate a ~30k run, and the existing while-loop in
`run-scoring-service.yml` already drains the change-detection set. **No new endpoint, no
parallelism, no batch-size tuning** — measured at ~77ms/game, ~30k is roughly 40 minutes.

Note that the loader's change-detection query has no ratings filter — it selects on year and
staleness only. Restricting the backfill to rated games needs either a ratings predicate added
there or the batch driven by `game_ids`.

### 2. Sample status on the prediction row

Emit two new fields per prediction, from the scorer, which knows which model it loaded:

| Field | Type | Meaning |
|---|---|---|
| `sample_status` | STRING | `in_sample` / `out_of_sample` |
| `training_cutoff_year` | INT64 | the `finalize_through` the scoring pipeline was fitted with |

`sample_status` is what consumers read. `training_cutoff_year` is what makes it auditable and
keeps it honest across refits — if the flag and the cutoff ever disagree, that is a visible
bug rather than a silent one.

**Why a string and not a boolean:** the config has real `train` / `tune` / `test` /
post-finalize distinctions, and a boolean forecloses ever exposing them. A string costs
nothing and can gain values later without a schema change.

**Why one flag and not five:** each target has its own model and version, so in principle a
game could be in-sample for `hurdle` and not for `complexity`. Today they share one
`finalize_through`, so one flag is correct. If the targets ever diverge, `training_cutoff_year`
is the field that would need to go per-target — flagged here so that is a deliberate choice
later, not a surprise.

**Decided:** `in_sample` is relative to the refit cutoff — what the shipped model actually saw
— which is `test_through`, **2024**. Not `train_through` (2022), and not the 2025 this spec
originally asserted. 2023 and 2024 games are therefore `in_sample`; 2025 games are not.

### 3. Warehouse plumbing

`bgg_predictions.sqlx` selects columns explicitly, so the two new fields must be added to its
`source_data` select. **This is a schema change on an `incremental` table — Dataform will not
`ALTER` an existing table, so the deploy requires a full refresh**, not an ordinary run.

Storage is not a consideration at this scale:

| | |
|---|---|
| Today | 17,067 rows, 5.86 MB |
| At ~47k rows (upcoming + rated) | **~16 MB** |
| Storage | ~**$0.0003 / month** |
| Full scan | ~**$0.0001** |

The per-game read already pays BigQuery's 10 MB minimum, so the point lookup barely moves.

### 4. Downstream, for reference

bgg-viewer's catalog artifact already carries the five predicted columns (+128 KB gzipped,
+2.9%). Once `sample_status` exists it joins them, and the game profile can label a fitted
value as such. No viewer change is in scope here.

## Risks and one-way doors

- **The full refresh of `bgg_predictions`** is the only irreversible-ish step. It rebuilds
  from `ml_predictions_landing`, which is append-only and retains everything, so the table can
  be reconstructed — but `first_prediction_ts` derives from `game_first_prediction` and should
  be checked before and after so "new game" flags do not all reset.
- **Scoring ~30k rated games in-sample will produce flattering numbers.** That is the point —
  they are labelled — but any aggregate model-quality figure computed over `bgg_predictions`
  without filtering on `sample_status` will be wrong. Worth a note wherever such aggregates
  are computed.
- **`is_new_1d` / `is_new_7d`** are computed from `first_prediction_ts`. A one-off backfill of
  ~30k games gives them all a first prediction on the same day; anything keying off "new"
  should be checked against that. Known consumers are the prediction and collection reports —
  the predictions report already scopes to `upcoming`, so the exposure may be nil. To be
  confirmed separately.

## Validation before implementing

1. ~~Confirm the embeddings join is really the ceiling.~~ **Resolved, and moot.** 127,949
   games have an embedding; the 12,018 without one all lack `year_published` and are
   deliberately not scored. The scoring target is upcoming + rated, not every embedded game.
2. ~~Time a scoring run.~~ **Measured:** ~77ms/game, ~4s fixed cost per request (250 games in
   23s, 2,000 in 157s, 6,000 in 673s — per-game cost rises slightly with batch size). ~30k is
   roughly 40 minutes in the existing loop.
3. ~~Confirm the cutoff is readable at scoring time from the loaded model.~~ **Resolved.**
   `original_experiment.metadata.test_through` is in the registration the scorer already
   loads. All five deployed models agree — `train_through: 2022, tune_through: 2023,
   test_through: 2024` (`hurdle` v3, `complexity` v3, `rating` v3, `users_rated` v3,
   `geek_rating` v2), so the single flag argued for above is correct.

## Open decisions

1. ~~Batch size / run mode.~~ Moot. The gap is ~30k rated games, ~40 minutes in the existing
   loop at the existing batch size.
2. ~~`in_sample` relative to which cutoff.~~ The refit cutoff, `test_through` = 2024.
3. ~~Finer `train`/`tune`/`test` values.~~ Binary. Once the cutoff is the refit year there is
   one boundary, not three.
4. **Open:** how to handle the `is_new_1d` / `is_new_7d` stampede from the backfill.
