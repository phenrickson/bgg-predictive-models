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
| Games with embeddings + complexity | **127,940** — the real scoreable ceiling |
| bgg-viewer catalog working set | 35,263 games; 7,424 currently have a prediction |

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

**Sample status is a property of the model, not the game.** From `config.yaml`:

```yaml
years:
  training:
    train_through: 2022
    tune_start: 2023 / tune_through: 2023
    test_start: 2024 / test_through: 2024
finalize_through: 2025   # finalize() refits on train+val+test filtered to <= this
```

The shipped pipeline is refit through 2025, so it has seen essentially every game published
through 2025. A prediction for a pre-2026 game is in-sample. Critically, `finalize_through`
moves when the model is refit — which is why this cannot be derived downstream.

## The gap

1. **Coverage.** 17,067 of ~127,940 scoreable games are scored.
2. **No status field.** `bgg_predictions` columns are `job_id, game_id, name, year_published,
   predicted_*, {target}_model_{name,version,experiment}, score_ts, source_environment,
   first_prediction_ts, is_new_1d, is_new_7d`. Nothing distinguishes a forecast from a fitted
   value.

Consumers today can only infer status by hardcoding "2025", which silently goes wrong at the
next refit. That is the reason this belongs in the pipeline.

## Design

### 1. Scoring population

Score every game that *can* be scored: has description embeddings and a complexity value.
That is ~127,940, not the ~140k total, because the models take `emb_0..emb_N` as features and
a game without an embedding cannot be scored at all.

Two changes:

- Drive the scoring run with `start_year=None` (or an explicit floor) rather than the 2024
  default. The request model's default stays 2024 so existing scheduled runs are unaffected;
  the full run passes the wider window explicitly.
- Raise or remove `max_games`. At 50,000 it silently truncates a 128k run. **Recommend
  batching** rather than one 128k call — the loader pulls embeddings into a pandas frame, and
  a single frame of 128k × 64 embedding columns plus features is a memory risk worth avoiding.

**Decision needed:** batch size and whether the full run is a separate endpoint/mode or the
same one with a wider window.

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

**Decision needed:** confirm `in_sample` should be relative to `finalize_through` (what the
shipped model actually saw) rather than `train_through` (the narrower training split). These
differ for 2023–2025 games, which is a large and interesting population.

### 3. Warehouse plumbing

`bgg_predictions.sqlx` selects columns explicitly, so the two new fields must be added to its
`source_data` select. **This is a schema change on an `incremental` table — Dataform will not
`ALTER` an existing table, so the deploy requires a full refresh**, not an ordinary run.

Storage is not a consideration at this scale:

| | |
|---|---|
| Today | 17,067 rows, 5.86 MB |
| At ~128k rows | **~44 MB** |
| Storage | ~**$0.0009 / month** |
| Full scan | ~**$0.0003** |

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
- **Scoring 128k games in-sample will produce flattering numbers.** That is the point — they
  are labelled — but any aggregate model-quality figure computed over `bgg_predictions`
  without filtering on `sample_status` will be wrong. Worth a note wherever such aggregates
  are computed.
- **`is_new_1d` / `is_new_7d`** are computed from `first_prediction_ts`. A one-off backfill of
  110k games gives them all a first prediction on the same day; anything keying off "new"
  should be checked against that.

## Validation before implementing

1. Confirm the embeddings join is really the ceiling — count games in `games_features` with no
   description embedding, and confirm they are genuinely unscoreable rather than merely
   unembedded-so-far.
2. Time a scoring run over a single batch (say 10k games) to get a per-game cost, then
   extrapolate to 128k before committing to a full run.
3. Confirm `finalize_through` is readable at scoring time from the loaded model/registry
   rather than only from `config.yaml` — if the scorer has to read the config, the flag can
   drift from the model actually loaded, which defeats the purpose.

## Open decisions

1. Batch size / run mode for the full-population score.
2. `in_sample` relative to `finalize_through` (recommended) or `train_through`.
3. Whether `sample_status` should carry the finer `train`/`tune`/`test` values now or later.
