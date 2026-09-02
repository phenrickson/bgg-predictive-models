# Promote the input-scaled PCA embedding to production

## Goal & success criteria

Make the input-scaled PCA game embedding (branch `feat/embedding-input-scaling`,
validated by the `neighbor_check` spot-check) the live embedding behind
bgg-viewer's "Similar games". Success = the game page for System Gateway / The
White Castle / Pandemic / TI4 shows the new neighbour lists, matching what
`neighbor_check` produced locally.

This replaces the registered model `embeddings-v2026` **in place** — no A/B. Every
game page's similar-games list changes at once when the re-score (step 5)
finishes. Reversible in ~1 pipeline cycle (re-register the prior experiment,
re-run the re-score).

## Decision already made

**Skip the systematic Stage A–E eval** (`2026-08-31-embedding-similarity-eval-design.md`).
Ship on the `neighbor_check` spot-check: System Gateway → Android: Netrunner; the
Resource-Queue cluster (White Castle / Sankoré / Project L / Inferno) dissolved
into real genre neighbourhoods; Pandemic → co-ops; TI4 → 4X space (no longer
Puerto Rico ×5). Structural diagnostic: max component concentration 0.87 → 0.48,
total EVR 0.32 → 0.80, 0 rare-feature-dominated components.

## Process (Phil's, confirmed)

Deploy the new embedding model → register it so `bgg-embeddings-service` points at
it → re-score the universe of games with the scoring service. Three steps; the
warehouse side is automatic.

## The one deploy-mechanics finding

`docker-embeddings-build.yml` triggers on push to `main` **only** for
`services/game_embeddings/**` or `docker/embeddings.Dockerfile` changes. This
branch changes `src/models/embeddings/transformer.py`, not the service dir — so
merging does **not** auto-rebuild the service image. The service unpickles the
model pipeline, which now contains `PlayerCountSanitizer` / `TwoSDScaler` /
`MinCountSelector`, so the image must be rebuilt (with the new `src/`) before the
new model can load. → manual `workflow_dispatch` of the build after merge.

## Not a problem (checked, ruled out)

- **Mixed vector spaces in the incremental warehouse tables.** `bgg_game_embeddings`
  and `game_similarity_search` are incremental (MERGE by `game_id`, no delete), so
  in principle a game left un-re-scored could keep an old-version row while
  `game_neighbors` computes cross-space distances. But the service's
  change-detection query (`services/game_embeddings/main.py:294`) re-embeds every
  game where `le.embedding_version != {model_version}`, over all of
  `games_features` with a non-null year, looping batches until zero remain. A
  version bump re-scores the whole universe → the MERGE overwrites every row → the
  tables converge to one version. No full-refresh step needed; just a post-check.
- **Schema.** Still 64-d; `embedding_8/16/32` still sliced by the service
  (`main.py:374`); only the `algorithm` string flips `svd` → `pca`. No Dataform
  schema drift.
- **`game_neighbors`** is `type: table` (full rebuild every run, ~13s/72MB) and
  its default profile uses 64-d `embedding`. Fine as soon as its input is one
  version.

## Affected files / systems

**bgg-predictive-models** (branch `feat/embedding-input-scaling`):

- `src/features/transformers.py` — `TwoSDScaler`, `MinCountSelector`
- `src/models/embeddings/transformer.py` — `PlayerCountSanitizer`, preprocessor wiring
- `src/models/embeddings/{train,trainer,diagnose_components,neighbor_check}.py`
- `config.yaml` — `embeddings.algorithm: pca`, `algorithms.pca`, `min_feature_count`
- `justfile` — new `embed-register` recipe; `Makefile` `register_embeddings`
  target removed (candidate is now `game-embeddings`, was `svd-embeddings`)

**GCS / Cloud Run** (bgg-predictive-models project):
- Registered model `embeddings-v2026` — overwritten by `just embed-register`
- `bgg-embeddings-service` (Cloud Run) — loads `embeddings-v2026`

**BigQuery** `raw.game_embeddings` — `run-generate-embeddings` appends new-version rows.

**bgg-data-warehouse** (Dataform, all automatic on the `embeddings_complete` dispatch):

- `predictions.bgg_game_embeddings` (incremental) → converges to new version
- `analytics.game_similarity_search` (incremental) → same
- `analytics.game_neighbors` (`type: table`) → full rebuild
- `monitoring.deployed_models` (`type: view`) → no action

**bgg-viewer**: no change — reads `game_neighbors`, picks up the rebuild.

## Steps

### 1. PR: merge `feat/embedding-input-scaling` → `main`

Open the PR, Phil reviews and merges. 8 commits: TwoSDScaler/MinCountSelector,
config swap, player-count features, diagnostic fix, neighbor_check, Makefile→justfile
register move, deployment plan.

**Verify:** CI green. `uv run -m pytest tests/` — the 8 pre-existing
`test_transformers.py` / `test_preprocessor.py` failures (fixture schema drift,
present on `main`) don't grow; everything else passes.

### 2. Rebuild the embeddings service image

`workflow_dispatch` `docker-embeddings-build.yml` against `main` (it won't
auto-fire — see the finding above).

**Verify:** workflow green; `gcloud run services describe bgg-embeddings-service`
shows a new revision serving.

### 3. Sync the validated experiment to GCS + register

- Confirm the validated experiment
  (`models/experiments/embeddings/game-embeddings/vN`) is latest and pushed to
  GCS (`uv run python -m src.utils.sync_experiments`).
- `just embed-register` — packages that experiment as `embeddings-v2026`,
  overwriting in place.

> `just embed-register` runs `services.game_embeddings.register_model` locally
> (writes to GCS). This is the same path used for the last deploy; there is no
> workflow for game-embedding registration. If that should change, it's a
> separate sub-task, not a blocker here.

**Verify:** registration output reports `algorithm: pca`, `embedding_dim: 64`,
the expected experiment hash. Service model-info endpoint (or restart + logs)
confirms it loads the new model.

### 4. Re-score the universe

`workflow_dispatch` `run-generate-embeddings.yml` (input `model_name:
embeddings-v2026`). Batches through every game with a non-current version, then 2D
coordinates, then dispatches `embeddings_complete` to bgg-data-warehouse (which
runs Dataform: `bgg_game_embeddings` → `game_similarity_search` → `game_neighbors`).

**Verify:** job summary — `Games Remaining: 0`. In BQ:
`SELECT embedding_version, algorithm, COUNT(*) FROM raw.game_embeddings GROUP BY 1,2`
— newest version is `pca` with a full-catalog row count.

### 5. Post-checks

- `SELECT COUNT(DISTINCT embedding_version) FROM predictions.bgg_game_embeddings` → **1**
- same for `analytics.game_similarity_search`
- `SELECT algorithm, COUNT(*) FROM analytics.game_similarity_search GROUP BY 1` → only `pca`
- every `game_neighbors.computed_ts` is post-run

If `COUNT(DISTINCT embedding_version)` is 2, a game slipped through the re-score
(gone from `games_features`) — then, and only then, do a one-off targeted
full-refresh of the two incremental tables via the Dataform API
(`reference_dataform_api.md`: `fullyRefreshIncrementalTablesEnabled: true`,
`includedTargets` with the `database` field, `transitiveDependentsIncludedEnabled`).

### 6. Verify on the live product

bgg-viewer game pages: System Gateway, The White Castle, Pandemic, Twilight
Imperium: Fourth Edition, Catan, Gloomhaven. Lists should match `neighbor_check`
for `game-embeddings vN`.

**Verify:** new lists; no API errors; the "Similar games" card renders for an
upcoming 0-rating game (`source_min_users_rated: 0` path).

## Risks / unknowns / rollback

- **In-place replacement, no staged rollout.** Steps 1–3 are inert until step 4;
  step 4 output is inspectable before the live list changes (the Dataform run at
  the tail of step 4 is what flips it).
- **Service can't unpickle** the new model if step 2 is skipped or the Dockerfile
  doesn't `COPY src/` — step 3 verification catches it before any re-score.
- **CI-vs-local drift.** The spot-check ran on the locally-trained `vN`;
  `random_state=42` is pinned so re-runs are near-identical, but re-run
  `neighbor_check` against whatever gets registered as a final gate.
- **Rollback:** `just embed-register svd-embeddings` (re-register the prior
  experiment as `embeddings-v2026`), re-run `run-generate-embeddings`. ~1 pipeline
  cycle. Old experiment + GCS artifacts retained.

## Out of scope

- The `game_neighbors` "quality" profile (family/reimplementation exclusion +
  rating percentile) — separate additive bgg-data-warehouse work.
- The bgg-viewer `/dev/similar` bench.
- Retraining the outcome models (they consume the text embeddings, not this one).
- `config.yaml embeddings.use_embeddings` — stays `false`.
- Moving game-embedding registration into a workflow.
