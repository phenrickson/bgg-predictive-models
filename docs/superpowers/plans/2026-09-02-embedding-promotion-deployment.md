# Promote the input-scaled PCA embedding to production

## Goal & success criteria

Make the input-scaled PCA game embedding (branch `feat/embedding-input-scaling`,
validated by the `neighbor_check` spot-check) the live embedding behind
bgg-viewer's "Similar games". Success = `game_neighbors` in bgg-data-warehouse is
rebuilt entirely from new-model vectors (no old SVD rows mixed in), and a
spot-check of System Gateway / The White Castle / Pandemic / TI4 on the live game
page shows the new neighbour lists.

This replaces `embeddings-v2026` **in place** — there is no A/B. Every game page's
similar-games list changes at once when step 6 lands. It is reversible in ~1
pipeline run (re-register the prior experiment, re-generate).

## Decision already made

- **Skip the systematic Stage A–E eval** (`2026-08-31-embedding-similarity-eval-design.md`).
  Ship on the `neighbor_check` spot-check: System Gateway → Android: Netrunner;
  the Resource-Queue cluster (White Castle / Sankoré / Project L / Inferno)
  dissolved into real genre neighbourhoods; Pandemic → co-ops; TI4 → 4X space
  (no longer Puerto Rico ×5). Structural diagnostic: max component concentration
  0.87 → 0.48, total EVR 0.32 → 0.80, 0 rare-feature-dominated components.

## Affected files / systems

**bgg-predictive-models** (branch `feat/embedding-input-scaling`, already has the
model + Makefile changes committed):
- `src/features/transformers.py` — `TwoSDScaler`, `MinCountSelector`
- `src/models/embeddings/transformer.py` — `PlayerCountSanitizer`, preprocessor wiring
- `src/models/embeddings/{train,trainer,diagnose_components,neighbor_check}.py`
- `config.yaml` — `embeddings.algorithm: pca`, `algorithms.pca`, `min_feature_count`
- `Makefile` — `EMBEDDINGS_CANDIDATE ?= game-embeddings` (was `svd-embeddings`)
- **No change** to `services/game_embeddings/**` → `docker-embeddings-build.yml`
  will NOT auto-fire on merge; must be dispatched manually (step 2).

**GCS / Cloud Run** (bgg-predictive-models project):
- Registered model `embeddings-v2026` — overwritten by `make register_embeddings`
- `bgg-embeddings-service` (Cloud Run) — loads `embeddings-v2026`; image must
  carry the new `src/models/embeddings/transformer.py` to unpickle the pipeline

**BigQuery** (bgg-predictive-models `raw`):
- `raw.game_embeddings` — `run-generate-embeddings` appends new-version rows

**bgg-data-warehouse** (Dataform):
- `predictions.bgg_game_embeddings` — **incremental**, filters `embedding_version =
  MAX(...)`, `uniqueKey [game_id]`. MERGE upserts new rows but leaves stale
  old-version rows for any game not re-embedded. **Needs full-refresh.**
- `analytics.game_similarity_search` — **incremental**, same stale-row hazard,
  reads `embedding_8/16/32`. **Needs full-refresh.**
- `analytics.game_neighbors` — `type: table`, full rebuild every run (~13s/72MB),
  `dims: 64` → uses `embedding` column only. Fine **once its input is clean**.
- `monitoring.deployed_models` — `type: view`, no action.
- No schema change: still 64-d, `embedding_8/16/32` still produced by the service
  (`services/game_embeddings/main.py`), `algorithm` string flips `svd` → `pca`.

**bgg-viewer**: no change — the game page reads `game_neighbors` and picks up the
rebuild automatically.

## Why the full-refresh is required (the key finding)

`bgg_game_embeddings` and `game_similarity_search` are incremental. The standard
`dataform.yml` invocation (`embeddings_complete` dispatch → `POST
workflowInvocations` with no `invocationConfig`) runs them **incrementally**. On a
model-version bump:

- New PCA rows land with a higher `embedding_version` and newer `created_ts`.
- The incremental MERGE upserts them by `game_id` — but any `game_id` present in
  the old SVD version and **not** re-embedded keeps its old row.
- `game_neighbors` then runs `ML.DISTANCE` between an old-SVD source vector and a
  new-PCA candidate vector — **different vector spaces, silently wrong distances.**

`run-generate-embeddings` loops until every eligible game is embedded, so it
*mostly* converges — but any game dropped from the new model's scored universe
leaves a permanent stale row. A one-time full-refresh removes that class of bug.

## Steps

### 1. PR: merge `feat/embedding-input-scaling` → `main` (bgg-predictive-models)

Open the PR, Phil reviews and merges. Contents: the 6 commits on the branch
(TwoSDScaler/MinCountSelector, config swap, player-count features, diagnostic
fix, neighbor_check, Makefile repoint).

**Verify:** CI green; `uv run -m pytest tests/` locally (the 8 pre-existing
`test_transformers.py` / `test_preprocessor.py` failures are fixture schema drift,
present on `main` too — confirm the count doesn't grow).

### 2. Rebuild the embeddings service image

`docker-embeddings-build.yml` does **not** trigger on this merge (no
`services/game_embeddings/**` change). Dispatch it manually against `main` after
the merge so the running service can unpickle a pipeline containing
`PlayerCountSanitizer` / `TwoSDScaler` / `MinCountSelector`.

**Verify:** workflow green; `gcloud run services describe bgg-embeddings-service`
shows the new revision serving (or check the deploy step's output).

### 3. Sync the trained experiment to GCS

The prod embedding model is trained locally (`just embed-train` — there is no CI
training job for game embeddings) and the experiment dir is pushed to GCS via
`src.utils.sync_experiments`. Confirm the validated experiment
(`models/experiments/embeddings/game-embeddings/vN`) is the latest and is
uploaded.

> **Open question for Phil:** is local `make register_embeddings` the accepted
> path here, given the "Actions-only deploys" rule? There is no workflow for
> game-embedding registration. If it should be a workflow, that's a prerequisite
> sub-task.

**Verify:** `gsutil ls` the experiments bucket shows the new version;
`ExperimentTracker("embeddings").load_experiment("game-embeddings")` resolves to
it.

### 4. Register as `embeddings-v2026`

`make register_embeddings` (now reads `EMBEDDINGS_CANDIDATE = game-embeddings`).
Overwrites the registered model in place.

**Verify:** the registration output reports `algorithm: pca`, `embedding_dim: 64`,
and the expected experiment hash. Hit the service's model-info endpoint (or
restart + check logs) to confirm it loads the new model.

### 5. Regenerate all embeddings

`run-generate-embeddings.yml` via `workflow_dispatch` (input `model_name:
embeddings-v2026`). Batches through the full universe, then generates 2D
coordinates, then dispatches `embeddings_complete` to bgg-data-warehouse.

**Verify:** job summary shows `Total Games Embedded` ≈ the full catalog
(~120–128k) and `Games Remaining: 0`. In BQ:
`SELECT embedding_version, algorithm, COUNT(*) FROM raw.game_embeddings GROUP BY 1,2`
— newest version has `algorithm = 'pca'` and the expected row count.

### 6. Full-refresh the two incremental warehouse tables

The `embeddings_complete` dispatch (step 5) will have already run `dataform.yml`
**incrementally**. Follow it with a targeted full-refresh
(`reference_dataform_api.md`):

1. `POST .../compilationResults` `{"gitCommitish": "main"}`
2. `POST .../workflowInvocations` with `invocationConfig`:
   - `fullyRefreshIncrementalTablesEnabled: true`
   - `includedTargets`: `bgg_game_embeddings` and `game_similarity_search`, each
     `{database: "bgg-data-warehouse", schema: <predictions|analytics>, name: ...}`
   - `transitiveDependentsIncludedEnabled: true` (so `game_neighbors` rebuilds
     from the refreshed `game_similarity_search`)

> **Open question for Phil:** do this by hand via the API once, or add a
> `full_refresh` boolean input to `bgg-data-warehouse/.github/workflows/dataform.yml`
> (small PR) so it's repeatable and Actions-tracked? Recommend the workflow input.

**Verify:** in BQ —
`SELECT COUNT(DISTINCT embedding_version) FROM predictions.bgg_game_embeddings`
returns **1**; same for `analytics.game_similarity_search`; every
`game_neighbors.computed_ts` is post-refresh; `SELECT algorithm, COUNT(*) FROM
analytics.game_similarity_search GROUP BY 1` shows only `pca`.

### 7. Verify on the live product

bgg-viewer game pages for: System Gateway, The White Castle, Pandemic, Twilight
Imperium: Fourth Edition, Catan, Gloomhaven. Neighbour lists should match what
`neighbor_check` produced for `game-embeddings vN`.

**Verify:** lists are the new ones; no console/API errors; the "Similar games"
card renders for an upcoming (0-rating) game (the `source_min_users_rated: 0`
path).

## Risks / unknowns / rollback

- **In-place replacement, no staged rollout.** Mitigation: steps 1–4 are inert
  until step 5; step 5 output is inspectable before step 6; the live list only
  changes after step 6.
- **Cross-vector-space `game_neighbors`** if step 6 is skipped or mis-targeted —
  silently wrong neighbours. This is the main reason the plan exists. The verify
  query in step 6 (`COUNT(DISTINCT embedding_version) = 1`) is the gate.
- **Service can't unpickle** the new model if step 2 is skipped or the image
  doesn't `COPY src/` — step 4 verification catches it before any regen.
- **CI-vs-local model drift.** The spot-check ran on the locally-trained `vN`.
  If registration trains fresh anywhere, it won't be bit-identical — but
  `random_state=42` is pinned, so re-run `neighbor_check` against whatever gets
  registered as a final gate.
- **`embedding_8/16/32` for the tuned/live path.** `game_neighbors` default uses
  64-d and is unaffected; the truncated columns feed only custom similarity
  (`game_similarity_search` direct). Confirm the service still slices them
  (`main.py:374`) — it does as of this writing.
- **Rollback:** set `EMBEDDINGS_CANDIDATE` back / re-register the prior
  `svd-embeddings` experiment as `embeddings-v2026`, re-run
  `run-generate-embeddings`, full-refresh again. ~1 pipeline cycle. The old
  experiment dir and its GCS artifacts are retained.

## Out of scope

- The `game_neighbors` "quality" profile (family/reimplementation exclusion +
  rating percentile) — separate bgg-data-warehouse work, and it's additive (a new
  profile alongside `default`).
- The bgg-viewer `/dev/similar` bench — not needed for this promotion.
- Retraining the outcome models (complexity/rating/…); they consume the text
  embeddings, not this one.
- `config.yaml embeddings.use_embeddings` — stays `false` (structural embedding).
