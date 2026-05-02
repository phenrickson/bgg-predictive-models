# Collection Scoring Service — design

A new Cloud Run service that hosts deployed per-user collection models and scores
games against them on a daily schedule. Scoped to the `own` (classification)
outcome for v1; other outcomes follow the same shape later.

## Scope

In scope:

- New Cloud Run service `services/collections/` exposing a `/predict_own`
  endpoint that scores games for a single user against their registered `own`
  model.
- New BigQuery landing table for collection predictions, append-only, keyed by
  `(username, game_id, model_version)`.
- New BigQuery registry table that records which user/outcome models are
  deployed and serves as the driver for the daily scoring job.
- A GitHub Actions workflow that triggers `/predict_own` daily for every active
  registry entry, with server-side change detection.

Out of scope (separate plans):

- Other outcomes (`ever_owned`, `rated`, `rating`, `love`).
- The training/finalize/promote workflow that *populates* the registry.
- The monthly retraining job that creates new model versions in response to
  collection changes.

## Architecture

```
┌─────────────────────────────────────┐
│ GitHub Action (daily 8 AM UTC)      │
│  - reads registry (status=active)   │
│  - loops users, POSTs /predict_own  │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐    ┌──────────────────────────┐
│ services/collections/ (Cloud Run)   │───▶│ GCS                      │
│  /predict_own                        │    │  registered models       │
│  - resolves user → registry entry   │    │  collections/{user}/own/ │
│  - loads RegisteredCollectionModel  │    │    v{N}/{pipeline.pkl,   │
│  - change detection vs landing tbl  │    │           threshold.json}│
│  - pulls features from BQ           │    └──────────────────────────┘
│  - scores, returns + uploads        │
└────┬─────────────────────┬──────────┘
     │                     │
     ▼                     ▼
┌──────────────┐    ┌──────────────────────────────────────┐
│ BQ features  │    │ BQ landing                           │
│ games_features│    │ raw.collection_predictions_landing  │
└──────────────┘    │  (append-only, history preserved)   │
                    └──────────────────────────────────────┘
                              ▲
                              │
                    ┌──────────────────────┐
                    │ BQ registry          │
                    │ raw.collection_models│
                    │ _registry            │
                    └──────────────────────┘
```

The service is one Cloud Run service that hosts every deployed user model. It
loads pipelines lazily on first request per `(username, outcome, version)` and
caches them in process. This mirrors how `services/scoring/` keys its model
cache by `(model_type, name, version)` — the cache key gains a `username`
dimension and otherwise behaves the same.

### Why a separate service from `services/scoring/`

- The path layout already exists at `gs://.../{env}/services/collections/...`,
  parallel to `gs://.../{env}/models/registered/...`. The split was anticipated.
- Different cache shape (per-user, unbounded by user count) and different
  deploy cadence (collections module is moving fast on a feature branch).
- Independent Cloud Run sizing — game scoring runs heavy (Bayesian
  simulation), collection scoring is single-model `predict_proba`.

## Components

### 1. FastAPI service (`services/collections/main.py`)

Endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check with auth status. |
| `/predict_own` | POST | Score `own` for one user. |
| `/models` | GET | List registry entries. |
| `/model/{username}/{outcome}/info` | GET | Latest registered version + metadata for one user/outcome. |

`/predict_own` request body:

```json
{
  "username": "phenrickson",
  "game_ids": [1, 2, 3],            // optional; omit with use_change_detection
  "use_change_detection": true,     // when true, server computes unscored set
  "upload_to_data_warehouse": true,
  "model_version": null             // optional pin; null = latest active
}
```

Response shape mirrors `/predict_games`: a list of prediction rows plus a
metadata block (`model_name`, `model_version`, `n_scored`, `score_ts`).

### 2. Registry table (`bgg-predictive-models.raw.collection_models_registry`)

| Column | Type | Notes |
|--------|------|-------|
| `username` | STRING | BGG username. |
| `outcome` | STRING | `own` for v1. |
| `model_version` | INT64 | Monotonic per `(username, outcome)`. |
| `finalize_through_year` | INT64 | Year cutoff used in finalize step. |
| `gcs_path` | STRING | Full `gs://…/v{N}/` of the deployed pipeline. |
| `registered_at` | TIMESTAMP | When the row was inserted. |
| `status` | STRING | `active` \| `inactive`. Daily job filters on `active`. |

Primary lookup pattern: latest `active` row per `(username, outcome)`.
Insert-only is fine — version bumps insert a new row; demoting a model flips
`status` on the old row to `inactive` (handled by the future
register/promote workflow, out of scope here).

### 3. Landing table (`bgg-predictive-models.raw.collection_predictions_landing`)

Append-only. Partitioned by `score_ts`, clustered by `(username, game_id)`.

| Column | Type |
|--------|------|
| `job_id` | STRING |
| `username` | STRING |
| `game_id` | INT64 |
| `outcome` | STRING |
| `predicted_prob` | FLOAT64 |
| `predicted_label` | BOOL |
| `threshold` | FLOAT64 |
| `model_name` | STRING |
| `model_version` | INT64 |
| `score_ts` | TIMESTAMP |

`job_id` is generated server-side per `/predict_own` call (UUID4) and is shared
across every row produced by that call. Matches the convention used by
`ml_predictions_landing` so all rows from one scoring run can be joined back to
a single invocation.

History is preserved across model versions: a v2 model produces new rows
without touching v1 rows. Downstream Dataform views can deduplicate to
"latest version per `(username, game_id)`" if needed.

### 4. Change detection

When `use_change_detection=true`, the endpoint:

1. Resolves `(username, outcome) → (model_version, gcs_path)` from the
   registry.
2. Queries the landing table for `game_id`s already scored under that exact
   `(username, model_version)`.
3. Pulls the universe of game_ids from `games_features` filtered to the
   service-config year range (matches `/predict_games` behavior).
4. Scores the difference.

If `game_ids` is provided explicitly, change detection is skipped and the
endpoint scores exactly that list.

### 5. Daily GitHub Action (`.github/workflows/run-collection-scoring.yml`)

- Trigger: cron `0 8 * * *` (08:00 UTC, one hour after `run-scoring-service.yml`).
- Also `workflow_dispatch` for manual runs.
- Steps mirror `run-scoring-service.yml`:
  - Authenticate to GCP.
  - Resolve Cloud Run service URL + ID token.
  - Query the registry for active rows.
  - Loop users, POST `/predict_own` with `use_change_detection=true` and
    `upload_to_data_warehouse=true`.
  - Collect per-user `n_scored` into the job summary.

Continue-on-error per user; the workflow exits non-zero if any user failed
but reports the rest. Same shape as the existing scoring action.

## Deployment

- Dockerfile at `docker/collections.Dockerfile`, built off the same Python
  3.12 + UV base as `docker/scoring.Dockerfile`.
- `services/collections/cloudbuild.yaml` builds the image, pushes to Artifact
  Registry, deploys to Cloud Run service `bgg-collection-scoring` in
  `us-central1`. Memory/CPU starts at 4GB / 2 CPU and adjusts based on
  observed model size and concurrency (single-model `predict_proba` is light).

## Error handling

- Registry miss (no active row for `(username, outcome)`): `404` with a clear
  message. Daily job logs and continues.
- GCS load failure: `502`. Daily job logs and continues.
- BQ feature query failure: `502`. Daily job logs and continues.
- Empty unscored set under change detection: `200` with `n_scored: 0` and an
  empty predictions list. This is the steady state for already-caught-up
  users.

## Testing

- Unit: `RegisteredCollectionModel` cache keying, change-detection SQL
  builder, request/response schema validation.
- Integration: end-to-end `/predict_own` against a fixture user with a
  pre-staged GCS pipeline and seeded landing table; assert exactly the
  expected unscored game_ids are scored and the appended rows match.
- The GitHub Action is exercised by `workflow_dispatch` against a dev
  environment before the cron is enabled in prod.

## Open questions

None blocking the plan.
