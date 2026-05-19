# Train Collection Model — Design

**Date:** 2026-05-19
**Status:** Draft

## Goal

One operation: given a username, produce a **deployed, registered
collection model** for that user — closing the loop from raw data →
split → train → finalize → register, and (via the existing promote →
model-report seam) surfacing it on the reports site.

Concretely, after running for `Gyges`:

- A finalized `logistic_row_norm` model exists for `Gyges/own` in
  `gs://<bucket>/<env>/collections/Gyges/own/...`.
- It is registered: a row in `raw.collection_models_registry` and the
  artifact in GCS where the scoring service reads it.
- The model report for `Gyges` is (re)rendered.

## Non-goals

- Multi-candidate sweeps. This trains exactly one candidate
  (`logistic_row_norm`). Sweep/compare is out of scope.
- Refactoring `run-training.yml` (BGG game-rating models — a different
  pipeline; a known future refactor, not touched here).
- A general orchestration framework. One user, one candidate, one
  outcome (`own`) by default.
- Running training compute on the GitHub Actions runner. The runner
  only triggers; compute runs in GCP.
- Terraform-managing the Cloud Run job. This project provisions Cloud
  Run via `gcloud` in workflows (see `docker-collections-build.yml`);
  the job follows that established pattern, not terraform.

## Background — established patterns this builds on

- **Collection lifecycle modules already exist** and operate on a local
  artifact tree via `CollectionArtifactStorage(local_root, environment)`
  (local filesystem only — GCS sync is a *separate* concern, by
  design):
  - `src.collection.split` — persists canonical train/val/test splits
  - `src.collection.train` — trains one candidate
  - `src.collection.finalize` — refits on train+val+test through
    `finalize_through`
  - `services.collections.register_model` — pushes the finalized model
    to GCS + inserts the `raw.collection_models_registry` row. CLI:
    `--username --outcome --candidate --description --version
    --environment --local-root`.
- **GCS artifact sync is a prefix mirror** between
  `models/collections/<user>/` and
  `gs://<bucket>/<env>/collections/<user>/` (restore → work locally →
  mirror back). The reports pipeline does this with `gsutil` *on the
  GitHub runner*, but this job runs the sync *inside the `collections`
  image*, which has `google-cloud-storage` but no `gsutil`. So this
  job uses a Python `google-cloud-storage` prefix sync
  (`src/collection/gcs_sync.py`: `download_prefix` / `upload_prefix`,
  unit-tested with a mocked storage client) instead of `gsutil rsync`.
  Same restore→work→mirror shape, different (in-image) mechanism.
- **Cloud Run is gcloud-provisioned in workflows.**
  `docker-collections-build.yml` builds the `collections` image and
  `gcloud run deploy bgg-collection-scoring` with: image tag
  `<env>-<sha>`, `--service-account
  bgg-predictive-models@<project>.iam.gserviceaccount.com`,
  `--set-env-vars ENVIRONMENT=<env>,GCP_PROJECT_ID=<project>,
  GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json`,
  `--timeout 1800`. The training job mirrors this (same image, SA, env
  convention) but as a Cloud Run **job**, not a service.
- **The `collections` image already `COPY src/`** (Dockerfile line 34)
  and `WORKDIR /app`; `CMD` is the uvicorn service — a Cloud Run job
  overrides the command/args.

## Design decisions (settled in brainstorming)

1. **Single in-process entrypoint, not four decoupled jobs.** A new
   `src/collection/train_model.py` runs split → train → finalize →
   register in one process against one local working dir. Rationale:
   for one user / one candidate the whole run is minutes; decoupled
   jobs would pay heavy complexity (4× gs:// sync round-trips, gs:// as
   an inter-job message bus) to buy stage-level re-run granularity not
   needed at this scale. The split→train→finalize handoff stays
   in-process; gs:// is a boundary concern (initial input / final
   output), not an inter-stage protocol.
2. **Compute runs as a Cloud Run job in GCP, not on the runner.** The
   GitHub workflow is a thin trigger: authenticate → launch the Cloud
   Run job → wait → surface result. Model training is real ML compute;
   the runner is an orchestrator, consistent with how the scoring
   service is operated.
3. **`docker-collections-build` gains a `src/collection/**` path
   trigger.** Today it only rebuilds on `services/collections/**`, so
   `src/` inside `collections:prod` is frozen relative to collection
   lifecycle code (the same frozen-`src/` class of bug hit with the
   reports image). Adding `src/collection/**` makes a training-code
   change rebuild the image so the job runs current code.
   **Accepted coupling:** that workflow also `gcloud run deploy`s the
   scoring service, so a training-code change redeploys
   `bgg-collection-scoring`. This is acceptable here: scoring runs on a
   daily cron (wide window to catch regressions), the project is low
   stakes, and revert+redeploy is fast. Mitigation is *visibility*
   (know when the scoring image changed, be able to roll back), not
   avoidance. The simplicity of one collections image/trigger outweighs
   isolating training from the scoring deploy.
4. **`train_model.py` is pure lifecycle; gs:// sync and report dispatch
   are around it, not in it.** Keeps the module unit-testable off-CI
   (the slow-CI-loop lesson) and the gs:// dependency at the boundary.

## Architecture

Three new pieces + one trigger change.

### 1. `src/collection/train_model.py` (new)

A single CLI entrypoint:

```bash
uv run python -m src.collection.train_model \
  --username Gyges --outcome own --candidate logistic_row_norm \
  --environment <env> --local-root models/collections
```

Behavior, in-process, in order, against the one `--local-root` tree:

1. `split` — same effect as `src.collection.split` for the user/outcome
2. `train` — train `--candidate` against the latest canonical splits
3. `finalize` — finalize `--candidate` (default `finalize_through` from
   `config.collections.finalize_through`)
4. `register` — invoke the `services.collections.register_model` logic
   (GCS push + `raw.collection_models_registry` row), with a default
   `--description` of `"<candidate> for <user>/<outcome>"` matching the
   justfile recipe's default.

It reuses the existing lifecycle logic in-process — not `just`, not
the GitHub runner. Whether that means calling already-exposed
functions or first lifting argparse `main()` bodies into callable
functions is an implementation-plan detail (the existing modules are
CLI entrypoints; the plan determines the cleanest reuse without
duplicating logic). Args mirror the existing CLIs:
`--username` (required), `--outcome` (default `own`), `--candidate`
(default `logistic_row_norm`), `--environment` (default `dev`),
`--local-root` (default `models/collections`), `--finalize-through`
(optional; default from config). Exit non-zero on any stage failure,
with a clear log line naming the failed stage.

It does **not** call `gsutil` and does **not** dispatch the report.

### 2. Cloud Run job `collection-training` (new GCP resource)

Created/updated via `gcloud run jobs` in the trigger workflow (same
gcloud-in-workflow pattern as the scoring service; not terraform).

- Image: `collections:<env>` (the existing collections image, post the
  `src/collection/**` trigger change so it carries current lifecycle
  code).
- Service account:
  `bgg-predictive-models@<project>.iam.gserviceaccount.com` (same as
  scoring).
- Env: `ENVIRONMENT=<env>`, `GCP_PROJECT_ID=<project>`,
  `GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json`
  (same convention as the scoring deploy).
- The job's container command runs a small wrapper that:
  1. `uv run python -m src.collection.gcs_sync download --bucket
     <project> --prefix <env>/collections/<user> --local-dir
     /app/models/collections/<user>` (pull prior state for the user;
     an empty/absent prefix returns 0 = first run, real client errors
     propagate and abort the job under `set -e`)
  2. `uv run python -m src.collection.train_model --username <user>
     --outcome <outcome> --candidate <candidate> --environment <env>
     --local-root /app/models/collections`
  3. `uv run python -m src.collection.gcs_sync up --bucket <project>
     --prefix <env>/collections/<user> --local-dir
     /app/models/collections/<user>` (push results; hard-fails on
     error so a trained model that can't reach GCS aborts the job)
- Per-execution args (`<user>`, `<outcome>`, `<candidate>`) supplied as
  Cloud Run job execution overrides.
- `--timeout` and resources sized for training (start from the scoring
  service's `--memory 4Gi --cpu 2 --timeout 1800`; revisit if a real
  run shows it needs more).

### 3. `train-collection-model.yml` (new workflow — thin trigger)

- Trigger: `workflow_dispatch` with input `username` (required); the
  trigger model is explicitly deferred for later discussion — username
  input only for now. `outcome` defaults to `own`, `candidate` to
  `logistic_row_norm` (not exposed as inputs yet).
- Steps (mirroring `docker-collections-build.yml`'s auth pattern):
  1. `google-github-actions/auth@v2` + `setup-gcloud@v2` with
     `secrets.GCP_SA_KEY_BGG_ML`
  2. Set `env_name` (`prod` on `main`, else `dev`)
  3. `gcloud run jobs create ... || gcloud run jobs update ...` (idempotent
     ensure the `collection-training` job exists with current image/SA/env)
  4. `gcloud run jobs execute collection-training --wait
     --args/--update-env` to run for the given username, waiting for
     completion
  5. On success, dispatch `build-model-reports.yml -f users=<username>
     -f outcome=own` (the report seam — placed in the workflow, not in
     `train_model.py` or `register_model`, keeping the entrypoint pure;
     mirrors what the justfile `promote` recipe did, just moved to the
     orchestration layer)

### Report-dispatch seam placement (resolved)

The justfile `promote` recipe dispatched `build-model-reports.yml`
after `register_model`. In this design that seam lives in the **trigger
workflow** (step 5 above), not in `train_model.py` (kept pure) and not
in `register_model` (a library entrypoint shouldn't fire CI). The
workflow is the orchestration layer; the report dispatch is
orchestration.

## Data flow

`gs://<project>/<env>/collections/<user>/` is the durable store. The
Cloud Run job pulls the user's tree down into the container, runs the
entire split→train→finalize→register lifecycle in-process against that
local tree, and pushes the tree back. No gs:// traffic between
lifecycle stages — that handoff is in-process. `register_model` writes
the BQ registry row from inside the job (it has the SA + env).

## Error handling

- `train_model.py` exits non-zero on the first failing stage, logging
  which stage failed. The Cloud Run job execution then fails.
- The trigger workflow waits on the job (`gcloud run jobs execute
  --wait`); a failed job → failed workflow step → no report dispatch
  (step 5 guarded on prior success).
- gs:// pull tolerates an absent/empty user prefix (first run). gs://
  push runs only after a successful `train_model.py` (so a failed run
  doesn't half-overwrite the durable tree); a failed run leaves the
  prior gs:// state intact.
- A failed run is fully re-runnable by re-dispatching the workflow for
  the same username (idempotent: split/train/finalize/register
  overwrite per-user artifacts; the registry insert is the one
  non-idempotent effect — acceptable, it versions).

## Testing

- `src/collection/train_model.py` is unit-testable off-CI: a test that
  runs it against a fixture user / fixture artifact tree with the
  BQ/GCS effects of `register_model` mocked, asserting it calls split →
  train → finalize → register in order and exits non-zero when a stage
  raises. (Mirrors the existing `tests/` patterns for collection
  modules.)
- The Cloud Run job + workflow are validated by a real dispatch for
  `Gyges` (the proof-of-functionality test) — there is no way to fully
  validate cloud-job execution locally; the deliberate first run is the
  test, watched end-to-end.

## Out of scope / explicitly deferred

- The trigger model beyond `workflow_dispatch`+username (cron?
  all-users? promotion-driven?) — flagged for a later, separate
  decision.
- Multi-candidate / sweep support.
- Resource tuning beyond a sensible starting point (4Gi/2cpu/1800s).
- `run-training.yml` refactor.
