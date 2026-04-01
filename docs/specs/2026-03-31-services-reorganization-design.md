# Services Reorganization Design

## Goal

Consolidate the three standalone service directories at the repo root (`scoring_service/`, `embeddings_service/`, `text_embeddings_service/`) into a single `services/` directory for clearer project organization.

## Structure

```
services/
├── scoring/            (from scoring_service/)
├── game_embeddings/    (from embeddings_service/, renamed)
└── text_embeddings/    (from text_embeddings_service/)
```

Each service retains its existing internal structure. No shared module extraction or code changes beyond path updates.

## Scope of Changes

### Directory moves
- `scoring_service/` -> `services/scoring/`
- `embeddings_service/` -> `services/game_embeddings/`
- `text_embeddings_service/` -> `services/text_embeddings/`

### Files requiring path updates

**Makefile** — All targets referencing service directories:
- Docker build contexts and Dockerfile paths
- Service start/stop/run targets
- Any `cd` or path references to the old directories

**Dockerfiles** (`docker/`):
- `scoring.Dockerfile` — COPY paths for scoring service
- `embeddings.Dockerfile` — COPY paths for embeddings service
- `text_embeddings.Dockerfile` — COPY paths for text embeddings service

**GitHub Actions workflows** (`.github/workflows/`):
- `docker-scoring-build.yml`
- `docker-embeddings-build.yml`
- `docker-text-embeddings-build.yml`
- `run-scoring-service.yml`
- `run-generate-embeddings.yml`
- `run-generate-text-embeddings.yml`
- `release-docker-images.yml`

**Cloud Build**:
- `scoring_service/cloudbuild.yaml` moves to `services/scoring/cloudbuild.yaml`

**Imports and references**:
- Any Python imports referencing old module names (e.g., `from scoring_service import ...`)
- README.md references to service paths
- `scoring_service/README.md` moves to `services/scoring/README.md`

### Rename: embeddings -> game_embeddings

In addition to the directory move, `embeddings_service/` is renamed to `game_embeddings/` under `services/`. This requires updating:
- Dockerfile references that mention `embeddings_service` specifically
- Workflow files that reference `embeddings_service`
- Any imports using the `embeddings_service` module name

## Out of Scope

- Extracting shared code (auth.py, registered_model.py patterns) into a common module
- Changing service internals or APIs
- Modifying Terraform configuration (Cloud Run service names stay the same)
- Changing Docker image names or tags

## Verification

- All Makefile service targets work with new paths
- Docker builds succeed with updated contexts
- Tests pass
- Linting passes
