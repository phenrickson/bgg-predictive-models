#!/usr/bin/env bash
# Cloud Run job command for collection model training. The gs:// sync
# boundary kept OUT of train_model.py (which stays pure/testable):
# pull the user's artifact tree, run the in-process lifecycle, push the
# results back. Per-execution args arrive as env overrides.
#
# The `collections` image has google-cloud-storage but not the gcloud
# CLI tooling, so the gs:// boundary uses src.collection.gcs_sync (a
# Python prefix sync) instead of a shell rsync. Semantics preserved:
# the down-sync treats an empty/absent prefix as first-run (returns 0)
# and propagates real client errors, so a bare call under `set -e` is
# correct without an `if` guard; the up-sync hard-fails so a trained
# model that can't reach GCS aborts the job.
set -euo pipefail

: "${TRAIN_USERNAME:?TRAIN_USERNAME required}"
TRAIN_OUTCOME="${TRAIN_OUTCOME:-own}"
TRAIN_CANDIDATE="${TRAIN_CANDIDATE:-logistic_row_norm}"
: "${ENVIRONMENT:?ENVIRONMENT required}"
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID required}"

LOCAL_ROOT="/app/models/collections"
USER_LOCAL="${LOCAL_ROOT}/${TRAIN_USERNAME}"
# Bucket name == project id by convention in this project.
USER_PREFIX="${ENVIRONMENT}/collections/${TRAIN_USERNAME}"

mkdir -p "${USER_LOCAL}"

echo "Pulling gs://${GCP_PROJECT_ID}/${USER_PREFIX} -> ${USER_LOCAL}"
uv run python -m src.collection.gcs_sync download \
  --bucket "${GCP_PROJECT_ID}" \
  --prefix "${USER_PREFIX}" \
  --local-dir "${USER_LOCAL}"

echo "Running train_model for ${TRAIN_USERNAME}/${TRAIN_OUTCOME}/${TRAIN_CANDIDATE}"
uv run python -m src.collection.train_model \
  --username "${TRAIN_USERNAME}" \
  --outcome "${TRAIN_OUTCOME}" \
  --candidate "${TRAIN_CANDIDATE}" \
  --environment "${ENVIRONMENT}" \
  --local-root "${LOCAL_ROOT}"

echo "Pushing ${USER_LOCAL} -> gs://${GCP_PROJECT_ID}/${USER_PREFIX}"
uv run python -m src.collection.gcs_sync up \
  --bucket "${GCP_PROJECT_ID}" \
  --prefix "${USER_PREFIX}" \
  --local-dir "${USER_LOCAL}"

echo "collection-training done: ${TRAIN_USERNAME}/${TRAIN_OUTCOME}/${TRAIN_CANDIDATE}"
