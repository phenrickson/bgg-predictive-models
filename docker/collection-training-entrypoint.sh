#!/usr/bin/env bash
# Cloud Run job command for collection model training. The gs:// sync
# boundary kept OUT of train_model.py (which stays pure/testable):
# pull the user's artifact tree, run the in-process lifecycle, push the
# results back. Per-execution args arrive as env overrides.
set -euo pipefail

: "${TRAIN_USERNAME:?TRAIN_USERNAME required}"
TRAIN_OUTCOME="${TRAIN_OUTCOME:-own}"
TRAIN_CANDIDATE="${TRAIN_CANDIDATE:-logistic_row_norm}"
: "${ENVIRONMENT:?ENVIRONMENT required}"
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID required}"

LOCAL_ROOT="/app/models/collections"
USER_LOCAL="${LOCAL_ROOT}/${TRAIN_USERNAME}"
USER_GS="gs://${GCP_PROJECT_ID}/${ENVIRONMENT}/collections/${TRAIN_USERNAME}"

mkdir -p "${USER_LOCAL}"

echo "Pulling ${USER_GS} -> ${USER_LOCAL}"
gsutil -m rsync -r "${USER_GS}" "${USER_LOCAL}" 2>&1 \
  || echo "(no prior artifacts for ${TRAIN_USERNAME}; first run)"

echo "Running train_model for ${TRAIN_USERNAME}/${TRAIN_OUTCOME}/${TRAIN_CANDIDATE}"
uv run python -m src.collection.train_model \
  --username "${TRAIN_USERNAME}" \
  --outcome "${TRAIN_OUTCOME}" \
  --candidate "${TRAIN_CANDIDATE}" \
  --environment "${ENVIRONMENT}" \
  --local-root "${LOCAL_ROOT}"

echo "Pushing ${USER_LOCAL} -> ${USER_GS}"
gsutil -m rsync -r "${USER_LOCAL}" "${USER_GS}"

echo "collection-training done: ${TRAIN_USERNAME}/${TRAIN_OUTCOME}/${TRAIN_CANDIDATE}"
